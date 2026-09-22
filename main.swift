import Foundation
import Metal

// Pack float -> OCP FP8 E4M3FN (1/4/3, bias 7). Max finite ~448.
@inline(__always)
func floatToE4M3(_ x: Float) -> UInt8 {
    if x.isNaN { return 0x7F }
    let sign: UInt8 = x < 0 ? 0x80 : 0
    let ax = abs(x)
    if ax == 0 { return sign }
    if ax >= 448 { return sign | 0x7E }
    let bits = ax.bitPattern
    let exp = Int((bits >> 23) & 0xFF) - 127
    var mant = bits & 0x7FFFFF
    var e = exp + 7
    if e <= 0 { return sign }
    if e >= 15 { return sign | 0x7E }
    let shift = 23 - 3
    let rounding = (mant >> (shift - 1)) & 1
    var m3 = (mant >> shift) + rounding
    if m3 == 8 {
        m3 = 0
        e += 1
        if e >= 15 { return sign | 0x7E }
    }
    return sign | UInt8((e << 3) | Int(m3))
}

struct Args {
    var N = 4096
    var K = 4096
    var M = 4096
    var iters = 50
    var warmup = 5
}

func parseArgs() -> Args {
    var a = Args()
    let argv = CommandLine.arguments
    var i = 1
    while i < argv.count {
        let key = argv[i]
        i += 1
        guard i < argv.count else { break }
        let val = Int(argv[i]) ?? 0
        switch key {
        case "--N": a.N = val
        case "--K": a.K = val
        case "--M": a.M = val
        case "--iters": a.iters = val
        case "--warmup": a.warmup = val
        default: break
        }
        i += 1
    }
    return a
}

guard let device = MTLCreateSystemDefaultDevice() else {
    fputs("No Metal device\n", stderr)
    exit(1)
}

let args = parseArgs()
print("device: \(device.name)")
print("shape: N=\(args.N) K=\(args.K) M=\(args.M)  (Y = X[N,K] @ W[K,M])")
print("dtype: X=half  W=fp8_e4m3  Y=float")

let cwd = FileManager.default.currentDirectoryPath
let libURL = URL(fileURLWithPath: cwd).appendingPathComponent("default.metallib")
let srcURL = URL(fileURLWithPath: cwd).appendingPathComponent("fp8_mlp.metal")

let library: MTLLibrary
if FileManager.default.fileExists(atPath: libURL.path) {
    do {
        library = try device.makeLibrary(URL: libURL)
        print("loaded: \(libURL.path)")
    } catch {
        fputs("Failed loading metallib: \(error)\n", stderr)
        exit(2)
    }
} else if let src = try? String(contentsOf: srcURL, encoding: .utf8) {
    let options = MTLCompileOptions()
    do {
        library = try device.makeLibrary(source: src, options: options)
        print("compiled from source: \(srcURL.path)")
    } catch {
        fputs("Metal compile failed (need Metal 4.1 + FP8 matmul):\n\(error)\n", stderr)
        exit(2)
    }
} else {
    fputs("Need default.metallib or fp8_mlp.metal in \(cwd)\n", stderr)
    exit(1)
}

guard let fn = library.makeFunction(name: "fp8_mlp_gemm") else {
    fputs("Missing kernel fp8_mlp_gemm\n", stderr)
    exit(1)
}

let pipeline: MTLComputePipelineState
do {
    pipeline = try device.makeComputePipelineState(function: fn)
} catch {
    fputs("Pipeline failed: \(error)\n", stderr)
    exit(1)
}

let N = args.N, K = args.K, M = args.M
let xCount = N * K
let wCount = K * M
let yCount = N * M

var xHost = [Float16](repeating: 0, count: xCount)
var wHost = [UInt8](repeating: 0, count: wCount)

for i in 0..<xCount {
    xHost[i] = Float16(Float((i % 97) - 48) * 0.02)
}
for i in 0..<wCount {
    wHost[i] = floatToE4M3(Float((i % 53) - 26) * 0.05)
}

guard let xBuf = device.makeBuffer(bytes: &xHost, length: xCount * 2, options: .storageModeShared),
      let wBuf = device.makeBuffer(bytes: &wHost, length: wCount, options: .storageModeShared),
      let yBuf = device.makeBuffer(length: yCount * 4, options: .storageModeShared),
      let queue = device.makeCommandQueue()
else {
    fputs("Buffer/queue alloc failed\n", stderr)
    exit(1)
}

func encodeOnce(_ cb: MTLCommandBuffer) {
    guard let enc = cb.makeComputeCommandEncoder() else { return }
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(xBuf, offset: 0, index: 0)
    enc.setBuffer(wBuf, offset: 0, index: 1)
    enc.setBuffer(yBuf, offset: 0, index: 2)
    var n32 = Int32(N), k32 = Int32(K), m32 = Int32(M)
    enc.setBytes(&n32, length: 4, index: 3)
    enc.setBytes(&k32, length: 4, index: 4)
    enc.setBytes(&m32, length: 4, index: 5)
    let tgM = 64, tgN = 32
    let grid = MTLSize(width: (M + tgN - 1) / tgN,
                       height: (N + tgM - 1) / tgM,
                       depth: 1)
    let threads = MTLSize(width: 128, height: 1, depth: 1)
    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: threads)
    enc.endEncoding()
}

for _ in 0..<args.warmup {
    guard let cb = queue.makeCommandBuffer() else { continue }
    encodeOnce(cb)
    cb.commit()
    cb.waitUntilCompleted()
}

var times = [Double]()
times.reserveCapacity(args.iters)
for _ in 0..<args.iters {
    guard let cb = queue.makeCommandBuffer() else { continue }
    encodeOnce(cb)
    let t0 = CFAbsoluteTimeGetCurrent()
    cb.commit()
    cb.waitUntilCompleted()
    times.append(CFAbsoluteTimeGetCurrent() - t0)
}

times.sort()
let median = times[times.count / 2]
let mean = times.reduce(0, +) / Double(times.count)
let flops = 2.0 * Double(N) * Double(K) * Double(M)

let yPtr = yBuf.contents().bindMemory(to: Float.self, capacity: yCount)
var checksum: Float = 0
let step = max(1, yCount / 4096)
for i in stride(from: 0, to: yCount, by: step) { checksum += yPtr[i] }

print(String(format: "median_ms: %.4f", median * 1000))
print(String(format: "mean_ms:   %.4f", mean * 1000))
print(String(format: "tflops_median: %.3f", (flops / median) / 1e12))
print(String(format: "tflops_mean:   %.3f", (flops / mean) / 1e12))
print(String(format: "checksum: %.6f", checksum))
print("ok")
