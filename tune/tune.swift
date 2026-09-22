import Foundation
import Metal

// Sweep host for tune.metal. Kernel names carry their tile geometry; see tune.metal.
// usage: tune --metallib path [--N n --K k --M m --iters i --filter substr[,substr]]

func floatToE4M3(_ x: Float) -> UInt8 {
    if x.isNaN { return 0x7F }
    let sign: UInt8 = x < 0 ? 0x80 : 0
    let ax = abs(x)
    if ax == 0 { return sign }
    if ax >= 448 { return sign | 0x7E }
    let bits = ax.bitPattern
    var e = Int((bits >> 23) & 0xFF) - 127 + 7
    let mant = bits & 0x7FFFFF
    if e <= 0 { return sign }
    var m3 = (mant >> 20) + ((mant >> 19) & 1)
    if m3 == 8 { m3 = 0; e += 1 }
    if e >= 15 { return sign | 0x7E }
    return sign | UInt8((e << 3) | Int(m3))
}

func e4m3ToFloat(_ b: UInt8) -> Float {
    let sign: Float = (b & 0x80) != 0 ? -1 : 1
    let exp = Int((b >> 3) & 0x0F), mant = Int(b & 0x07)
    if exp == 0 { return sign * Float(mant) * exp2(-9) }
    return sign * (1 + Float(mant) / 8) * exp2(Float(exp - 7))
}

let e2m1Mag: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
let e2m1Codes: [UInt8] = [0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x9, 0xA, 0xC]
func e2m1ToFloat(_ n: UInt8) -> Float { ((n & 0x8) != 0 ? -1 : 1) * e2m1Mag[Int(n & 0x7)] }
func packE2M1(_ count: Int, _ code: (Int) -> UInt8) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: count / 2)
    for i in 0..<count { bytes[i / 2] |= (code(i) & 0xF) << (i % 2 == 0 ? 0 : 4) }
    return bytes
}
func e2m1At(_ b: [UInt8], _ i: Int) -> Float { e2m1ToFloat(i % 2 == 0 ? b[i / 2] & 0xF : b[i / 2] >> 4) }

struct Cfg {
    var name: String
    var dt: String
    var tileM: Int   // rows of Y per threadgroup (N direction)
    var tileN: Int   // cols of Y per threadgroup (M direction)
    var threads: Int
    var bk: Int
}

func parse(_ name: String) -> Cfg? {
    let p = name.split(separator: "_").map(String.init)
    guard p.count >= 3 else { return nil }
    func pair(_ s: String) -> (Int, Int)? {
        let q = s.dropFirst().split(separator: "x").compactMap { Int($0) }
        return q.count == 2 ? (q[0], q[1]) : nil
    }
    switch p[0] {
    case "tg":
        guard let b = pair(p[2]), let s = Int(p[3].dropFirst()) else { return nil }
        return Cfg(name: name, dt: p[1], tileM: b.0, tileN: b.1, threads: 32 * s, bk: 1)
    case "sg":
        guard let f = pair(p[2]), let k = Int(p[3].dropFirst()), let w = pair(p[4]) else { return nil }
        return Cfg(name: name, dt: p[1], tileM: f.0 * w.0, tileN: f.1 * w.1, threads: 32 * w.0 * w.1, bk: k)
    case "alu":
        guard let f = pair(p[2]), let w = pair(p[3]) else { return nil }
        return Cfg(name: name, dt: p[1], tileM: f.0 * w.0, tileN: f.1 * w.1, threads: 32 * w.0 * w.1, bk: 8)
    default:
        return nil
    }
}

var N = 4096, K = 4096, M = 4096, iters = 50, warmup = 5
var filters: [String] = []
var libPath = "tune.metallib"
var argv = Array(CommandLine.arguments.dropFirst())
while argv.count >= 2 {
    let k = argv.removeFirst(), v = argv.removeFirst()
    switch k {
    case "--N": N = Int(v)!
    case "--K": K = Int(v)!
    case "--M": M = Int(v)!
    case "--iters": iters = Int(v)!
    case "--warmup": warmup = Int(v)!
    case "--filter": filters = v.split(separator: ",").map(String.init)
    case "--metallib": libPath = v
    default: break
    }
}

let device = MTLCreateSystemDefaultDevice()!
let lib = try! device.makeLibrary(URL: URL(fileURLWithPath: libPath))
let queue = device.makeCommandQueue()!
print("device: \(device.name)  N=\(N) K=\(K) M=\(M) iters=\(iters)")

let xc = N * K, wc = K * M, yc = N * M
var xH = [Float16](repeating: 0, count: xc), xI8 = [Int8](repeating: 0, count: xc), xF8 = [UInt8](repeating: 0, count: xc)
var wH = [Float16](repeating: 0, count: wc), wI8 = [Int8](repeating: 0, count: wc), wF8 = [UInt8](repeating: 0, count: wc)
for i in 0..<xc {
    let f = Float((i % 97) - 48) * 0.02
    xH[i] = Float16(f); xI8[i] = Int8((i % 97) - 48); xF8[i] = floatToE4M3(f)
}
for i in 0..<wc {
    let f = Float((i % 53) - 26) * 0.05
    wH[i] = Float16(f); wI8[i] = Int8((i % 53) - 26); wF8[i] = floatToE4M3(f)
}
let xF4 = packE2M1(xc) { e2m1Codes[$0 % e2m1Codes.count] }
let wF4 = packE2M1(wc) { e2m1Codes[($0 + 3) % e2m1Codes.count] }

func buf<T>(_ a: [T]) -> MTLBuffer { a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! } }
let bufs: [String: (MTLBuffer, MTLBuffer)] = [
    "fp16": (buf(xH), buf(wH)), "fp16h": (buf(xH), buf(wH)), "f8f8h": (buf(xF8), buf(wF8)), "i8i8": (buf(xI8), buf(wI8)), "f8f8": (buf(xF8), buf(wF8)),
    "f4f4": (buf(xF4), buf(wF4)), "fp8": (buf(xH), buf(wF8)), "int8": (buf(xH), buf(wI8)),
]
let yBuf = device.makeBuffer(length: yc * 4, options: .storageModeShared)!

func refX(_ dt: String, _ i: Int) -> Double {
    switch dt {
    case "i8i8": return Double(xI8[i])
    case "f8f8", "f8f8h": return Double(e4m3ToFloat(xF8[i]))
    case "f4f4": return Double(e2m1At(xF4, i))
    default: return Double(xH[i])
    }
}
func refW(_ dt: String, _ i: Int) -> Double {
    switch dt {
    case "i8i8", "int8": return Double(wI8[i])
    case "f8f8", "f8f8h", "fp8": return Double(e4m3ToFloat(wF8[i]))
    case "f4f4": return Double(e2m1At(wF4, i))
    default: return Double(wH[i])
    }
}
let samples = [(0, 0), (1, 33), (N / 2 + 7, M / 3 + 5), (N - 1, M - 1), (N - 17, 70), (63, M - 40)]

var names = lib.functionNames.sorted()
if !filters.isEmpty { names = names.filter { n in filters.contains { n.contains($0) } } }

print(String(format: "%-34@ %9@ %8@ %8@", "kernel" as NSString, "median_ms" as NSString, "tflops" as NSString, "maxrel" as NSString))
for name in names {
    guard let c = parse(name), let (xb, wb) = bufs[c.dt] else { continue }
    if N % c.tileM != 0 || M % c.tileN != 0 || (c.bk > 1 && K % c.bk != 0) {
        print("\(name) skipped (shape not a multiple of tile)")
        continue
    }
    let pso: MTLComputePipelineState
    do { pso = try device.makeComputePipelineState(function: lib.makeFunction(name: name)!) }
    catch { print("\(name) pipeline failed: \(error)"); continue }
    if c.threads > pso.maxTotalThreadsPerThreadgroup {
        print("\(name) skipped (threads \(c.threads) > max \(pso.maxTotalThreadsPerThreadgroup))")
        continue
    }
    memset(yBuf.contents(), 0, yc * 4)
    let grid = MTLSize(width: M / c.tileN, height: N / c.tileM, depth: 1)
    let tg = MTLSize(width: c.threads, height: 1, depth: 1)
    func once() -> Double {
        let cb = queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(pso)
        e.setBuffer(xb, offset: 0, index: 0); e.setBuffer(wb, offset: 0, index: 1); e.setBuffer(yBuf, offset: 0, index: 2)
        var n = Int32(N), k = Int32(K), m = Int32(M)
        e.setBytes(&n, length: 4, index: 3); e.setBytes(&k, length: 4, index: 4); e.setBytes(&m, length: 4, index: 5)
        e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if cb.status != .completed { print("\(name) GPU error \(String(describing: cb.error))"); return -1 }
        return cb.gpuEndTime - cb.gpuStartTime
    }
    for _ in 0..<warmup { _ = once() }
    var t = (0..<iters).map { _ in once() }
    t.sort()
    let med = t[t.count / 2]
    var maxRel = 0.0
    for (n, m) in samples {
        var acc = 0.0
        for k in 0..<K { acc += refX(c.dt, n * K + k) * refW(c.dt, k * M + m) }
        let got: Double
        switch c.dt {
        case "i8i8": got = Double(yBuf.contents().load(fromByteOffset: (n * M + m) * 4, as: Int32.self))
        case "fp16h", "f8f8h": got = Double(yBuf.contents().load(fromByteOffset: (n * M + m) * 2, as: Float16.self))
        default: got = Double(yBuf.contents().load(fromByteOffset: (n * M + m) * 4, as: Float.self))
        }
        maxRel = max(maxRel, abs(got - acc) / max(1e-3, abs(acc)))
    }
    let tf = 2.0 * Double(N) * Double(K) * Double(M) / med / 1e12
    print(String(format: "%-34@ %9.4f %8.2f %8.1e%@", name as NSString, med * 1000, tf, maxRel,
                 (maxRel > 1e-2 ? "  BAD" : "") as NSString))
    fflush(stdout)
}
