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
    let mant = bits & 0x7FFFFF
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

// Decode the stored E4M3FN byte (including subnormals). NaN -> NaN.
func e4m3ToFloat(_ b: UInt8) -> Float {
    let sign: Float = (b & 0x80) != 0 ? -1 : 1
    let exp = Int((b >> 3) & 0x0F)
    let mant = Int(b & 0x07)
    if exp == 0 {
        return sign * Float(mant) * exp2(-9)
    }
    if exp == 15 && mant == 7 {
        return Float.nan
    }
    return sign * (1 + Float(mant) / 8) * exp2(Float(exp - 7))
}

// OCP E2M1: 1 sign, 2 exp, 1 mantissa, bias 1. Finite set is 0, 0.5, 1, 1.5, 2, 3, 4, 6.
func e2m1ToFloat(_ nibble: UInt8) -> Float {
    let mag: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    let sign: Float = (nibble & 0x8) != 0 ? -1 : 1
    return sign * mag[Int(nibble & 0x7)]
}

func e2m1Code(_ i: Int) -> UInt8 {
    let codes: [UInt8] = [0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x9, 0xA, 0xC]
    return codes[i % codes.count]
}

// Low nibble is the even element.
func packE2M1(_ count: Int, _ code: (Int) -> UInt8) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: count / 2)
    for i in 0..<count {
        let nibble = code(i) & 0xF
        if i % 2 == 0 {
            bytes[i / 2] |= nibble
        } else {
            bytes[i / 2] |= nibble << 4
        }
    }
    return bytes
}

func e2m1At(_ bytes: [UInt8], _ index: Int) -> Float {
    let byte = bytes[index / 2]
    let nibble = (index % 2 == 0) ? (byte & 0xF) : (byte >> 4)
    return e2m1ToFloat(nibble)
}

// OCP UE8M0: scale = 2^(byte - 127). One scale covers 32 elements along K.
func ue8m0ToFloat(_ code: UInt8) -> Float {
    exp2(Float(Int(code) - 127))
}

func ue8m0Code(_ i: Int) -> UInt8 {
    let codes: [UInt8] = [125, 126, 127, 128]
    return codes[i % codes.count]
}

enum DType: String, CaseIterable {
    case fp16
    case int8
    case i8i8
    case fp8
    case f8f8
    case f4f4
    case mxfp4
    case mx4x4 = "mxfp4/a4"
    case mx8x4 = "mxfp4/a8"
    case f8x4 = "mxfp4/f8"

    var kernel: String {
        switch self {
        case .fp16: return "fp16_mlp_gemm"
        case .int8: return "int8_mlp_gemm"
        case .i8i8: return "i8i8_mlp_gemm"
        case .fp8: return "fp8_mlp_gemm"
        case .f8f8: return "f8f8_mlp_gemm"
        case .f4f4: return "f4f4_mlp_gemm"
        case .mxfp4: return "mxfp4_mlp_gemm"
        case .mx4x4: return "mx4x4_mlp_gemm"
        case .mx8x4: return "mx8x4_mlp_gemm"
        case .f8x4: return "f8x4_mlp_gemm"
        }
    }
}

struct ShapeCase {
    var name: String
    var N: Int
    var K: Int
    var M: Int
    var iters: Int
}

struct Args {
    var N = 4096
    var K = 4096
    var M = 4096
    var iters = 100
    var warmup = 5
    var dtypes: [DType] = Array(DType.allCases)
    var suite = false
    var shapeSet = false
    var tile: Tile? = nil
    var autotune = false
    var useProfile = true
}

// Output tile per threadgroup: BM rows of Y (N direction) x BN cols (M direction), 4 simdgroups.
struct Tile: Hashable {
    var bm: Int
    var bn: Int
    var name: String { "\(bm)x\(bn)" }
}

let tiles = [Tile(bm: 128, bn: 64), Tile(bm: 64, bn: 128), Tile(bm: 128, bn: 128), Tile(bm: 64, bn: 64), Tile(bm: 64, bn: 32), Tile(bm: 32, bn: 64),
             Tile(bm: 16, bn: 64), Tile(bm: 16, bn: 32), Tile(bm: 8, bn: 64), Tile(bm: 8, bn: 32)]

// Best measured tile per dtype on M5 Max (tune/README.md). The Apple sample tile (64x32)
// reaches only half the NAX peak there. N <= 16 (decode) only fills the first rows of a
// tile, so it takes a 16-row tile with more columns in flight.
func defaultTile(_ dtype: DType, _ N: Int) -> Tile {
    if N <= 16 { return Tile(bm: 16, bn: 32) }
    switch dtype {
    case .fp16, .i8i8: return Tile(bm: 128, bn: 64)
    case .int8, .fp8, .f8f8, .f4f4, .mxfp4, .mx4x4, .mx8x4, .f8x4: return Tile(bm: 64, bn: 64)
    }
}

let suiteShapes = [
    ShapeCase(name: "square", N: 4096, K: 4096, M: 4096, iters: 100),
    ShapeCase(name: "thin", N: 1, K: 4096, M: 11008, iters: 200),
    ShapeCase(name: "fat", N: 2048, K: 4096, M: 11008, iters: 50),
]

func parseArgs() -> Args {
    var a = Args()
    let argv = CommandLine.arguments
    var i = 1
    while i < argv.count {
        let key = argv[i]
        if key == "--suite" || key == "--autotune" || key == "--no-profile" {
            if key == "--suite" { a.suite = true }
            if key == "--autotune" { a.autotune = true }
            if key == "--no-profile" { a.useProfile = false }
            i += 1
            continue
        }
        i += 1
        guard i < argv.count else { break }
        if key == "--tile" {
            let name = argv[i]
            i += 1
            guard let t = tiles.first(where: { $0.name == name }) else {
                fputs("Unknown --tile \(name) (\(tiles.map(\.name).joined(separator: ", ")))\n", stderr)
                exit(1)
            }
            a.tile = t
            continue
        }
        if key == "--dtype" {
            let name = argv[i]
            i += 1
            if name == "all" {
                a.dtypes = Array(DType.allCases)
            } else if let dtype = DType(rawValue: name) {
                a.dtypes = [dtype]
            } else {
                fputs("Unknown --dtype \(name) (fp16, int8, i8i8, fp8, f8f8, f4f4, mxfp4, mxfp4/a4, mxfp4/a8, mxfp4/f8, all)\n", stderr)
                exit(1)
            }
            continue
        }
        let val = Int(argv[i]) ?? 0
        switch key {
        case "--N": a.N = val; a.shapeSet = true
        case "--K": a.K = val; a.shapeSet = true
        case "--M": a.M = val; a.shapeSet = true
        case "--iters": a.iters = val; a.shapeSet = true
        case "--warmup": a.warmup = val
        default: break
        }
        i += 1
    }
    return a
}

func loadLibrary(device: MTLDevice) -> MTLLibrary {
    let cwd = FileManager.default.currentDirectoryPath
    let envPath = ProcessInfo.processInfo.environment["FP8_MLP_METALLIB"]
    let libPath = envPath ?? URL(fileURLWithPath: cwd).appendingPathComponent("default.metallib").path
    if FileManager.default.fileExists(atPath: libPath) {
        do {
            let library = try device.makeLibrary(URL: URL(fileURLWithPath: libPath))
            print("loaded: \(libPath)")
            return library
        } catch {
            fputs("Failed loading metallib: \(error)\n", stderr)
            exit(2)
        }
    }

    let srcURL = URL(fileURLWithPath: cwd).appendingPathComponent("fp8_mlp.metal")
    guard let src = try? String(contentsOf: srcURL, encoding: .utf8) else {
        fputs("Need FP8_MLP_METALLIB, default.metallib, or fp8_mlp.metal in \(cwd)\n", stderr)
        exit(1)
    }
    let options = MTLCompileOptions()
    options.languageVersion = .version4_1
    do {
        let library = try device.makeLibrary(source: src, options: options)
        print("compiled from source (metal4.1): \(srcURL.path)")
        return library
    } catch {
        fputs("Metal compile failed (need Metal 4.1 + FP8 matmul):\n\(error)\n", stderr)
        exit(2)
    }
}

func makePipeline(library: MTLLibrary, device: MTLDevice, name: String) -> MTLComputePipelineState {
    guard let fn = library.makeFunction(name: name) else {
        fputs("Missing kernel \(name)\n", stderr)
        exit(1)
    }
    do {
        return try device.makeComputePipelineState(function: fn)
    } catch {
        fputs("Pipeline \(name) failed: \(error)\n", stderr)
        exit(1)
    }
}

struct Row {
    var shape: String
    var N: Int
    var K: Int
    var M: Int
    var dtype: DType
    var tile: Tile
    var medianMs: Double
    var tflops: Double
    var vsFp16: String
    var refRel: Float
}

guard let device = MTLCreateSystemDefaultDevice() else {
    fputs("No Metal device\n", stderr)
    exit(1)
}

let args = parseArgs()
let shapes: [ShapeCase] = (args.suite || !args.shapeSet)
    ? suiteShapes
    : [ShapeCase(name: "custom", N: args.N, K: args.K, M: args.M, iters: args.iters)]
for shape in shapes {
    guard shape.N > 0, shape.K > 0, shape.M > 0, shape.iters > 0 else {
        fputs("N, K, M, iters must be > 0\n", stderr)
        exit(1)
    }
}

print("device: \(device.name)")
print("timing: gpu")
print("dtypes: \(args.dtypes.map(\.rawValue).joined(separator: ","))")
print("shapes: \(shapes.map { "\($0.name) N=\($0.N) K=\($0.K) M=\($0.M) iters=\($0.iters)" }.joined(separator: " | "))")

let library = loadLibrary(device: device)
guard let queue = device.makeCommandQueue() else {
    fputs("command queue alloc failed\n", stderr)
    exit(1)
}

struct PipelineKey: Hashable {
    var dtype: DType
    var tile: Tile
}
var pipelines: [PipelineKey: MTLComputePipelineState] = [:]
func pipeline(_ dtype: DType, _ tile: Tile) -> MTLComputePipelineState {
    let key = PipelineKey(dtype: dtype, tile: tile)
    if let p = pipelines[key] { return p }
    let p = makePipeline(library: library, device: device, name: "\(dtype.kernel)_\(tile.name)")
    pipelines[key] = p
    return p
}

// Per-device tile profile, written by --autotune and read by later runs.
// Keyed by device name and GPU architecture so M5, M5 Max, M6 each keep their own file.
struct Profile: Codable {
    var device: String
    var architecture: String
    var tiles: [String: String]  // "<dtype> N=.. K=.. M=.." -> "BMxBN"
}

let architecture = device.architecture.name
let profileURL: URL = {
    let slug = "\(device.name)_\(architecture)".lowercased()
        .map { $0.isLetter || $0.isNumber ? String($0) : "-" }.joined()
    return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appendingPathComponent("profiles").appendingPathComponent("\(slug).json")
}()
var profile: Profile = {
    if args.useProfile, let data = try? Data(contentsOf: profileURL),
       let p = try? JSONDecoder().decode(Profile.self, from: data) {
        return p
    }
    return Profile(device: device.name, architecture: architecture, tiles: [:])
}()
let profileLoaded = !profile.tiles.isEmpty
print("architecture: \(architecture)")
if args.autotune {
    print("tiles: autotune (writes \(profileURL.path))")
} else if args.tile != nil {
    print("tiles: --tile \(args.tile!.name)")
} else if profileLoaded {
    print("tiles: profile \(profileURL.path)")
} else {
    print("tiles: built-in defaults (run --autotune to fit this device)")
}

func profileKey(_ dtype: DType, _ N: Int, _ K: Int, _ M: Int) -> String {
    "\(dtype.rawValue) N=\(N) K=\(K) M=\(M)"
}

func chosenTile(_ dtype: DType, _ N: Int, _ K: Int, _ M: Int) -> Tile {
    if let t = args.tile { return t }
    if args.useProfile, let name = profile.tiles[profileKey(dtype, N, K, M)],
       let t = tiles.first(where: { $0.name == name }) {
        return t
    }
    return defaultTile(dtype, N)
}

func bench(_ shape: ShapeCase) -> [Row] {
    let N = shape.N, K = shape.K, M = shape.M
    let xCount = N * K
    let wCount = K * M
    let yCount = N * M
    var xHost = [Float16](repeating: 0, count: xCount)
    var xInt8 = [Int8](repeating: 0, count: xCount)
    var xFp8 = [UInt8](repeating: 0, count: xCount)
    var wFp16 = [Float16](repeating: 0, count: wCount)
    var wInt8 = [Int8](repeating: 0, count: wCount)
    var wFp8 = [UInt8](repeating: 0, count: wCount)
    let fp4OK = xCount % 2 == 0 && wCount % 2 == 0
    let mxOK = fp4OK && K % 32 == 0
    var xFp4 = fp4OK ? packE2M1(xCount) { e2m1Code($0) } : [UInt8(0)]
    var wFp4 = fp4OK ? packE2M1(wCount) { e2m1Code($0 + 3) } : [UInt8(0)]
    // MX weight is [M, K], so element m*K+k sits next to the rest of that row.
    var wMx = mxOK ? packE2M1(wCount) { e2m1Code($0 + 5) } : [UInt8(0)]
    let kBlocks = max(K / 32, 1)
    var wScale = mxOK ? (0..<(M * kBlocks)).map { ue8m0Code($0) } : [UInt8(0)]
    var xScale = mxOK ? (0..<(N * kBlocks)).map { ue8m0Code($0 + 1) } : [UInt8(0)]
    for i in 0..<xCount {
        let xf = Float((i % 97) - 48) * 0.02
        xHost[i] = Float16(xf)
        xInt8[i] = Int8((i % 97) - 48)
        xFp8[i] = floatToE4M3(xf)
    }
    for i in 0..<wCount {
        let f = Float((i % 53) - 26) * 0.05
        wFp16[i] = Float16(f)
        wInt8[i] = Int8(max(-128, min(127, Int((i % 53) - 26))))
        wFp8[i] = floatToE4M3(f)
    }
    guard let xBuf = device.makeBuffer(bytes: &xHost, length: xCount * MemoryLayout<Float16>.size, options: .storageModeShared),
          let xInt8Buf = device.makeBuffer(bytes: &xInt8, length: xCount, options: .storageModeShared),
          let xFp8Buf = device.makeBuffer(bytes: &xFp8, length: xCount, options: .storageModeShared),
          let xFp4Buf = device.makeBuffer(bytes: &xFp4, length: xFp4.count, options: .storageModeShared),
          let wFp4Buf = device.makeBuffer(bytes: &wFp4, length: wFp4.count, options: .storageModeShared),
          let wMxBuf = device.makeBuffer(bytes: &wMx, length: wMx.count, options: .storageModeShared),
          let wScaleBuf = device.makeBuffer(bytes: &wScale, length: wScale.count, options: .storageModeShared),
          let xScaleBuf = device.makeBuffer(bytes: &xScale, length: xScale.count, options: .storageModeShared),
          let wFp16Buf = device.makeBuffer(bytes: &wFp16, length: wCount * MemoryLayout<Float16>.size, options: .storageModeShared),
          let wInt8Buf = device.makeBuffer(bytes: &wInt8, length: wCount, options: .storageModeShared),
          let wFp8Buf = device.makeBuffer(bytes: &wFp8, length: wCount, options: .storageModeShared),
          let yBuf = device.makeBuffer(length: yCount * MemoryLayout<Float>.size, options: .storageModeShared)
    else {
        fputs("Buffer alloc failed for \(shape.name)\n", stderr)
        exit(1)
    }

    func weightBuffer(_ dtype: DType) -> MTLBuffer {
        switch dtype {
        case .fp16: return wFp16Buf
        case .int8: return wInt8Buf
        case .i8i8: return wInt8Buf
        case .fp8: return wFp8Buf
        case .f8f8: return wFp8Buf
        case .f4f4: return wFp4Buf
        case .mxfp4, .mx4x4, .mx8x4, .f8x4: return wMxBuf
        }
    }
    func refActivation(_ dtype: DType, _ i: Int) -> Float {
        switch dtype {
        case .fp16, .int8, .fp8: return Float(xHost[i])
        case .i8i8: return Float(xInt8[i])
        case .f8f8: return e4m3ToFloat(xFp8[i])
        case .f4f4: return e2m1At(xFp4, i)
        case .mxfp4: return Float(xHost[i])
        case .mx4x4:
            let block = K / 32
            let n = i / K
            let k = i % K
            return e2m1At(xFp4, i) * ue8m0ToFloat(xScale[n * block + k / 32])
        case .mx8x4:
            let block = K / 32
            return e4m3ToFloat(xFp8[i]) * ue8m0ToFloat(xScale[(i / K) * block + (i % K) / 32])
        case .f8x4: return e4m3ToFloat(xFp8[i])
        }
    }
    func refWeight(_ dtype: DType, _ k: Int, _ m: Int) -> Float {
        let i = k * M + m
        switch dtype {
        case .fp16: return Float(wFp16[i])
        case .int8: return Float(wInt8[i])
        case .i8i8: return Float(wInt8[i])
        case .fp8, .f8f8: return e4m3ToFloat(wFp8[i])
        case .f4f4: return e2m1At(wFp4, i)
        case .mxfp4, .mx4x4, .mx8x4, .f8x4:
            let block = K / 32
            return e2m1At(wMx, m * K + k) * ue8m0ToFloat(wScale[m * block + k / 32])
        }
    }

    // Tile corners and the far edge of the matrix.
    func samples(_ tile: Tile) -> [(Int, Int)] {
        var out = [(Int, Int)]()
        func add(_ n: Int, _ m: Int) {
            if n >= 0 && n < N && m >= 0 && m < M && !out.contains(where: { $0.0 == n && $0.1 == m }) {
                out.append((n, m))
            }
        }
        add(0, 0)
        add(0, min(tile.bn - 1, M - 1))
        add(0, min(tile.bn, M - 1))
        add(min(tile.bm - 1, N - 1), 0)
        add(min(tile.bm, N - 1), min(tile.bn, M - 1))
        add(N - 1, M - 1)
        return out
    }

    let flops = 2.0 * Double(N) * Double(K) * Double(M)
    var medians: [DType: Double] = [:]
    var rows: [Row] = []

    for dtype in args.dtypes {
        if (dtype == .f4f4 && !fp4OK) || ([.mxfp4, .mx4x4].contains(dtype) && !mxOK) || ([.mx8x4, .f8x4].contains(dtype) && (!mxOK || K % 128 != 0)) {
            fputs("\(dtype.rawValue) needs even X/W counts and K a multiple of 32 (128 for FP8 × MXFP4)\n", stderr)
            exit(1)
        }
        let xArg: MTLBuffer
        switch dtype {
        case .i8i8: xArg = xInt8Buf
        case .f8f8, .mx8x4, .f8x4: xArg = xFp8Buf
        case .f4f4, .mx4x4: xArg = xFp4Buf
        case .fp16, .int8, .fp8, .mxfp4: xArg = xBuf
        }
        let wBuf = weightBuffer(dtype)

        func runOnce(_ tile: Tile) -> Double {
            let pipeline = pipeline(dtype, tile)
            let grid = MTLSize(width: (M + tile.bn - 1) / tile.bn,
                               height: (N + tile.bm - 1) / tile.bm,
                               depth: 1)
            let threads = MTLSize(width: pipeline.threadExecutionWidth * 4, height: 1, depth: 1)
            guard let cb = queue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else {
                fputs("command buffer alloc failed\n", stderr)
                exit(1)
            }
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(xArg, offset: 0, index: 0)
            enc.setBuffer(wBuf, offset: 0, index: 1)
            enc.setBuffer(yBuf, offset: 0, index: 2)
            var n32 = Int32(N), k32 = Int32(K), m32 = Int32(M)
            enc.setBytes(&n32, length: 4, index: 3)
            enc.setBytes(&k32, length: 4, index: 4)
            enc.setBytes(&m32, length: 4, index: 5)
            if dtype == .mxfp4 || dtype == .f8x4 {
                enc.setBuffer(wScaleBuf, offset: 0, index: 6)
            } else if dtype == .mx4x4 || dtype == .mx8x4 {
                enc.setBuffer(xScaleBuf, offset: 0, index: 6)
                enc.setBuffer(wScaleBuf, offset: 0, index: 7)
            }
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: threads)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if cb.status != .completed {
                fputs("\(shape.name) \(dtype.rawValue) GPU error: \(cb.error?.localizedDescription ?? "\(cb.status)")\n", stderr)
                exit(1)
            }
            let gpu = cb.gpuEndTime - cb.gpuStartTime
            return gpu > 0 ? gpu : 0
        }

        var tile = chosenTile(dtype, N, K, M)
        if args.autotune {
            // Short median per tile; the full timing below reruns the winner.
            let tuneIters = max(5, min(20, shape.iters / 5))
            var results: [(Tile, Double)] = []
            for candidate in tiles {
                for _ in 0..<2 { _ = runOnce(candidate) }
                var t = (0..<tuneIters).map { _ in runOnce(candidate) }
                t.sort()
                results.append((candidate, t[t.count / 2]))
            }
            results.sort { $0.1 < $1.1 }
            tile = results[0].0
            profile.tiles[profileKey(dtype, N, K, M)] = tile.name
            let top = results.prefix(4).map { String(format: "%@ %.4f", $0.0.name, $0.1 * 1000) }
            print("autotune \(shape.name) \(dtype.rawValue): \(top.joined(separator: " | ")) ms")
        }
        let samples = samples(tile)
        print("running \(shape.name) \(dtype.rawValue) tile \(tile.name)")
        fflush(stdout)
        yBuf.contents().initializeMemory(as: UInt8.self, repeating: 0, count: yCount * MemoryLayout<Float>.size)

        for _ in 0..<args.warmup { _ = runOnce(tile) }
        var times = [Double]()
        times.reserveCapacity(shape.iters)
        for _ in 0..<shape.iters {
            let dt = runOnce(tile)
            if dt <= 0 {
                fputs("GPU timestamps unavailable\n", stderr)
                exit(1)
            }
            times.append(dt)
        }
        times.sort()
        let median = times[times.count / 2]
        medians[dtype] = median

        var maxRel: Float = 0
        if dtype == .i8i8 {
            let yPtr = yBuf.contents().bindMemory(to: Int32.self, capacity: yCount)
            for (n, m) in samples {
                var acc: Int32 = 0
                let xBase = n * K
                for k in 0..<K {
                    acc += Int32(xInt8[xBase + k]) * Int32(wInt8[k * M + m])
                }
                let got = yPtr[n * M + m]
                if got != acc {
                    fputs(String(format: "%@ i8i8 mismatch Y[%d,%d] gpu=%d ref=%d\n",
                                 shape.name, n, m, got, acc), stderr)
                    exit(3)
                }
            }
        } else {
            let yPtr = yBuf.contents().bindMemory(to: Float.self, capacity: yCount)
            for (n, m) in samples {
                var acc: Float = 0
                let xBase = n * K
                for k in 0..<K {
                    acc += refActivation(dtype, xBase + k) * refWeight(dtype, k, m)
                }
                let got = yPtr[n * M + m]
                let absErr = abs(got - acc)
                let rel = absErr / max(1e-6, abs(acc))
                if rel > maxRel { maxRel = rel }
                if rel > 1e-2 && absErr > 1e-3 {
                    fputs(String(format: "%@ %@ mismatch Y[%d,%d] gpu=%.6g ref=%.6g abs=%.3g rel=%.3g\n",
                                 shape.name, dtype.rawValue, n, m, got, acc, absErr, rel), stderr)
                    exit(3)
                }
            }
        }

        let vs: String
        if dtype == .fp16 {
            vs = "1.000"
        } else if let fp16 = medians[.fp16] {
            vs = String(format: "%.3f", fp16 / median)
        } else {
            vs = "-"
        }
        rows.append(Row(shape: shape.name, N: N, K: K, M: M, dtype: dtype, tile: tile,
                        medianMs: median * 1000, tflops: (flops / median) / 1e12,
                        vsFp16: vs, refRel: maxRel))
    }
    return rows
}

var rows: [Row] = []
for shape in shapes {
    rows.append(contentsOf: bench(shape))
}

func cell(_ text: String, _ width: Int, right: Bool) -> String {
    let pad = max(0, width - text.count)
    let spaces = String(repeating: " ", count: pad)
    return right ? spaces + text : text + spaces
}

func tableLine(_ shape: String, _ n: String, _ k: String, _ m: String,
               _ dtype: String, _ tile: String, _ ms: String, _ tflops: String,
               _ vs: String, _ rel: String) -> String {
    [
        cell(shape, 8, right: false),
        cell(n, 6, right: true),
        cell(k, 6, right: true),
        cell(m, 6, right: true),
        cell(dtype, 9, right: false),
        cell(tile, 6, right: false),
        cell(ms, 10, right: true),
        cell(tflops, 8, right: true),
        cell(vs, 8, right: true),
        cell(rel, 10, right: true),
    ].joined(separator: " ")
}

if args.autotune {
    do {
        try FileManager.default.createDirectory(at: profileURL.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(profile).write(to: profileURL)
        print("wrote profile: \(profileURL.path)")
    } catch {
        fputs("could not write profile \(profileURL.path): \(error)\n", stderr)
    }
}

print("")
let header = tableLine("shape", "N", "K", "M", "dtype", "tile", "median_ms", "tflops", "vs_fp16", "ref_rel")
print(header)
print(String(repeating: "-", count: header.count))
for row in rows {
    print(tableLine(
        row.shape,
        String(row.N),
        String(row.K),
        String(row.M),
        row.dtype.rawValue,
        row.tile.name,
        String(format: "%.4f", row.medianMs),
        String(format: "%.3f", row.tflops),
        row.vsFp16,
        String(format: "%.3g", row.refRel)))
}
print("ok")
fflush(stdout)
