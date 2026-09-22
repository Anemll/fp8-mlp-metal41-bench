import Foundation
import Metal

// Host for peak.metal. Reports NAX, ALU and combined TFLOPs for each probe.
// usage: peak --metallib path [--tgs n --R r --RA ra --iters i --filter substr[,substr]]

var tgs = 0, R = 2048, RA = 0, iters = 10
var filters: [String] = []
var libPath = "peak.metallib"
var argv = Array(CommandLine.arguments.dropFirst())
while argv.count >= 2 {
    let k = argv.removeFirst(), v = argv.removeFirst()
    switch k {
    case "--tgs": tgs = Int(v)!
    case "--R": R = Int(v)!
    case "--RA": RA = Int(v)!
    case "--iters": iters = Int(v)!
    case "--filter": filters = v.split(separator: ",").map(String.init)
    case "--metallib": libPath = v
    default: break
    }
}

let device = MTLCreateSystemDefaultDevice()!
let lib = try! device.makeLibrary(URL: URL(fileURLWithPath: libPath))
let queue = device.makeCommandQueue()!
// Enough threadgroups to fill every core several times over.
if tgs == 0 { tgs = 40 * 32 }
// ALU step is a 32x32x8 FMA block (16 8x8x8 MACs); default RA so ALU and NAX run ~equal time.
let aluFlopsPerIter = 2.0 * 32 * 32 * 8
print("device: \(device.name)  threadgroups=\(tgs) R=\(R)")
let yBuf = device.makeBuffer(length: tgs * 1024 * 4, options: .storageModeShared)!

var names = lib.functionNames.sorted()
if !filters.isEmpty { names = names.filter { n in filters.contains { n.contains($0) } } }
print(String(format: "%-28@ %4@ %4@ %9@ %9@ %9@ %9@", "kernel" as NSString, "nax" as NSString, "alu" as NSString,
             "ms" as NSString, "nax_tf" as NSString, "alu_tf" as NSString, "total_tf" as NSString))
for name in names {
    let p = name.split(separator: "_").map(String.init)
    let f = p[2].dropFirst().split(separator: "x").map { Int($0)! }
    let bk = Int(p[3].dropFirst())!, nn = Int(p[4].dropFirst())!, na = Int(p[5].dropFirst())!
    let pso = try! device.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    let naxFlopsPerIter = 2.0 * Double(f[0] * f[1] * bk)
    // Match ALU work to NAX time is unknown up front; use --RA or R/8 by default.
    let ra = RA > 0 ? RA : max(1, R / 8)
    var r32 = Int32(R), ra32 = Int32(ra)
    func once() -> Double {
        let cb = queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(pso)
        e.setBuffer(yBuf, offset: 0, index: 0)
        e.setBytes(&r32, length: 4, index: 1)
        e.setBytes(&ra32, length: 4, index: 2)
        e.dispatchThreadgroups(MTLSize(width: tgs, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32 * (nn + na), height: 1, depth: 1))
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        return cb.gpuEndTime - cb.gpuStartTime
    }
    _ = once(); _ = once()
    var t = (0..<iters).map { _ in once() }
    t.sort()
    let med = t[t.count / 2]
    let nax = Double(tgs * nn) * Double(R) * naxFlopsPerIter / med / 1e12
    let alu = Double(tgs * na) * Double(ra) * aluFlopsPerIter / med / 1e12
    print(String(format: "%-28@ %4d %4d %9.3f %9.2f %9.2f %9.2f", name as NSString, nn, na, med * 1000, nax, alu, nax + alu))
    fflush(stdout)
}
