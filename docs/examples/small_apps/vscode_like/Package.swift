// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "VSCodeLike",
    platforms: [.macOS(.v13)],
    products: [.executable(name: "VSCodeLike", targets: ["VSCodeLike"])],
    targets: [.executableTarget(name: "VSCodeLike", path: "Sources")]
)
