// swift-tools-version:5.4
import PackageDescription

let package = Package(
    name: "GeneratedApp",
    platforms: [.macOS(.v11)],
    products: [.executable(name: "GeneratedApp", targets: ["GeneratedApp"])],
    targets: [.executableTarget(name: "GeneratedApp", path: "Sources/GeneratedApp")]
)