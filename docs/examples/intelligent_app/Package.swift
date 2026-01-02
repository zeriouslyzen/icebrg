// swift-tools-version:5.4
import PackageDescription

let package = Package(
    name: "RealApp",
    platforms: [.macOS(.v11)],
    products: [.executable(name: "RealApp", targets: ["RealApp"])],
    targets: [.executableTarget(name: "RealApp", path: "Sources/RealApp", swiftSettings: [.unsafeFlags(["-parse-as-library"])])]
)