// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ICEBURGCalculator",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "ICEBURGCalculator",
            targets: ["ICEBURGCalculator"]
        )
    ],
    targets: [
        .executableTarget(
            name: "ICEBURGCalculator",
            dependencies: [],
            path: "Sources"
        )
    ]
)
