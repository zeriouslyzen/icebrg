// swift-tools-version:5.4
import PackageDescription

let package = Package(
    name: "AdvancedSiriAssistant",
    platforms: [.macOS(.v11)],
    products: [.executable(name: "AdvancedSiriAssistant", targets: ["AdvancedSiriAssistant"])],
    dependencies: [
        .package(url: "https://github.com/apple/swift-algorithms", from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-collections", from: "1.0.0")
    ],
    targets: [
        .executableTarget(
            name: "AdvancedSiriAssistant",
            dependencies: [
                .product(name: "Algorithms", package: "swift-algorithms"),
                .product(name: "Collections", package: "swift-collections")
            ],
            swiftSettings: [.unsafeFlags(["-parse-as-library"])]
        )
    ]
)