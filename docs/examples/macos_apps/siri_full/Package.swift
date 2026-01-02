// swift-tools-version:5.4
import PackageDescription

let package = Package(
    name: "SiriAssistant",
    platforms: [.macOS(.v11)],
    products: [.executable(name: "SiriAssistant", targets: ["SiriAssistant"])],
    targets: [.executableTarget(name: "SiriAssistant", swiftSettings: [.unsafeFlags(["-parse-as-library"])])]
)