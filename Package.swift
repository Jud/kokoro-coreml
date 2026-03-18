// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "KokoroANE",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "KokoroANE", targets: ["KokoroANE"]),
        .executable(name: "kokoro", targets: ["CLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/Jud/swift-bart-g2p.git", from: "0.4.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
        .package(url: "https://github.com/edgeengineer/cbor.git", .upToNextMinor(from: "0.0.6")),
    ],
    targets: [
        .target(
            name: "KokoroANE",
            dependencies: [
                .product(name: "BARTG2P", package: "swift-bart-g2p")
            ],
            path: "Sources/KokoroANE",
            resources: [
                .process("Resources")
            ]
        ),
        .executableTarget(
            name: "CLI",
            dependencies: [
                "KokoroANE",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "CBOR", package: "cbor"),
            ],
            path: "Sources/CLI"
        ),
        .testTarget(
            name: "KokoroANETests",
            dependencies: ["KokoroANE"],
            path: "Tests/KokoroANETests",
            resources: [.process("kokoro_g2p_reference.json")]
        ),
    ]
)
