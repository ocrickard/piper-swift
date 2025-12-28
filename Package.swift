// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "PiperSwift",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(name: "PiperCore", targets: ["PiperCore"]),
        .library(name: "PiperONNX", targets: ["PiperONNX"]),
        .library(name: "PiperMetal", targets: ["PiperMetal"]),
        .executable(name: "piper-swift", targets: ["PiperCLI"]),
    ],
    targets: [
        .target(
            name: "PiperCore",
            dependencies: [],
            resources: [
                .process("Resources")
            ]
        ),
        .target(
            name: "PiperONNX",
            dependencies: [
                "PiperCore",
            ]
        ),
        .target(
            name: "PiperMetal",
            dependencies: [
                "PiperCore",
                "PiperONNX",
            ]
            ,
            resources: [
                .process("Kernels")
            ]
        ),
        .executableTarget(
            name: "PiperCLI",
            dependencies: [
                "PiperCore",
                "PiperMetal",
            ],
            linkerSettings: [
                .linkedFramework("AVFoundation"),
            ]
        ),
        .testTarget(
            name: "PiperMetalTests",
            dependencies: [
                "PiperCore",
                "PiperMetal",
            ]
        ),
        .testTarget(
            name: "PiperONNXTests",
            dependencies: [
                "PiperONNX",
                "PiperCore",
            ]
        ),
    ]
)


