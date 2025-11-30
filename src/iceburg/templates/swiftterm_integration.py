"""
SwiftTerm Integration for SwiftUI

Provides terminal integration using SwiftTerm library:
    - Terminal emulation
- Process execution
- Command history
- Custom themes
- Integration with IDE
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json


class SwiftTermIntegration:
    """SwiftTerm integration for IDE applications."""
    
    def __init__(self):
        self.terminal_themes = {
            'dark': {
                'background': '#1e1e1e',
                'foreground': '#d4d4d4',
                'cursor': '#aeafad',
                'selection': '#264f78'
            },
            'light': {
                'background': '#ffffff',
                'foreground': '#000000',
                'cursor': '#000000',
                'selection': '#add6ff'
            }
        }
    
    def generate_terminal_swift(self, features: List[str] = None) -> str:
        """
        Generate Swift code for terminal integration.
        
        Args:
            features: List of terminal features
            
        Returns:
            Swift code for terminal integration
        """
        features = features or []
        
        swift_code = f'''import SwiftUI
import SwiftTerm
import Foundation

// MARK: - Terminal View
struct TerminalView: NSViewRepresentable {{
    @ObservedObject var terminalManager: TerminalManager
    @State private var terminal: TerminalView as TerminalView
    
    func makeNSView(context: Context) -> TerminalView {{
        let terminal = TerminalView()
        terminal.setupTerminal()
        return terminal
    }}
    
    func updateNSView(_ nsView: TerminalView, context: Context) {{
        // Update terminal if needed
    }}
    
    func setupTerminal() {{
        // Configure terminal appearance
        terminal.font = NSFont.monospacedSystemFont(ofSize: 14, weight: .regular)
        terminal.backgroundColor = NSColor(hex: "{self.terminal_themes['dark']['background']}")
        terminal.foregroundColor = NSColor(hex: "{self.terminal_themes['dark']['foreground']}")
        terminal.cursorColor = NSColor(hex: "{self.terminal_themes['dark']['cursor']}")
        
        // Configure terminal behavior
        terminal.isScrollEnabled = true
        terminal.isScrollToBottomOnTtyWrite = true
        terminal.isScrollToBottomOnNewLine = true
        
        // Set up command execution
        terminal.setupCommandExecution()
    }}
}}

// MARK: - Terminal Manager
class TerminalManager: ObservableObject {{
    @Published var isRunning = false
    @Published var currentDirectory = FileManager.default.homeDirectoryForCurrentUser.path
    @Published var commandHistory: [String] = []
    @Published var currentCommand = ""
    
    private var terminal: TerminalView?
    private var currentProcess: Process?
    
    func executeCommand(_ command: String) {{
        guard !command.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {{ return }}
        
        // Add to history
        commandHistory.append(command)
        
        // Execute command
        let process = Process()
        process.launchPath = "/bin/bash"
        process.arguments = ["-c", command]
        process.currentDirectoryPath = currentDirectory
        
        // Set up output handling
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        
        // Handle output
        pipe.fileHandleForReading.readabilityHandler = {{ handle in
            let data = handle.availableData
            if !data.isEmpty {{
                let output = String(data: data, encoding: .utf8) ?? ""
                DispatchQueue.main.async {{
                    self.handleCommandOutput(output)
                }}
            }}
        }}
        
        // Handle process completion
        process.terminationHandler = {{ process in
            DispatchQueue.main.async {{
                self.isRunning = false
                self.handleCommandCompletion(process.terminationStatus)
            }}
        }}
        
        // Start process
        do {{
            try process.run()
            currentProcess = process
            isRunning = true
            
            // Send telemetry
            self.sendTelemetry("command_executed", {{
                "command": command,
                "directory": self.currentDirectory
            }})
        }} catch {{
            self.handleCommandError(error)
        }}
    }}
    
    func stopCurrentCommand() {{
        currentProcess?.terminate()
        isRunning = false
    }}
    
    func changeDirectory(_ path: String) {{
        let newPath = path.hasPrefix("/") ? path : "\(currentDirectory)/\(path)"
        
        if FileManager.default.fileExists(atPath: newPath) {{
            currentDirectory = newPath
            sendTelemetry("directory_changed", {{ "path": newPath }})
        }}
    }}
    
    func getCommandSuggestions(for input: String) -> [String] {{
        // Basic command suggestions
        let commands = [
            "ls", "cd", "pwd", "mkdir", "rm", "cp", "mv", "cat", "grep", "find",
            "git", "swift", "python", "node", "npm", "brew", "xcodebuild"
        ]
        
        return commands.filter {{ $0.hasPrefix(input.lowercased()) }}
    }}
    
    private func handleCommandOutput(_ output: String) {{
        // Send output to terminal
        terminal?.feed(text: output)
        
        // Send telemetry
        sendTelemetry("command_output", {{
            "output": output.substring(to: 100) + (output.count > 100 ? "..." : "")
        }})
    }}
    
    private func handleCommandCompletion(_ status: Int32) {{
        let statusText = status == 0 ? "success" : "error"
        sendTelemetry("command_completed", {{
            "status": statusText,
            "exit_code": status
        }})
    }}
    
    private func handleCommandError(_ error: Error) {{
        let errorMessage = error.localizedDescription
        terminal?.feed(text: "Error: \\(errorMessage)\\n")
        
        sendTelemetry("command_error", {{
            "error": errorMessage
        }})
    }}
    
    private func sendTelemetry(_ event: String, _ data: [String: Any]) {{
        // Send telemetry to parent app
        let telemetryData = [
            "event": event,
            "data": data,
            "timestamp": Date().timeIntervalSince1970
        ] as [String : Any]
        
        // This would integrate with the main app's telemetry system
    }}
}}

// MARK: - Terminal Configuration
struct TerminalConfiguration {{
    let theme: TerminalTheme
    let fontSize: CGFloat
    let fontFamily: String
    let cursorStyle: CursorStyle
    let scrollbackLines: Int
    
    static let `default` = TerminalConfiguration(
        theme: .dark,
        fontSize: 14,
        fontFamily: "Monaco",
        cursorStyle: .block,
        scrollbackLines: 1000
    )
}}

enum TerminalTheme {{
    case dark
    case light
    
    var colors: [String: String] {{
        switch self {{
        case .dark:
            return {{
                "background": "#1e1e1e",
                "foreground": "#d4d4d4",
                "cursor": "#aeafad",
                "selection": "#264f78"
            }}
        case .light:
            return {{
                "background": "#ffffff",
                "foreground": "#000000",
                "cursor": "#000000",
                "selection": "#add6ff"
            }}
        }}
    }}
}}

enum CursorStyle {{
    case block
    case underline
    case bar
}}

// MARK: - Terminal Extensions
extension NSColor {{
    convenience init(hex: String) {{
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {{
        case 3: // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (1, 1, 1, 0)
        }}
        
        self.init(
            red: CGFloat(r) / 255,
            green: CGFloat(g) / 255,
            blue: CGFloat(b) / 255,
            alpha: CGFloat(a) / 255
        )
    }}
}}

// MARK: - Terminal Commands
struct TerminalCommands {{
    static let builtInCommands = [
        "clear": "Clear the terminal screen",
        "history": "Show command history",
        "pwd": "Print working directory",
        "ls": "List directory contents",
        "cd": "Change directory",
        "help": "Show available commands"
    ]
    
    static func executeBuiltInCommand(_ command: String, in manager: TerminalManager) -> String {{
        switch command.lowercased() {{
        case "clear":
            return "\\u{001b}[2J\\u{001b}[H" // Clear screen
        case "history":
            return manager.commandHistory.joined(separator: "\\n")
        case "pwd":
            return manager.currentDirectory
        case "help":
            return builtInCommands.map {{ "\($0.key): \($0.value)" }}.joined(separator: "\\n")
        default:
            return "Command not found: \\(command)"
        }}
    }}
}}'''
        
        return swift_code
    
    def generate_terminal_package_swift(self, app_name: str) -> str:
        """
        Generate Package.swift with SwiftTerm dependency.
        
        Args:
            app_name: Name of the application
            
        Returns:
            Package.swift content
        """
        return f'''// swift-tools-version:5.4
import PackageDescription

let package = Package(
    name: "{app_name}",
    platforms: [.macOS(.v11)],
    products: [
        .executable(name: "{app_name}", targets: ["{app_name}"])
    ],
    dependencies: [
        .package(url: "https://github.com/migueldeicaza/SwiftTerm.git", from: "1.0.0")
    ],
    targets: [
        .executableTarget(
            name: "{app_name}",
            dependencies: ["SwiftTerm"],
            path: "Sources/{app_name}",
            linkerSettings: [
                .linkedFramework("SwiftUI"),
                .linkedFramework("AppKit"),
                .linkedFramework("Foundation")
            ]
        )
    ]
)'''
    
    def generate_terminal_integration_swift(self, app_name: str, features: List[str] = None) -> str:
        """
        Generate complete terminal integration Swift code.
        
        Args:
            app_name: Name of the application
            features: List of terminal features
            
        Returns:
            Complete Swift code for terminal integration
        """
        features = features or []
        
        # Generate the main terminal integration code
        terminal_swift = self.generate_terminal_swift(features)
        
        # Add feature-specific code
        if 'git-integration' in features:
            terminal_swift += self._generate_git_integration()
        
        if 'swift-integration' in features:
            terminal_swift += self._generate_swift_integration()
        
        if 'python-integration' in features:
            terminal_swift += self._generate_python_integration()
        
        return terminal_swift
    
    def _generate_git_integration(self) -> str:
        """Generate Git integration code."""
        return '''

// MARK: - Git Integration
extension TerminalManager {
    func executeGitCommand(_ command: String) {
        let gitCommand = "git \(command)"
        executeCommand(gitCommand)
        
        sendTelemetry("git_command", {
            "command": command
        })
    }
    
    func getGitStatus() -> String {
        executeCommand("git status --porcelain")
        return "Git status updated"
    }
    
    func getGitBranch() -> String {
        executeCommand("git branch --show-current")
        return "Current branch displayed"
    }
}'''
    
    def _generate_swift_integration(self) -> str:
        """Generate Swift integration code."""
        return '''

// MARK: - Swift Integration
extension TerminalManager {
    func executeSwiftCommand(_ command: String) {
        let swiftCommand = "swift \(command)"
        executeCommand(swiftCommand)
        
        sendTelemetry("swift_command", {
            "command": command
        })
    }
    
    func runSwiftFile(_ file: String) {
        executeSwiftCommand("run \(file)")
    }
    
    func buildSwiftProject() {
        executeSwiftCommand("build")
    }
}'''
    
    def _generate_python_integration(self) -> str:
        """Generate Python integration code."""
        return '''

// MARK: - Python Integration
extension TerminalManager {
    func executePythonCommand(_ command: String) {
        let pythonCommand = "python \(command)"
        executeCommand(pythonCommand)
        
        sendTelemetry("python_command", {
            "command": command
        })
    }
    
    func runPythonFile(_ file: String) {
        executePythonCommand(file)
    }
    
    func installPythonPackage(_ package: String) {
        executeCommand("pip install \(package)")
    }
}'''
