"""
VS Code-like IDE Template for ICEBURG
Generates SwiftPM-based IDE applications with Monaco editor, terminal, and file explorer.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VSCodeProject:
    """VS Code-like project structure."""
    project_name: str
    project_path: str
    features: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class VSCodeTemplate:
    """
    VS Code-like IDE template generator.
    
    Features:
    - Monaco editor integration
    - Terminal with SwiftTerm
    - File explorer
    - LSP client
    - Git operations
    - SwiftPM project structure
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize VS Code template.
        
        Args:
            config: Template configuration
        """
        self.config = config or {}
        self.template_dir = self.config.get("template_dir", "templates/vscode")
        self.swift_version = self.config.get("swift_version", "5.9")
        
        # Ensure template directory exists
        os.makedirs(self.template_dir, exist_ok=True)
    
    async def generate_project(self, 
        project_name: str,
                             description: str,
                             features: List[str] = None,
                             output_dir: str = None) -> VSCodeProject:
        """
        Generate VS Code-like IDE project.
        
        Args:
            project_name: Project name
            description: Project description
            features: IDE features to include
            output_dir: Output directory
            
        Returns:
            Generated VS Code project
        """
        if output_dir is None:
            output_dir = os.path.join(self.template_dir, project_name)
        
        project_path = Path(output_dir)
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Create project structure
        await self._create_project_structure(project_path)
        
        # Generate SwiftPM files
        await self._generate_package_swift(project_path, project_name)
        
        # Generate main app
        await self._generate_main_app(project_path, project_name, features or [])
        
        # Generate Monaco editor integration
        await self._generate_monaco_integration(project_path)
        
        # Generate terminal integration
        await self._generate_terminal_integration(project_path)
        
        # Generate file explorer
        await self._generate_file_explorer(project_path)
        
        # Generate LSP client
        await self._generate_lsp_client(project_path)
        
        # Generate Git operations
        await self._generate_git_operations(project_path)
        
        # Generate project files
        await self._generate_project_files(project_path, project_name, description)
        
        project = VSCodeProject(
            project_name=project_name,
            project_path=str(project_path),
            features=features or [],
            metadata={
                "description": description,
                "swift_version": self.swift_version,
                "created_time": time.time()
            }
        )
        
        logger.info(f"Generated VS Code project: {project_name}")
        return project
    
    async def _create_project_structure(self, project_path: Path):
        """Create VS Code project directory structure."""
        directories = [
            "Sources/ICEBURGIDE",
            "Sources/ICEBURGIDE/Editor",
            "Sources/ICEBURGIDE/Terminal",
            "Sources/ICEBURGIDE/Explorer",
            "Sources/ICEBURGIDE/LSP",
            "Sources/ICEBURGIDE/Git",
            "Sources/ICEBURGIDE/UI",
            "Resources",
            "Resources/Monaco",
            "Resources/Themes",
            "Tests/ICEBURGIDETests"
        ]
        
        for directory in directories:
            (project_path / directory).mkdir(parents=True, exist_ok=True)
    
    async def _generate_package_swift(self, project_path: Path, project_name: str):
        """Generate Package.swift file."""
        package_swift = f"""// swift-tools-version:{self.swift_version}
import PackageDescription

let package = Package(
    name: "{project_name}",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "{project_name}",
            targets: ["{project_name}"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/pointfreeco/swift-composable-architecture", from: "1.0.0"),
        .package(url: "https://github.com/migueldeicaza/SwiftTerm", from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-log", from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.0.0")
    ],
    targets: [
        .executableTarget(
            name: "{project_name}",
            dependencies: [
                .product(name: "ComposableArchitecture", package: "swift-composable-architecture"),
                .product(name: "SwiftTerm", package: "SwiftTerm"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
            resources: [
                .process("Resources")
            ]
        ),
        .testTarget(
            name: "{project_name}Tests",
            dependencies: ["{project_name}"]
        )
    ]
)
"""
        
        with open(project_path / "Package.swift", "w") as f:
            f.write(package_swift)
    
    async def _generate_main_app(self, project_path: Path, project_name: str, features: List[str]):
        """Generate main app file."""
        main_swift = f"""import SwiftUI
import ComposableArchitecture
import Logging

@main
struct {project_name}App: App {{
    let store = Store(
        initialState: IDEState(),
        reducer: IDEReducer()
    )
    
    var body: some Scene {{
        WindowGroup {{
            IDEView(store: store)
                .frame(minWidth: 1200, minHeight: 800)
        }}
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified)
    }}
}}

struct IDEView: View {{
    let store: Store<IDEState, IDEAction>
    
    var body: some View {{
        WithViewStore(store) {{ viewStore in
            HSplitView {{
                // Sidebar
                SidebarView(store: store)
                    .frame(minWidth: 200, maxWidth: 400)
                
                // Main content
                VStack(spacing: 0) {{
                    // Tab bar
                    TabBarView(store: store)
                        .frame(height: 30)
                    
                    // Editor area
                    EditorAreaView(store: store)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    
                    // Status bar
                    StatusBarView(store: store)
                        .frame(height: 20)
                }}
            }}
            .onAppear {{
                viewStore.send(.appLaunched)
            }}
        }}
    }}
}}
"""
        
        with open(project_path / f"Sources/{project_name}/main.swift", "w") as f:
            f.write(main_swift)
    
    async def _generate_monaco_integration(self, project_path: Path):
        """Generate Monaco editor integration."""
        monaco_swift = """import SwiftUI
import WebKit

struct MonacoEditorView: NSViewRepresentable {
    @Binding var content: String
    @Binding var language: String
    @Binding var theme: String
    
    func makeNSView(context: Context) -> WKWebView {
        let webView = WKWebView()
        webView.navigationDelegate = context.coordinator
        
        // Load Monaco editor
        if let monacoPath = Bundle.main.path(forResource: "monaco", ofType: "html") {
            let url = URL(fileURLWithPath: monacoPath)
            webView.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
        }
        
        return webView
    }
    
    func updateNSView(_ webView: WKWebView, context: Context) {
        // Update editor content
        let script = """
            editor.setValue('\(content.replacingOccurrences(of: "'", with: "\\'"))');
            monaco.editor.setModelLanguage(editor.getModel(), '\(language)');
            monaco.editor.setTheme('\(theme)');
        """
        
        webView.evaluateJavaScript(script)
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, WKNavigationDelegate {
        var parent: MonacoEditorView
        
        init(_ parent: MonacoEditorView) {
            self.parent = parent
        }
        
        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            // Initialize Monaco editor
            let initScript = """
                var editor = monaco.editor.create(document.getElementById('container'), {
                    value: '',
                    language: 'javascript',
                    theme: 'vs-dark',
                    automaticLayout: true,
                    minimap: { enabled: true },
                    scrollBeyondLastLine: false,
                    fontSize: 14,
                    lineNumbers: 'on',
                    wordWrap: 'on'
                });
            """
            
            webView.evaluateJavaScript(initScript)
        }
    }
}
"""
        
        with open(project_path / "Sources/ICEBURGIDE/Editor/MonacoEditorView.swift", "w") as f:
            f.write(monaco_swift)
    
    async def _generate_terminal_integration(self, project_path: Path):
        """Generate terminal integration with SwiftTerm."""
        terminal_swift = """import SwiftUI
import SwiftTerm

struct TerminalView: NSViewRepresentable {
    @Binding var currentDirectory: String
    @Binding var commandHistory: [String]
    
    func makeNSView(context: Context) -> TerminalView {
        let terminal = TerminalView()
        terminal.setupTerminal()
        return terminal
    }
    
    func updateNSView(_ terminal: TerminalView, context: Context) {
        // Update terminal state
        terminal.setCurrentDirectory(currentDirectory)
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject {
        var parent: TerminalView
        
        init(_ parent: TerminalView) {
            self.parent = parent
        }
    }
}

extension TerminalView {
    func setupTerminal() {
        // Configure terminal appearance
        backgroundColor = NSColor.black
        textColor = NSColor.green
        
        // Set up command handling
        onCommand = { [weak self] command in
            self?.handleCommand(command)
        }
    }
    
    func handleCommand(_ command: String) {
        // Add to history
        commandHistory.append(command)
        
        // Execute command
        executeCommand(command)
    }
    
    func executeCommand(_ command: String) {
        // Simple command execution
        let task = Process()
        task.launchPath = "/bin/bash"
        task.arguments = ["-c", command]
        
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        
        task.launch()
        task.waitUntilExit()
        
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        if let output = String(data: data, encoding: .utf8) {
            appendOutput(output)
        }
    }
}
"""
        
        with open(project_path / "Sources/ICEBURGIDE/Terminal/TerminalView.swift", "w") as f:
            f.write(terminal_swift)
    
    async def _generate_file_explorer(self, project_path: Path):
        """Generate file explorer."""
        explorer_swift = """import SwiftUI
import Foundation

struct FileExplorerView: View {
    @Binding var selectedPath: String
    @Binding var currentDirectory: String
    @State private var fileTree: [FileItem] = []
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Toolbar
            HStack {
                Button("Refresh") {
                    loadFileTree()
                }
                .buttonStyle(.borderless)
                
                Spacer()
                
                Button("New File") {
                    createNewFile()
                }
                .buttonStyle(.borderless)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color(NSColor.controlBackgroundColor))
            
            // File tree
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(fileTree, id: \\.path) { item in
                        FileItemView(
                            item: item,
                            selectedPath: $selectedPath,
                            onSelect: { path in
                                selectedPath = path
                            }
                        )
                    }
                }
                .padding(.horizontal, 8)
            }
        }
        .onAppear {
            loadFileTree()
        }
    }
    
    func loadFileTree() {
        fileTree = loadDirectory(currentDirectory)
    }
    
    func loadDirectory(_ path: String) -> [FileItem] {
        let url = URL(fileURLWithPath: path)
        
        do {
            let contents = try FileManager.default.contentsOfDirectory(
                at: url,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )
            
            return contents.map { url in
                let isDirectory = (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
                return FileItem(
                    name: url.lastPathComponent,
                    path: url.path,
                    isDirectory: isDirectory,
                    children: isDirectory ? loadDirectory(url.path) : []
                )
            }.sorted { $0.name < $1.name }
            
        } catch {
            return []
        }
    }
    
    func createNewFile() {
        // Create new file logic
    }
}

struct FileItem {
    let name: String
    let path: String
    let isDirectory: Bool
    let children: [FileItem]
}

struct FileItemView: View {
    let item: FileItem
    @Binding var selectedPath: String
    let onSelect: (String) -> Void
    @State private var isExpanded = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Button(action: {
                    if item.isDirectory {
                        isExpanded.toggle()
                    } else {
                        onSelect(item.path)
                    }
                }) {
                    HStack {
                        Image(systemName: item.isDirectory ? (isExpanded ? "folder.fill" : "folder") : "doc.text")
                            .foregroundColor(item.isDirectory ? .blue : .primary)
                        
                        Text(item.name)
                            .foregroundColor(selectedPath == item.path ? .blue : .primary)
                    }
                }
                .buttonStyle(.plain)
                
                Spacer()
            }
            .padding(.leading, CGFloat(item.path.components(separatedBy: "/").count - 1) * 16)
            
            // Children
            if isExpanded && !item.children.isEmpty {
                ForEach(item.children, id: \\.path) { child in
                    FileItemView(
                        item: child,
                        selectedPath: $selectedPath,
                        onSelect: onSelect
                    )
                }
            }
        }
    }
}
"""
        
        with open(project_path / "Sources/ICEBURGIDE/Explorer/FileExplorerView.swift", "w") as f:
            f.write(explorer_swift)
    
    async def _generate_lsp_client(self, project_path: Path):
        """Generate LSP client."""
        lsp_swift = """import Foundation
import Network

class LSPClient: ObservableObject {
    @Published var isConnected = false
    @Published var diagnostics: [Diagnostic] = []
    
    private var connection: NWConnection?
    private let serverPath: String
    private let serverArgs: [String]
    
    init(serverPath: String, serverArgs: [String] = []) {
        self.serverPath = serverPath
        self.serverArgs = serverArgs
    }
    
    func connect() {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: serverPath)
        process.arguments = serverArgs
        
        let inputPipe = Pipe()
        let outputPipe = Pipe()
        
        process.standardInput = inputPipe
        process.standardOutput = outputPipe
        
        do {
            try process.run()
            setupConnection(input: inputPipe, output: outputPipe)
        } catch {
        }
    }
    
    private func setupConnection(input: Pipe, output: Pipe) {
        // Set up bidirectional communication
        input.fileHandleForWriting.writeHandler = { [weak self] data in
            self?.handleServerMessage(data)
        }
        
        output.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if !data.isEmpty {
                self?.handleServerMessage(data)
            }
        }
    }
    
    func sendRequest(_ request: LSPRequest) {
        let jsonData = try! JSONEncoder().encode(request)
        let jsonString = String(data: jsonData, encoding: .utf8)!
        let message = "Content-Length: \\(jsonString.utf8.count)\\r\\n\\r\\n\\(jsonString)"
        
        // Send to LSP server
        if let data = message.data(using: .utf8) {
            // Send via process input
        }
    }
    
    private func handleServerMessage(_ data: Data) {
        // Parse LSP response
        if let response = try? JSONDecoder().decode(LSPResponse.self, from: data) {
            handleLSPResponse(response)
        }
    }
    
    private func handleLSPResponse(_ response: LSPResponse) {
        switch response.method {
        case "textDocument/publishDiagnostics":
            if let params = response.params as? PublishDiagnosticsParams {
                diagnostics = params.diagnostics
            }
        default:
            break
        }
    }
}

struct Diagnostic {
    let range: LSPRange
    let severity: Int
    let message: String
    let source: String?
}

struct LSPRange {
    let start: LSPPosition
    let end: LSPPosition
}

struct LSPPosition {
    let line: Int
    let character: Int
}

struct LSPRequest {
    let jsonrpc = "2.0"
    let id: Int
    let method: String
    let params: [String: Any]
}

struct LSPResponse {
    let jsonrpc: String
    let id: Int?
    let method: String?
    let params: Any?
    let result: Any?
    let error: LSPError?
}

struct LSPError {
    let code: Int
    let message: String
}

struct PublishDiagnosticsParams {
    let uri: String
    let diagnostics: [Diagnostic]
}
"""
        
        with open(project_path / "Sources/ICEBURGIDE/LSP/LSPClient.swift", "w") as f:
            f.write(lsp_swift)
    
    async def _generate_git_operations(self, project_path: Path):
        """Generate Git operations."""
        git_swift = """import Foundation
import Process

class GitOperations: ObservableObject {
    @Published var status: GitStatus = .clean
    @Published var branches: [String] = []
    @Published var currentBranch: String = "main"
    
    private let workingDirectory: String
    
    init(workingDirectory: String) {
        self.workingDirectory = workingDirectory
    }
    
    func initialize() {
        executeGitCommand(["init"])
    }
    
    func add(_ files: [String]) {
        executeGitCommand(["add"] + files)
    }
    
    func commit(_ message: String) {
        executeGitCommand(["commit", "-m", message])
    }
    
    func push() {
        executeGitCommand(["push"])
    }
    
    func pull() {
        executeGitCommand(["pull"])
    }
    
    func checkout(_ branch: String) {
        executeGitCommand(["checkout", branch])
        currentBranch = branch
    }
    
    func createBranch(_ name: String) {
        executeGitCommand(["checkout", "-b", name])
        loadBranches()
    }
    
    func loadStatus() {
        let result = executeGitCommand(["status", "--porcelain"])
        status = result.isEmpty ? .clean : .modified
    }
    
    func loadBranches() {
        let result = executeGitCommand(["branch", "--list"])
        branches = result.components(separatedBy: "\\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }
    
    private func executeGitCommand(_ arguments: [String]) -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = arguments
        process.currentDirectoryURL = URL(fileURLWithPath: workingDirectory)
        
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        
        do {
            try process.run()
            process.waitUntilExit()
            
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            return String(data: data, encoding: .utf8) ?? ""
        } catch {
            return ""
        }
    }
}

enum GitStatus {
    case clean
    case modified
    case conflicted
}
"""
        
        with open(project_path / "Sources/ICEBURGIDE/Git/GitOperations.swift", "w") as f:
            f.write(git_swift)
    
    async def _generate_project_files(self, project_path: Path, project_name: str, description: str):
        """Generate project files."""
        # README
        readme_content = f"""# {project_name}

{description}

## VS Code-like IDE

This project was generated by ICEBURG VS Code template.

### Features

- Monaco editor integration
- Terminal with SwiftTerm
- File explorer
- LSP client
- Git operations
- SwiftPM project structure

### Requirements

- macOS 13.0+
- Swift {self.swift_version}+
- Xcode 15.0+

### Building

```bash
swift build
swift run {project_name}
```

### Development

```bash
swift test
```

### Project Structure

- `Sources/{project_name}/` - Main application code
- `Sources/{project_name}/Editor/` - Monaco editor integration
- `Sources/{project_name}/Terminal/` - Terminal with SwiftTerm
- `Sources/{project_name}/Explorer/` - File explorer
- `Sources/{project_name}/LSP/` - Language Server Protocol client
- `Sources/{project_name}/Git/` - Git operations
- `Resources/` - Monaco editor and themes
"""
        
        with open(project_path / "README.md", "w") as f:
            f.write(readme_content)
        
        # Git ignore
        gitignore_content = """# Xcode
.DS_Store
*/build/*
*.pbxuser
!default.pbxuser
*.mode1v3
!default.mode1v3
*.mode2v3
!default.mode2v3
*.perspectivev3
!default.perspectivev3
xcuserdata/
*.moved-aside
*.xccheckout
*.xcscmblueprint

# Swift Package Manager
.build/
Packages/
Package.pins
Package.resolved
*.xcodeproj

# CocoaPods
Pods/

# Carthage
Carthage/Build/

# fastlane
fastlane/report.xml
fastlane/Preview.html
fastlane/screenshots/**/*.png
fastlane/test_output

# Code Injection
iOSInjectionProject/
"""
        
        with open(project_path / ".gitignore", "w") as f:
            f.write(gitignore_content)


# Convenience functions
async def create_vscode_template(config: Dict[str, Any] = None) -> VSCodeTemplate:
    """Create VS Code template."""
    return VSCodeTemplate(config)


async def generate_vscode_project(project_name: str,
    description: str,
                                template: VSCodeTemplate = None) -> VSCodeProject:
    """Generate VS Code project."""
    if template is None:
        template = await create_vscode_template()
    
    return await template.generate_project(
        project_name=project_name,
        description=description,
        features=["monaco_editor", "terminal", "file_explorer", "lsp_client", "git_operations"]
    )