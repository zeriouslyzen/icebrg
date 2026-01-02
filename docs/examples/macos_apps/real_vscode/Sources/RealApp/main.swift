import SwiftUI
import AppKit

@available(macOS 12.0, *)
struct RealApp: App {
    init() {
        print("RealApp initialized. Attempting to open VS Code window...")
    }
    var body: some Scene {
        WindowGroup {
            VSCodeView()
        }
        // Removed .frame(minWidth: 800, minHeight: 600) as it's not supported
    }
}

struct VSCodeView: View {
    @State private var code = "// Welcome to VS Code\n// Start coding here...\n\nimport SwiftUI\n\nstruct ContentView: View {\n    var body: some View {\n        Text(\"Hello, World!\")\n    }\n}"
    @State private var selectedFile = "main.swift"
    @State private var files: [FileItem] = []
    @State private var showTerminal = true
    @State private var showGit = true
    @State private var showExplorer = true
    @State private var terminalOutput = ""
    @State private var command = ""
    @State private var gitStatus = "No changes"
    @State private var lineNumber = 1
    @State private var columnNumber = 1
    
    var body: some View {
        HSplitView {
            // Left Sidebar
            if showExplorer {
                VStack(alignment: .leading, spacing: 0) {
                    // Explorer Header
                    HStack {
                        Image(systemName: "folder")
                        Text("EXPLORER")
                            .font(.caption)
                            .fontWeight(.semibold)
                        Spacer()
                        Button(action: { showExplorer.toggle() }) {
                            Image(systemName: "chevron.left")
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.gray.opacity(0.1))
                    
                    // File Tree
                    ScrollView {
                        VStack(alignment: .leading, spacing: 2) {
                            ForEach(files) { file in
                                FileRowView(file: file, selectedFile: $selectedFile)
                            }
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                    }
                }
                .frame(width: 250)
                .background(Color(NSColor.controlBackgroundColor))
            }
            
            // Main Editor Area
            VStack(spacing: 0) {
                // Tab Bar
                HStack(spacing: 0) {
                    if !selectedFile.isEmpty {
                        HStack {
                            Image(systemName: "swift")
                                .foregroundColor(.orange)
                            Text(selectedFile)
                                .font(.system(.caption, design: .monospaced))
                            Button(action: { selectedFile = "" }) {
                                Image(systemName: "xmark")
                                    .font(.caption)
                            }
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(4)
                    }
                    Spacer()
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color.gray.opacity(0.05))
                
                // Code Editor
                HStack(spacing: 0) {
                    // Line Numbers
                    VStack(alignment: .trailing, spacing: 0) {
                        ForEach(Array(1...max(1, code.components(separatedBy: "\n").count)), id: \.self) { line in
                            Text("\(line)")
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(.secondary)
                                .frame(height: 20)
                        }
                    }
                    .frame(width: 50)
                    .padding(.horizontal, 8)
                    .background(Color.gray.opacity(0.05))
                    
                    // Code Text
                    TextEditor(text: $code)
                        .font(.system(.body, design: .monospaced))
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .background(Color(NSColor.textBackgroundColor))
                        .onChange(of: code) { _ in
                            updateLineNumbers()
                        }
                }
                
                // Status Bar
                HStack {
                    Text("\(lineNumber):\(columnNumber)")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Text("Swift")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.orange.opacity(0.2))
                        .cornerRadius(3)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 4)
                .background(Color.gray.opacity(0.1))
            }
            
            // Right Sidebar
            VStack(spacing: 0) {
                if showTerminal {
                    TerminalPanelView(output: $terminalOutput, command: $command)
                }
                
                if showGit {
                    GitPanelView(status: $gitStatus)
                }
            }
            .frame(width: 300)
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button(action: { showExplorer.toggle() }) {
                    Image(systemName: showExplorer ? "sidebar.left" : "sidebar.left")
                }
                
                Button(action: { showTerminal.toggle() }) {
                    Image(systemName: "terminal")
                }
                
                Button(action: { showGit.toggle() }) {
                    Image(systemName: "git.branch")
                }
                
                Divider()
                
                Button(action: { runCode() }) {
                    Image(systemName: "play.fill")
                }
                
                Button(action: { saveFile() }) {
                    Image(systemName: "square.and.arrow.down")
                }
            }
        }
        .onAppear {
            loadFiles()
        }
    }
    
    func loadFiles() {
        files = [
            FileItem(name: "main.swift", type: .swift, isFolder: false),
            FileItem(name: "Package.swift", type: .swift, isFolder: false),
            FileItem(name: "README.md", type: .markdown, isFolder: false),
            FileItem(name: "Sources", type: .folder, isFolder: true),
            FileItem(name: "Tests", type: .folder, isFolder: true)
        ]
    }
    
    func updateLineNumbers() {
        let lines = code.components(separatedBy: "\n")
        lineNumber = lines.count
        if let lastLine = lines.last {
            columnNumber = lastLine.count + 1
        }
    }
    
    func runCode() {
        terminalOutput += "$ swift run\n"
        terminalOutput += "Building project...\n"
        terminalOutput += "Running main.swift\n"
        terminalOutput += "Hello, World!\n"
        terminalOutput += "\n"
    }
    
    func saveFile() {
        terminalOutput += "$ File saved: \(selectedFile)\n"
    }
}

struct FileItem: Identifiable {
    let id = UUID()
    let name: String
    let type: FileType
    let isFolder: Bool
    
    enum FileType {
        case swift, markdown, folder, other
        
        var icon: String {
            switch self {
            case .swift: return "swift"
            case .markdown: return "doc.text"
            case .folder: return "folder"
            case .other: return "doc"
            }
        }
    }
}

struct FileRowView: View {
    let file: FileItem
    @Binding var selectedFile: String
    
    var body: some View {
        HStack {
            Image(systemName: file.type.icon)
                .foregroundColor(file.isFolder ? .blue : .orange)
                .frame(width: 16)
            
            Text(file.name)
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(selectedFile == file.name ? .white : .primary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 2)
        .background(selectedFile == file.name ? Color.blue : Color.clear)
        .cornerRadius(3)
        .onTapGesture {
            if !file.isFolder {
                selectedFile = file.name
            }
        }
    }
}

struct TerminalPanelView: View {
    @Binding var output: String
    @Binding var command: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Image(systemName: "terminal")
                Text("TERMINAL")
                    .font(.caption)
                    .fontWeight(.semibold)
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color.gray.opacity(0.1))
            
            ScrollView {
                Text(output.isEmpty ? "Terminal ready..." : output)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
            }
            .frame(height: 150)
            
            HStack {
                Text("$")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(.green)
                
                TextField("Enter command...", text: $command)
                    .font(.system(.caption, design: .monospaced))
                    .textFieldStyle(PlainTextFieldStyle())
                    // Removed onSubmit for compatibility
                    .onChange(of: command) { _ in
                        // Placeholder for command execution logic
                    }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color.gray.opacity(0.05))
        }
        .background(Color(NSColor.controlBackgroundColor))
    }
    
    func executeCommand() {
        output += "$ \(command)\n"
        output += "Command executed\n"
        command = ""
    }
}

struct GitPanelView: View {
    @Binding var status: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Image(systemName: "git.branch")
                Text("SOURCE CONTROL")
                    .font(.caption)
                    .fontWeight(.semibold)
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color.gray.opacity(0.1))
            
            VStack(alignment: .leading, spacing: 8) {
                Text(status)
                    .font(.system(.caption, design: .monospaced))
                    .padding(.horizontal, 8)
                
                Button("Commit Changes") {
                    status = "Changes committed to main branch"
                }
                .buttonStyle(BorderlessButtonStyle())
                .padding(.horizontal, 8)
                
                Button("Push to Remote") {
                    status = "Pushed to origin/main"
                }
                .buttonStyle(BorderlessButtonStyle())
                .padding(.horizontal, 8)
            }
            .padding(.vertical, 8)
        }
        .background(Color(NSColor.controlBackgroundColor))
    }
}