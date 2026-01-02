import SwiftUI

struct WorkingIDEFinal: App {
    var body: some Scene {
        WindowGroup {
            IDEView()
        }
    }
}

struct IDEView: View {
    @State private var text = "// Welcome to ICEBURG IDE\n// Start coding here...\n\nimport SwiftUI\n\nstruct ContentView: View {\n    var body: some View {\n        Text(\"Hello, World!\")\n    }\n}"
    @State private var showTerminal = true
    @State private var terminalOutput = "Terminal ready...\n"
    @State private var showExplorer = true
    @State private var selectedFile = "main.swift"
    
    var body: some View {
        HSplitView {
            // Left Sidebar - File Explorer
            if showExplorer {
                VStack(alignment: .leading, spacing: 0) {
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
                    
                    ScrollView {
                        VStack(alignment: .leading, spacing: 2) {
                            FileRowView(name: "main.swift", selectedFile: $selectedFile)
                            FileRowView(name: "Package.swift", selectedFile: $selectedFile)
                            FileRowView(name: "README.md", selectedFile: $selectedFile)
                            FileRowView(name: "Sources", selectedFile: $selectedFile, isFolder: true)
                            FileRowView(name: "Tests", selectedFile: $selectedFile, isFolder: true)
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
                HStack {
                    HStack {
                        Image(systemName: "swift")
                            .foregroundColor(.orange)
                        Text(selectedFile)
                            .font(.system(.caption, design: .monospaced))
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(4)
                    Spacer()
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color.gray.opacity(0.05))
                
                // Editor
                TextEditor(text: $text)
                    .font(.system(.body, design: .monospaced))
                    .background(Color(NSColor.textBackgroundColor))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            
            // Right Sidebar - Terminal
            if showTerminal {
                VStack(alignment: .leading, spacing: 0) {
                    HStack {
                        Image(systemName: "terminal")
                        Text("TERMINAL")
                            .font(.caption)
                            .fontWeight(.semibold)
                        Spacer()
                        Button(action: { showTerminal.toggle() }) {
                            Image(systemName: "xmark")
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.gray.opacity(0.1))
                    
                    ScrollView {
                        Text(terminalOutput)
                            .font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(8)
                    }
                    .frame(height: 150)
                    .background(Color.black)
                    .foregroundColor(.green)
                }
                .frame(width: 300)
                .background(Color(NSColor.controlBackgroundColor))
            }
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button(action: { showExplorer.toggle() }) {
                    Image(systemName: "sidebar.left")
                }
                
                Button(action: { showTerminal.toggle() }) {
                    Image(systemName: "terminal")
                }
                
                Button(action: { runCode() }) {
                    Image(systemName: "play.fill")
                }
                
                Button(action: { buildProject() }) {
                    Image(systemName: "hammer.fill")
                }
            }
        }
    }
    
    func runCode() {
        terminalOutput += "$ swift run\n"
        terminalOutput += "Building project...\n"
        terminalOutput += "Running main.swift\n"
        terminalOutput += "Hello, World!\n\n"
    }
    
    func buildProject() {
        terminalOutput += "$ swift build\n"
        terminalOutput += "Building project...\n"
        terminalOutput += "Build successful!\n\n"
    }
}

struct FileRowView: View {
    let name: String
    @Binding var selectedFile: String
    let isFolder: Bool
    
    init(name: String, selectedFile: Binding<String>, isFolder: Bool = false) {
        self.name = name
        self._selectedFile = selectedFile
        self.isFolder = isFolder
    }
    
    var body: some View {
        HStack {
            Image(systemName: isFolder ? "folder" : "swift")
                .foregroundColor(isFolder ? .blue : .orange)
                .frame(width: 16)
            
            Text(name)
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(selectedFile == name ? .white : .primary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 2)
        .background(selectedFile == name ? Color.blue : Color.clear)
        .cornerRadius(3)
        .onTapGesture {
            if !isFolder {
                selectedFile = name
            }
        }
    }
}

// Main function
func main() {
    WorkingIDEFinal.main()
}