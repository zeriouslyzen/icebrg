import SwiftUI

struct MinimalWorkingIDE: App {
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
    
    var body: some View {
        HSplitView {
            // Main Editor Area
            VStack(spacing: 0) {
                // Tab Bar
                HStack {
                    HStack {
                        Image(systemName: "swift")
                            .foregroundColor(.orange)
                        Text("main.swift")
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
            
            // Terminal
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

// Main function
func main() {
    MinimalWorkingIDE.main()
}
