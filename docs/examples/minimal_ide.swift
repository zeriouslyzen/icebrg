import SwiftUI

struct MinimalIDE: App {
    var body: some Scene {
        WindowGroup {
            IDEView()
        }
    }
}

struct IDEView: View {
    @State private var text = "// Welcome to ICEBURG IDE\n// Start coding here...\n\nimport SwiftUI\n\nstruct ContentView: View {\n    var body: some View {\n        Text(\"Hello, World!\")\n    }\n}"
    
    var body: some View {
        HStack(spacing: 0) {
            // Left sidebar
            VStack(alignment: .leading) {
                Text("EXPLORER")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Image(systemName: "swift")
                            .foregroundColor(.orange)
                        Text("main.swift")
                            .font(.system(.caption, design: .monospaced))
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(3)
                    
                    HStack {
                        Image(systemName: "folder")
                            .foregroundColor(.blue)
                        Text("Sources")
                            .font(.system(.caption, design: .monospaced))
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                }
            }
            .frame(width: 200)
            .background(Color(NSColor.controlBackgroundColor))
            
            // Main editor
            VStack(spacing: 0) {
                // Tab bar
                HStack {
                    HStack {
                        Image(systemName: "swift")
                            .foregroundColor(.orange)
                        Text("main.swift")
                            .font(.system(.caption, design: .monospaced))
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(4)
                    Spacer()
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color.gray.opacity(0.05))
                
                // Editor
                TextEditor(text: $text)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color.black)
                    .foregroundColor(.green)
            }
        }
        .frame(width: 800, height: 600)
    }
}

// Main function
func main() {
    MinimalIDE.main()
}

// Ensure the app stays running
@main
struct AppLauncher {
    static func main() {
        MinimalIDE.main()
    }
}
