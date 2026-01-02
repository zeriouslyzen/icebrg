import SwiftUI
import WebKit
import Foundation

struct Telemetry {
    static func write(_ event: String) {
        let url = URL(fileURLWithPath: "/tmp/iceburg_last_app_status.json")
        var dict: [String: Any] = [:]
        if let data = try? Data(contentsOf: url),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            dict = obj
        }
        dict[event] = true
        if let data = try? JSONSerialization.data(withJSONObject: dict) {
            try? data.write(to: url)
        }
    }
}

@main
struct IDEApp: App {
    var body: some Scene {
        WindowGroup {
            IDEView()
        }
    }
}

struct IDEView: View {
    @State private var selectedFile = ""
    @State private var files: [FileItem] = []
    @State private var showTerminal = true
    @State private var showExplorer = true
    @State private var terminalOutput = ""
    @State private var command = ""
    
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
                
                // Monaco Editor (WebKit)
                MonacoEditorView()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            
            // Right Sidebar - Terminal
            if showTerminal {
                TerminalPanelView(output: $terminalOutput, command: $command)
                    .frame(width: 300)
            }
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button(action: { showExplorer.toggle() }) {
                    Image(systemName: showExplorer ? "sidebar.left" : "sidebar.left")
                }
                
                Button(action: { showTerminal.toggle() }) {
                    Image(systemName: "terminal")
                }
                
                Button(action: { runCode() }) {
                    Image(systemName: "play.fill")
                }
            }
        }
        .onAppear {
            Telemetry.write("app_started")
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
    
    func runCode() {
        terminalOutput += "$ swift run\n"
        terminalOutput += "Building project...\n"
        terminalOutput += "Running main.swift\n"
        terminalOutput += "Hello, World!\n\n"
    }
}

struct MonacoEditorView: NSViewRepresentable {
    class Coordinator: NSObject, WKScriptMessageHandler {
        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            if message.name == "telemetry", let s = message.body as? String {
                Telemetry.write(s)
            }
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> WKWebView {
        let content = WKUserContentController()
        content.add(context.coordinator, name: "telemetry")
        let cfg = WKWebViewConfiguration()
        cfg.userContentController = content
        let webView = WKWebView(frame: .zero, configuration: cfg)
        let html = """
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs/loader.js"></script>
        </head>
        <body>
            <div id="container" style="height: 100vh; width: 100vw;"></div>
            <script>
                require.config({ paths: { 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs' }});
                require(['vs/editor/editor.main'], function() {
                    const editor = monaco.editor.create(document.getElementById('container'), {
                        value: '// Welcome to VS Code-like Editor\n// Start coding here...\n\nimport SwiftUI\n\nstruct ContentView: View {\n    var body: some View {\n        Text("Hello, World!")\n    }\n}',
                        language: 'swift',
                        theme: 'vs-dark',
                        automaticLayout: true,
                        minimap: { enabled: true },
                        lineNumbers: 'on',
                        folding: true,
                        wordWrap: 'on'
                    });
                    try { window.webkit.messageHandlers.telemetry.postMessage('monaco_loaded'); } catch (e) {}
                });
            </script>
        </body>
        </html>
        """
        webView.loadHTMLString(html, baseURL: nil)
        return webView
    }
    
    func updateNSView(_ nsView: WKWebView, context: Context) {}
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
                    .onChange(of: command) { newValue in
                        if newValue.hasSuffix("\n") {
                            output += "$ \(command)\n"
                            output += "Command executed\n"
                            command = ""
                        }
                    }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color.gray.opacity(0.05))
        }
        .background(Color(NSColor.controlBackgroundColor))
    }
}