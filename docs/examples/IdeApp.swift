import SwiftUI

@main
struct IdeApp: App {
    var body: some Scene {
        WindowGroup("ICEBURG IDE") {
            ContentView()
        }
    }
}

struct ContentView: View {
    @State private var text: String = "// ICEBURG IDE\n// Type here...\n"
    var body: some View {
        HSplitView {
            VStack(alignment: .leading) {
                Text("Explorer")
                    .font(.caption).bold()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(6)
                    .background(Color.gray.opacity(0.1))
                List {
                    Label("main.swift", systemImage: "swift")
                    Label("README.md", systemImage: "doc.text")
                    Label("Sources", systemImage: "folder")
                }
            }
            .frame(width: 220)
            
            VStack(spacing: 0) {
                HStack {
                    Label("main.swift", systemImage: "swift")
                    Spacer()
                }
                .padding(6)
                .background(Color.gray.opacity(0.08))
                
                TextEditor(text: $text)
                    .font(.system(.body, design: .monospaced))
                    .frame(minWidth: 600, minHeight: 400)
            }
        }
    }
}
