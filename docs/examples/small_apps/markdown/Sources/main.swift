import SwiftUI

@main
struct MarkdownViewerApp: App {
    var body: some Scene {
        WindowGroup { MarkdownView() }
            .windowResizability(.contentSize)
    }
}

struct MarkdownView: View {
    @State private var text = "# Hello\n\nThis is **Markdown**."
    var body: some View {
        HStack(spacing: 0) {
            TextEditor(text: $text)
                .font(.system(.body, design: .monospaced))
                .frame(minWidth: 350, minHeight: 400)
            Divider()
            ScrollView {
                Text(.init(text))
                    .frame(minWidth: 350, alignment: .leading)
                    .padding(12)
            }
        }
        .frame(width: 720, height: 420)
    }
}
