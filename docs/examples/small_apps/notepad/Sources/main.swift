import SwiftUI

@main
struct NotepadApp: App {
    var body: some Scene {
        WindowGroup {
            NotepadView()
        }
        .windowResizability(.contentSize)
    }
}

struct NotepadView: View {
    @State private var text = ""
    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Notepad")
                    .font(.headline)
                Spacer()
                Button("Clear") { text = "" }
            }
            .padding(8)
            .background(Color.gray.opacity(0.1))
            TextEditor(text: $text)
                .font(.system(.body, design: .monospaced))
                .frame(minWidth: 500, minHeight: 400)
        }
    }
}
