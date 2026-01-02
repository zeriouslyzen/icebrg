import SwiftUI

@main
struct PomodoroApp: App {
    var body: some Scene {
        WindowGroup { PomodoroView() }
            .windowResizability(.contentSize)
    }
}

struct PomodoroView: View {
    @State private var seconds = 25 * 60
    @State private var running = false
    @State private var timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 12) {
            Text("Pomodoro")
                .font(.headline)
            Text(timeString)
                .font(.system(size: 48, weight: .bold, design: .monospaced))
            HStack {
                Button(running ? "Pause" : "Start") { running.toggle() }
                Button("Reset") { seconds = 25 * 60; running = false }
            }
        }
        .padding(20)
        .frame(width: 260, height: 200)
        .onReceive(timer) { _ in if running && seconds > 0 { seconds -= 1 } }
    }

    var timeString: String {
        String(format: "%02d:%02d", seconds / 60, seconds % 60)
    }
}
