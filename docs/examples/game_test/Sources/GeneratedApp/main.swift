import SwiftUI

@main
struct GameApp: App {
    var body: some Scene {
        WindowGroup {
            GameView()
        }
    }
}

struct GameView: View {
    @State private var score = 0
    @State private var gameState = "Ready to play"
    @State private var level = 1
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Game")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            Text(gameState)
                .font(.headline)
            
            Text("Score: \(score)")
                .font(.title2)
            
            Text("Level: \(level)")
                .font(.title3)
            
            Button("Play") {
                score += 1
                gameState = "Playing..."
                if score % 10 == 0 {
                    level += 1
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            
            Button("Reset") {
                score = 0
                level = 1
                gameState = "Ready to play"
            }
            .buttonStyle(.bordered)
        }
        .padding()
    }
}