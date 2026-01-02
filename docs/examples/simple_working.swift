import SwiftUI

struct SimpleWorking: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    var body: some View {
        VStack {
            Text("ICEBURG IDE")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            Text("This is a working IDE!")
                .font(.headline)
                .padding()
            
            Button("Test Button") {
                print("Button clicked!")
            }
            .padding()
        }
        .frame(width: 400, height: 300)
    }
}

// Main function
func main() {
    SimpleWorking.main()
}
