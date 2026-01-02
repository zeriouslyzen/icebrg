import SwiftUI
import Combine

class ViewModel: ObservableObject {
    @Published var inputText: String = ""
    @Published var state: [String: Any] = [:]
    
    func handleEvent(_ handlerId: String) {
        print("Event handler triggered: \(handlerId)")
        // Custom event handling logic
    }
    
    func handleAction() {
        print("Button action triggered")
    }
}
