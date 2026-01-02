import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = ViewModel()
    
    var body: some View {
        Button("IRValue(name='Button', ty=ScalarType(dtype='string'))") {
            viewModel.handleEvent("handler_0")
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
