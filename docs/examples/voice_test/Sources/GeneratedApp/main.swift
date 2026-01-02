import SwiftUI
import AVFoundation
import Speech

@main
struct VoiceApp: App {
    var body: some Scene {
        WindowGroup {
            VoiceView()
        }
    }
}

struct VoiceView: View {
    @StateObject private var voice = VoiceManager()
    @State private var isListening = false
    @State private var conversation: [String] = []
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Voice Assistant")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(conversation, id: \.self) { message in
                        Text(message)
                            .padding()
                            .background(Color.blue.opacity(0.1))
                            .cornerRadius(8)
                    }
                }
                .padding()
            }
            .frame(height: 300)
            
            Button(action: {
                if isListening {
                    voice.stopListening()
                } else {
                    voice.startListening()
                }
                isListening.toggle()
            }) {
                Image(systemName: isListening ? "mic.fill" : "mic")
                    .font(.system(size: 60))
                    .foregroundColor(isListening ? .red : .blue)
            }
        }
        .onReceive(voice.$response) { response in
            if !response.isEmpty {
                conversation.append("Assistant: \(response)")
            }
        }
    }
}

class VoiceManager: NSObject, ObservableObject {
    @Published var response = ""
    
    private let speechRecognizer = SFSpeechRecognizer()
    private let audioEngine = AVAudioEngine()
    private let synthesizer = AVSpeechSynthesizer()
    
    override init() {
        super.init()
        requestPermissions()
    }
    
    func startListening() {
        do {
            try startRecognition()
        } catch {
            response = "Error: \(error.localizedDescription)"
        }
    }
    
    func stopListening() {
        audioEngine.stop()
    }
    
    private func startRecognition() throws {
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            // Process audio buffer
        }
        
        audioEngine.prepare()
        try audioEngine.start()
    }
    
    private func requestPermissions() {
        SFSpeechRecognizer.requestAuthorization { _ in }
    }
}