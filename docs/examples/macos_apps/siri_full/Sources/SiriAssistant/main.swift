import SwiftUI
import AVFoundation
import Speech
import NaturalLanguage

@main
struct SiriAssistantApp: App {
    var body: some Scene {
        WindowGroup {
            SiriView()
        }
    }
}

struct SiriView: View {
    @StateObject private var siri = SiriManager()
    @State private var isListening = false
    @State private var response = ""
    
    var body: some View {
        VStack(spacing: 20) {
            // Microphone button
            Button(action: {
                if isListening {
                    siri.stopListening()
                } else {
                    siri.startListening()
                }
                isListening.toggle()
            }) {
                Image(systemName: isListening ? "mic.fill" : "mic")
                    .font(.system(size: 60))
                    .foregroundColor(isListening ? .red : .blue)
            }
            .buttonStyle(PlainButtonStyle())
            
            // Status text
            Text(isListening ? "Listening..." : "Tap to speak")
                .font(.headline)
                .foregroundColor(.secondary)
            
            // Response area
            ScrollView {
                Text(response)
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 200)
            .background(Color.gray.opacity(0.1))
            .cornerRadius(10)
        }
        .padding()
        .frame(width: 400, height: 500)
        .onReceive(siri.$response) { newResponse in
            response = newResponse
        }
    }
}

class SiriManager: NSObject, ObservableObject {
    @Published var response = ""
    
    private let speechRecognizer = SFSpeechRecognizer()
    private let audioEngine = AVAudioEngine()
    private let speechSynthesizer = AVSpeechSynthesizer()
    
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    
    override init() {
        super.init()
        requestPermissions()
    }
    
    func startListening() {
        do {
            try startAudioSession()
            try startRecognition()
        } catch {
            response = "Error starting speech recognition: \(error.localizedDescription)"
        }
    }
    
    func stopListening() {
        audioEngine.stop()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
    }
    
    private func requestPermissions() {
        SFSpeechRecognizer.requestAuthorization { authStatus in
            DispatchQueue.main.async {
                if authStatus != .authorized {
                    self.response = "Speech recognition not authorized"
                }
            }
        }
    }
    
    private func startAudioSession() throws {
        // macOS doesn't use AVAudioSession - audio engine handles this automatically
        // Just ensure we have proper audio permissions
        return
    }
    
    private func startRecognition() throws {
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else {
            throw NSError(domain: "SiriManager", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unable to create recognition request"])
        }
        
        recognitionRequest.shouldReportPartialResults = true
        
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }
        
        audioEngine.prepare()
        try audioEngine.start()
        
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { result, error in
            if let result = result {
                let spokenText = result.bestTranscription.formattedString
                DispatchQueue.main.async {
                    self.processCommand(spokenText)
                }
            }
            
            if let error = error {
                DispatchQueue.main.async {
                    self.response = "Recognition error: \(error.localizedDescription)"
                }
            }
        }
    }
    
    private func processCommand(_ text: String) {
        let lowerText = text.lowercased()
        
        if lowerText.contains("hello") || lowerText.contains("hi") {
            speak("Hello! How can I help you today?")
        } else if lowerText.contains("weather") {
            speak("I'd need to connect to a weather API to get current conditions.")
        } else if lowerText.contains("time") {
            let formatter = DateFormatter()
            formatter.timeStyle = .medium
            speak("The current time is \(formatter.string(from: Date()))")
        } else if lowerText.contains("calendar") {
            speak("I can help with calendar management. What would you like to do?")
        } else if lowerText.contains("search") {
            speak("I can perform web searches. What would you like me to search for?")
        } else {
            speak("I heard: \(text). How can I assist you with that?")
        }
    }
    
    private func speak(_ text: String) {
        response = text
        
        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = 0.5
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        
        speechSynthesizer.speak(utterance)
    }
}