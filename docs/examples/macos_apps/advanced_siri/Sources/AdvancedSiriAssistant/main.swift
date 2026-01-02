import SwiftUI
import AVFoundation
import Speech
import NaturalLanguage
import Algorithms
import Collections

@main
struct AdvancedSiriAssistantApp: App {
    var body: some Scene {
        WindowGroup {
            AdvancedSiriView()
        }
    }
}

struct AdvancedSiriView: View {
    @StateObject private var siri = AdvancedSiriManager()
    @State private var isListening = false
    @State private var response = ""
    @State private var conversationHistory: [String] = []
    @State private var isProcessing = false
    
    var body: some View {
        VStack(spacing: 20) {
            // Enhanced microphone button with animation
            Button(action: {
                if isListening {
                    siri.stopListening()
                } else {
                    siri.startListening()
                }
                isListening.toggle()
            }) {
                ZStack {
                    Circle()
                        .fill(isListening ? Color.red.opacity(0.3) : Color.blue.opacity(0.3))
                        .frame(width: 120, height: 120)
                        .scaleEffect(isListening ? 1.2 : 1.0)
                        .animation(.easeInOut(duration: 0.5), value: isListening)
                    
                    Image(systemName: isListening ? "mic.fill" : "mic")
                        .font(.system(size: 60))
                        .foregroundColor(isListening ? .red : .blue)
                }
            }
            .buttonStyle(PlainButtonStyle())
            
            // Enhanced status with processing indicator
            HStack {
                if isProcessing {
                    ProgressView()
                        .scaleEffect(0.8)
                }
                Text(isListening ? "Listening..." : isProcessing ? "Processing..." : "Tap to speak")
                    .font(.headline)
                    .foregroundColor(.secondary)
            }
            
            // Enhanced response area with conversation history
            VStack(alignment: .leading, spacing: 8) {
                Text("Conversation History")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 4) {
                        ForEach(conversationHistory, id: \.self) { message in
                            Text(message)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(Color.blue.opacity(0.1))
                                .cornerRadius(8)
                        }
                        
                        if !response.isEmpty {
                            Text(response)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(Color.green.opacity(0.1))
                                .cornerRadius(8)
                        }
                    }
                }
            }
            .frame(height: 200)
            .background(Color.gray.opacity(0.1))
            .cornerRadius(10)
        }
        .padding()
        .frame(width: 400, height: 500)
        .onReceive(siri.$response) { newResponse in
            response = newResponse
            if !newResponse.isEmpty {
                conversationHistory.append(newResponse)
            }
        }
        .onReceive(siri.$isProcessing) { processing in
            isProcessing = processing
        }
    }
}

class AdvancedSiriManager: NSObject, ObservableObject {
    @Published var response = ""
    @Published var isProcessing = false
    @Published var isListening = false
    
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
        guard !isListening else { return }
        
        do {
            try startAudioSession()
            try startRecognition()
            isListening = true
        } catch {
            response = "Error starting speech recognition: \(error.localizedDescription)"
        }
    }
    
    func stopListening() {
        guard isListening else { return }
        
        audioEngine.stop()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        isListening = false
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
        isProcessing = true
        let lowerText = text.lowercased()
        
        // Enhanced command processing with ICEBURG insights
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            if lowerText.contains("hello") || lowerText.contains("hi") {
                self.speak("Hello! I'm your advanced AI assistant. How can I help you today?")
            } else if lowerText.contains("weather") {
                self.speak("I can check the weather for you. What city would you like to know about?")
            } else if lowerText.contains("time") {
                let formatter = DateFormatter()
                formatter.timeStyle = .medium
                formatter.dateStyle = .full
                self.speak("The current time is \(formatter.string(from: Date()))")
            } else if lowerText.contains("calendar") {
                self.speak("I can help with calendar management. Would you like to create an event or check your schedule?")
            } else if lowerText.contains("search") {
                self.speak("I can perform web searches for you. What would you like me to search for?")
            } else if lowerText.contains("ai") || lowerText.contains("artificial intelligence") {
                self.speak("I'm powered by advanced AI technology, including natural language processing and machine learning.")
            } else if lowerText.contains("help") {
                self.speak("I can help with weather, time, calendar, web search, and general conversation. What would you like to do?")
            } else {
                self.speak("I heard: \(text). How can I assist you with that?")
            }
            self.isProcessing = false
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