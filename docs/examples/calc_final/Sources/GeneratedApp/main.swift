import SwiftUI

struct CalculatorApp: App {
    var body: some Scene {
        WindowGroup {
            CalculatorView()
        }
    }
}

struct CalculatorView: View {
    @State private var display = "0"
    @State private var previousNumber: Double = 0
    @State private var operation: String = ""
    @State private var waitingForNewNumber = true
    
    var body: some View {
        VStack(spacing: 12) {
            // Display
            Text(display)
                .font(.system(size: 48, weight: .light))
                .frame(maxWidth: .infinity, alignment: .trailing)
                .padding()
                .background(Color.black)
                .foregroundColor(.white)
            
            // Buttons
            VStack(spacing: 12) {
                HStack(spacing: 12) {
                    CalculatorButton("C", color: .gray) { clear() }
                    CalculatorButton("±", color: .gray) { toggleSign() }
                    CalculatorButton("%", color: .gray) { percentage() }
                    CalculatorButton("÷", color: .orange) { setOperation("÷") }
                }
                
                HStack(spacing: 12) {
                    CalculatorButton("7") { inputNumber("7") }
                    CalculatorButton("8") { inputNumber("8") }
                    CalculatorButton("9") { inputNumber("9") }
                    CalculatorButton("×", color: .orange) { setOperation("×") }
                }
                
                HStack(spacing: 12) {
                    CalculatorButton("4") { inputNumber("4") }
                    CalculatorButton("5") { inputNumber("5") }
                    CalculatorButton("6") { inputNumber("6") }
                    CalculatorButton("−", color: .orange) { setOperation("−") }
                }
                
                HStack(spacing: 12) {
                    CalculatorButton("1") { inputNumber("1") }
                    CalculatorButton("2") { inputNumber("2") }
                    CalculatorButton("3") { inputNumber("3") }
                    CalculatorButton("+", color: .orange) { setOperation("+") }
                }
                
                HStack(spacing: 12) {
                    CalculatorButton("0", width: 2) { inputNumber("0") }
                    CalculatorButton(".") { inputNumber(".") }
                    CalculatorButton("=", color: .orange) { calculate() }
                }
            }
        }
        .padding()
        .frame(width: 300, height: 500)
    }
    
    func inputNumber(_ number: String) {
        if waitingForNewNumber {
            display = number
            waitingForNewNumber = false
        } else {
            display += number
        }
    }
    
    func setOperation(_ op: String) {
        if !waitingForNewNumber {
            calculate()
        }
        previousNumber = Double(display) ?? 0
        operation = op
        waitingForNewNumber = true
    }
    
    func calculate() {
        let currentNumber = Double(display) ?? 0
        var result: Double = 0
        
        switch operation {
        case "+":
            result = previousNumber + currentNumber
        case "−":
            result = previousNumber - currentNumber
        case "×":
            result = previousNumber * currentNumber
        case "÷":
            result = currentNumber != 0 ? previousNumber / currentNumber : 0
        default:
            return
        }
        
        display = String(result)
        waitingForNewNumber = true
    }
    
    func clear() {
        display = "0"
        previousNumber = 0
        operation = ""
        waitingForNewNumber = true
    }
    
    func toggleSign() {
        if let number = Double(display) {
            display = String(-number)
        }
    }
    
    func percentage() {
        if let number = Double(display) {
            display = String(number / 100)
        }
    }
}

struct CalculatorButton: View {
    let title: String
    let color: Color
    let width: Int
    let action: () -> Void
    
    init(_ title: String, color: Color = .gray, width: Int = 1, action: @escaping () -> Void) {
        self.title = title
        self.color = color
        self.width = width
        self.action = action
    }
    
    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.title2)
                .fontWeight(.medium)
                .foregroundColor(.white)
                .frame(width: width == 2 ? 140 : 60, height: 60)
                .background(color)
                .cornerRadius(30)
        }
    }
}

// Main function
func main() {
    struct CalculatorApp.main()
}