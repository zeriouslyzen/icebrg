import SwiftUI

@main
struct CalculatorApp: App {
    var body: some Scene {
        WindowGroup {
            CalculatorView()
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
    }
}

struct CalculatorView: View {
    @StateObject private var viewModel = CalculatorViewModel()
    
    var body: some View {
        VStack(spacing: 12) {
            // Display
            HStack {
                Spacer()
                Text(viewModel.display)
                    .font(.system(size: 48, weight: .light, design: .monospaced))
                    .foregroundColor(.white)
                    .padding(.horizontal, 20)
            }
            .frame(height: 80)
            .background(Color.black)
            
            // Buttons
            VStack(spacing: 12) {
                // Row 1: Clear, +/-, %, ÷
                HStack(spacing: 12) {
                    CalculatorButton(title: "C", color: .gray, action: { viewModel.clear() })
                    CalculatorButton(title: "±", color: .gray, action: { viewModel.toggleSign() })
                    CalculatorButton(title: "%", color: .gray, action: { viewModel.percentage() })
                    CalculatorButton(title: "÷", color: .orange, action: { viewModel.setOperation(.divide) })
                }
                
                // Row 2: 7, 8, 9, ×
                HStack(spacing: 12) {
                    CalculatorButton(title: "7", color: .darkGray, action: { viewModel.inputNumber("7") })
                    CalculatorButton(title: "8", color: .darkGray, action: { viewModel.inputNumber("8") })
                    CalculatorButton(title: "9", color: .darkGray, action: { viewModel.inputNumber("9") })
                    CalculatorButton(title: "×", color: .orange, action: { viewModel.setOperation(.multiply) })
                }
                
                // Row 3: 4, 5, 6, -
                HStack(spacing: 12) {
                    CalculatorButton(title: "4", color: .darkGray, action: { viewModel.inputNumber("4") })
                    CalculatorButton(title: "5", color: .darkGray, action: { viewModel.inputNumber("5") })
                    CalculatorButton(title: "6", color: .darkGray, action: { viewModel.inputNumber("6") })
                    CalculatorButton(title: "-", color: .orange, action: { viewModel.setOperation(.subtract) })
                }
                
                // Row 4: 1, 2, 3, +
                HStack(spacing: 12) {
                    CalculatorButton(title: "1", color: .darkGray, action: { viewModel.inputNumber("1") })
                    CalculatorButton(title: "2", color: .darkGray, action: { viewModel.inputNumber("2") })
                    CalculatorButton(title: "3", color: .darkGray, action: { viewModel.inputNumber("3") })
                    CalculatorButton(title: "+", color: .orange, action: { viewModel.setOperation(.add) })
                }
                
                // Row 5: 0, ., =
                HStack(spacing: 12) {
                    CalculatorButton(title: "0", color: .darkGray, action: { viewModel.inputNumber("0") }, isWide: true)
                    CalculatorButton(title: ".", color: .darkGray, action: { viewModel.inputDecimal() })
                    CalculatorButton(title: "=", color: .orange, action: { viewModel.calculate() })
                }
            }
            .padding(20)
        }
        .background(Color.black)
        .cornerRadius(20)
        .frame(width: 300, height: 500)
    }
}

struct CalculatorButton: View {
    let title: String
    let color: Color
    let action: () -> Void
    var isWide: Bool = false
    
    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 32, weight: .medium))
                .foregroundColor(.white)
                .frame(width: isWide ? 132 : 60, height: 60)
                .background(color)
                .cornerRadius(30)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

enum Operation {
    case add, subtract, multiply, divide
}

class CalculatorViewModel: ObservableObject {
    @Published var display = "0"
    
    private var currentNumber: Double = 0
    private var previousNumber: Double = 0
    private var operation: Operation?
    private var isNewNumber = true
    
    func inputNumber(_ number: String) {
        if isNewNumber {
            display = number
            isNewNumber = false
        } else {
            display += number
        }
        currentNumber = Double(display) ?? 0
    }
    
    func inputDecimal() {
        if isNewNumber {
            display = "0."
            isNewNumber = false
        } else if !display.contains(".") {
            display += "."
        }
    }
    
    func setOperation(_ op: Operation) {
        if operation != nil && !isNewNumber {
            calculate()
        }
        operation = op
        previousNumber = currentNumber
        isNewNumber = true
    }
    
    func calculate() {
        guard let op = operation else { return }
        
        let result: Double
        switch op {
        case .add:
            result = previousNumber + currentNumber
        case .subtract:
            result = previousNumber - currentNumber
        case .multiply:
            result = previousNumber * currentNumber
        case .divide:
            result = currentNumber != 0 ? previousNumber / currentNumber : 0
        }
        
        display = formatNumber(result)
        currentNumber = result
        operation = nil
        isNewNumber = true
    }
    
    func clear() {
        display = "0"
        currentNumber = 0
        previousNumber = 0
        operation = nil
        isNewNumber = true
    }
    
    func toggleSign() {
        currentNumber = -currentNumber
        display = formatNumber(currentNumber)
    }
    
    func percentage() {
        currentNumber = currentNumber / 100
        display = formatNumber(currentNumber)
        isNewNumber = true
    }
    
    private func formatNumber(_ number: Double) -> String {
        if number == floor(number) {
            return String(format: "%.0f", number)
        } else {
            return String(number)
        }
    }
}

extension Color {
    static let darkGray = Color(red: 0.2, green: 0.2, blue: 0.2)
}
