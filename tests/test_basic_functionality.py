#!/usr/bin/env python3
"""
Basic functionality test for Elite Financial AI Oracle
Tests core quantum, RL, and financial components
"""

import sys
import os
import numpy as np
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_quantum_functionality():
    """Test basic quantum computing functionality"""
    print("🔬 Testing Quantum Functionality...")
    
    try:
        import pennylane as qml
        print(f"  ✅ PennyLane version: {qml.__version__}")
        
        # Test quantum device
        dev = qml.device("default.qubit", wires=2)
        print("  ✅ Quantum device created")
        
        # Test simple quantum circuit
        @qml.qnode(dev)
        def simple_circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        
        result = simple_circuit()
        print(f"  ✅ Quantum circuit executed: {result}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Quantum test failed: {e}")
        return False

def test_rl_functionality():
    """Test basic RL functionality"""
    print("🤖 Testing RL Functionality...")
    
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        print(f"  ✅ Gymnasium version: {gym.__version__}")
        
        # Test basic environment
        env = gym.make("CartPole-v1")
        print("  ✅ RL environment created")
        
        # Test PPO agent
        model = PPO("MlpPolicy", env, verbose=0)
        print("  ✅ PPO agent created")
        
        # Test agent training (short)
        model.learn(total_timesteps=100)
        print("  ✅ RL agent trained")
        
        return True
        
    except Exception as e:
        print(f"  ❌ RL test failed: {e}")
        return False

def test_financial_functionality():
    """Test basic financial functionality"""
    print("💰 Testing Financial Functionality...")
    
    try:
        import pandas as pd
        import yfinance as yf
        print(f"  ✅ Pandas version: {pd.__version__}")
        
        # Test financial data retrieval
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="5d")
        print(f"  ✅ Financial data retrieved: {len(data)} records")
        
        # Test basic financial calculations
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        print(f"  ✅ Volatility calculated: {volatility:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Financial test failed: {e}")
        return False

def test_quantum_rl_integration():
    """Test quantum-RL integration"""
    print("🔗 Testing Quantum-RL Integration...")
    
    try:
        import pennylane as qml
        import torch
        import torch.nn as nn
        from stable_baselines3 import PPO
        import gymnasium as gym
        
        # Create quantum circuit
        dev = qml.device("default.qubit", wires=2)
        
        @qml.qnode(dev, interface="torch")
        def quantum_circuit(weights):
            qml.AngleEmbedding(weights, wires=range(2))
            qml.StronglyEntanglingLayers(weights.reshape(1, 2, 3), wires=range(2))
            return [qml.expval(qml.PauliZ(i)) for i in range(2)]
        
        # Test quantum circuit with PyTorch
        weights = torch.randn(2, 3, requires_grad=True)
        quantum_output = quantum_circuit(weights)
        print(f"  ✅ Quantum circuit with PyTorch: {quantum_output}")
        
        # Test RL environment
        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=0)
        
        # Test integration
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        print(f"  ✅ RL agent action: {action}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Quantum-RL integration test failed: {e}")
        return False

def test_performance():
    """Test system performance"""
    print("⚡ Testing Performance...")
    
    try:
        import time
        import numpy as np
        
        # Test quantum circuit performance
        import pennylane as qml
        dev = qml.device("default.qubit", wires=4)
        
        @qml.qnode(dev)
        def performance_circuit():
            for i in range(4):
                qml.Hadamard(wires=i)
            for i in range(3):
                qml.CNOT(wires=[i, i+1])
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]
        
        start_time = time.time()
        for _ in range(10):
            result = performance_circuit()
        quantum_time = time.time() - start_time
        
        print(f"  ✅ Quantum circuit (10 runs): {quantum_time:.4f}s")
        
        # Test RL performance
        from stable_baselines3 import PPO
        import gymnasium as gym
        
        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=0)
        
        start_time = time.time()
        model.learn(total_timesteps=1000)
        rl_time = time.time() - start_time
        
        print(f"  ✅ RL training (1000 steps): {rl_time:.4f}s")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Elite Financial AI Oracle - Basic Functionality Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Quantum Functionality", test_quantum_functionality),
        ("RL Functionality", test_rl_functionality),
        ("Financial Functionality", test_financial_functionality),
        ("Quantum-RL Integration", test_quantum_rl_integration),
        ("Performance", test_performance)
    ]
    
    results = {}
    total_tests = len(tests)
    passed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            success = test_func()
            results[test_name] = success
            if success:
                passed_tests += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}")
    
    if total_tests > 0:
        success_rate = (passed_tests / total_tests) * 100
        print(f"📈 Success Rate: {success_rate:.1f}%")
    
    print(f"\n⏱️  Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return success status
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
