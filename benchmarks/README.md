# ICEBURG Elite Financial AI Benchmarking Suite

Comprehensive performance benchmarking for quantum, RL, and hybrid systems.

## Overview

This benchmarking suite provides detailed performance analysis for:

- **Quantum Computing**: VQCs, quantum kernels, quantum sampling, QGANs
- **Reinforcement Learning**: PPO, SAC, multi-agent systems, emergence detection
- **Hybrid Systems**: Quantum-RL integration, hybrid policies, quantum-enhanced agents

## Quick Start

### Run All Benchmarks

```bash
# Run all benchmarks
python benchmarks/run_benchmarks.py --type all

# Run specific benchmark types
python benchmarks/run_benchmarks.py --type quantum
python benchmarks/run_benchmarks.py --type rl
python benchmarks/run_benchmarks.py --type hybrid
```

### Individual Benchmark Modules

```bash
# Run quantum benchmarks
python benchmarks/quantum_performance.py

# Run RL benchmarks
python benchmarks/rl_performance.py

# Run hybrid benchmarks
python benchmarks/hybrid_performance.py
```

## Benchmark Types

### 1. Quantum Performance Benchmarks

**Quantum Circuits**:
- VQC performance across different qubit counts (2-10 qubits)
- Layer configurations (1-5 layers)
- Execution time and memory usage
- Quantum advantage measurement

**Quantum Sampling**:
- Quantum amplitude estimation
- Monte Carlo acceleration
- Shot configurations (100-10000 shots)
- Accuracy and performance metrics

**Quantum GANs**:
- Quantum generator performance
- Classical discriminator performance
- Hybrid training efficiency
- Financial data generation quality

**Quantum-RL Integration**:
- Quantum oracle performance
- Hybrid policy decision making
- Quantum-enhanced agent training
- Emergence detection capabilities

### 2. RL Performance Benchmarks

**Single Agent Performance**:
- PPO agent convergence rates
- SAC agent stability
- Training episode efficiency
- Final reward optimization

**Multi-Agent Performance**:
- Multi-agent coordination
- Emergent behavior detection
- Cartel formation analysis
- Nash equilibrium identification

**Environment Performance**:
- Trading environment simulation
- Order book execution speed
- Market simulator accuracy
- Real-time performance metrics

### 3. Hybrid Performance Benchmarks

**Quantum-RL Integration**:
- Quantum oracle query performance
- Hybrid policy decision speed
- Quantum-enhanced agent training
- Performance improvement over classical

**Quantum Advantage Measurement**:
- Quantum vs classical performance
- Speedup calculations
- Accuracy improvements
- Memory usage optimization

**Emergent Behavior Analysis**:
- Multi-agent emergence detection
- Cartel formation patterns
- Nash equilibrium identification
- System stability analysis

## Output Files

### Results Files

- `quantum_performance_results.json` - Quantum benchmark results
- `rl_performance_results.json` - RL benchmark results
- `hybrid_performance_results.json` - Hybrid benchmark results
- `comprehensive_benchmark_report.json` - Complete benchmark report

### CSV Files

- `quantum_performance_results.csv` - Quantum results in CSV format
- `rl_performance_results.csv` - RL results in CSV format
- `hybrid_performance_results.csv` - Hybrid results in CSV format

### Visualizations

- `quantum_performance_visualization.png` - Quantum performance charts
- `rl_performance_visualization.png` - RL performance charts
- `hybrid_performance_visualization.png` - Hybrid performance charts

## Performance Metrics

### Quantum Metrics

- **Execution Time**: Time to complete quantum operations
- **Memory Usage**: Memory consumption during execution
- **Quantum Advantage**: Performance improvement over classical
- **Accuracy**: Correctness of quantum computations
- **Speedup**: Classical time / Quantum time

### RL Metrics

- **Convergence Time**: Time to reach stable policy
- **Final Reward**: Maximum reward achieved
- **Convergence Rate**: Percentage of successful convergences
- **Stability**: Consistency of agent performance
- **Emergence Detection**: Detection of emergent behaviors

### Hybrid Metrics

- **Performance Improvement**: Hybrid vs classical performance
- **Quantum Advantage**: Quantum contribution to performance
- **Integration Efficiency**: Quantum-RL integration speed
- **Emergence Rate**: Rate of emergent behavior detection
- **System Stability**: Overall system stability

## Configuration

### Benchmark Parameters

```python
# Quantum configurations
qubit_configs = [2, 4, 6, 8, 10]
layer_configs = [1, 2, 3, 4, 5]
shot_configs = [100, 500, 1000, 5000, 10000]

# RL configurations
agent_configs = [1, 2, 4, 8, 16]
episode_configs = [1000, 5000, 10000, 20000, 50000]

# Hybrid configurations
integration_configs = {
    "n_qubits": [2, 4, 6, 8, 10],
    "n_layers": [1, 2, 3, 4, 5],
    "n_agents": [1, 2, 4, 8, 16]
}
```

### Output Directory

```bash
# Default output directory
benchmarks/results/

# Custom output directory
python benchmarks/run_benchmarks.py --output /path/to/results
```

## Advanced Usage

### Custom Benchmark Configurations

```python
from benchmarks.quantum_performance import QuantumPerformanceBenchmark

# Create custom benchmark
benchmark = QuantumPerformanceBenchmark(output_dir="custom_results")

# Run specific benchmarks
circuit_results = benchmark.benchmark_quantum_circuits()
sampling_results = benchmark.benchmark_quantum_sampling()
qgan_results = benchmark.benchmark_quantum_gan()
```

### Benchmark Analysis

```python
import pandas as pd
import json

# Load results
with open("benchmarks/results/quantum_performance_results.json", "r") as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(results["results"])

# Analyze performance
print(f"Average execution time: {df['execution_time'].mean():.4f}s")
print(f"Average quantum advantage: {df['quantum_advantage'].mean():.2%}")
print(f"Best performance: {df['quantum_advantage'].max():.2%}")
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install required packages
   ```bash
   pip install pennylane torch stable-baselines3 gymnasium
   ```

2. **Memory Issues**: Reduce benchmark configurations
   ```python
   # Reduce qubit counts for memory-constrained systems
   qubit_configs = [2, 4, 6]  # Instead of [2, 4, 6, 8, 10]
   ```

3. **Timeout Issues**: Increase timeout limits
   ```python
   # Increase timeout for long-running benchmarks
   timeout = 300  # 5 minutes
   ```

### Performance Optimization

1. **GPU Acceleration**: Enable CUDA for quantum simulations
2. **Parallel Processing**: Use multiprocessing for multiple benchmarks
3. **Memory Management**: Monitor memory usage during benchmarks
4. **Caching**: Enable result caching for repeated benchmarks

## Results Interpretation

### Quantum Performance

- **High Quantum Advantage**: Quantum algorithms provide significant speedup
- **Low Execution Time**: Efficient quantum circuit implementation
- **High Accuracy**: Reliable quantum computations
- **Memory Efficiency**: Optimal memory usage

### RL Performance

- **Fast Convergence**: Quick policy learning
- **High Final Reward**: Effective trading strategies
- **Stable Performance**: Consistent agent behavior
- **Emergent Behavior**: Detection of complex multi-agent interactions

### Hybrid Performance

- **Performance Improvement**: Hybrid systems outperform classical
- **Quantum Advantage**: Quantum components provide measurable benefit
- **Integration Efficiency**: Seamless quantum-RL integration
- **System Stability**: Robust hybrid system operation

## Contributing

To add new benchmarks:

1. Create benchmark class inheriting from base benchmark
2. Implement benchmark methods
3. Add to benchmark runner
4. Update documentation
5. Add tests for new benchmarks

## License

This benchmarking suite is part of the ICEBURG Elite Financial AI project.
