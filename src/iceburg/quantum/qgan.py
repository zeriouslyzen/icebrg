"""
Quantum GAN (QGAN) for ICEBURG Elite Financial AI

This module provides Quantum Generative Adversarial Networks for financial data generation,
including quantum generators, classical discriminators, and hybrid training loops.
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class QGANConfig:
    """Configuration for Quantum GAN."""
    n_qubits: int = 8
    n_layers: int = 3
    n_latent: int = 4
    n_outputs: int = 4
    device: str = "default.qubit"
    shots: int = 1000
    interface: str = "torch"
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    discriminator_lr: float = 0.001
    generator_lr: float = 0.001
    beta1: float = 0.5
    beta2: float = 0.999


class QuantumGenerator(nn.Module):
    """
    Quantum Generator for financial time series generation.
    
    Uses variational quantum circuits to generate synthetic financial data
    that captures quantum correlations and entanglement patterns.
    """
    
    def __init__(self, config: QGANConfig):
        """Initialize Quantum Generator."""
        super().__init__()
        
        self.config = config
        self.n_qubits = config.n_qubits
        self.n_latent = config.n_latent
        self.n_outputs = config.n_outputs
        
        # Initialize quantum device
        self.device = qml.device(config.device, wires=config.n_qubits, shots=config.shots)
        
        # Create quantum circuit
        self.qnode = qml.QNode(self._quantum_circuit, self.device, interface=config.interface)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize variational parameters."""
        # Latent encoding parameters
        self.latent_weights = nn.Parameter(torch.randn(self.n_latent, self.n_qubits))
        
        # Variational layer parameters
        self.variational_weights = nn.Parameter(
            torch.randn(self.config.n_layers, self.n_qubits, 3)
        )
        
        # Entangling layer parameters
        self.entangling_weights = nn.Parameter(
            torch.randn(self.config.n_layers, self.n_qubits - 1)
        )
        
        # Output measurement parameters
        self.output_weights = nn.Parameter(torch.randn(self.n_outputs, self.n_qubits))
    
    def _quantum_circuit(self, latent_inputs: torch.Tensor) -> torch.Tensor:
        """
        Define quantum circuit for generation.
        
        Args:
            latent_inputs: Latent noise inputs
            
        Returns:
            Generated outputs
        """
        # Encode latent inputs
        for i in range(self.n_latent):
            if i < self.n_qubits:
                qml.RY(latent_inputs[i] * self.latent_weights[i, i], wires=i)
        
        # Initialize remaining qubits
        for i in range(self.n_latent, self.n_qubits):
            qml.RY(torch.pi / 4, wires=i)
        
        # Variational layers
        for layer in range(self.config.n_layers):
            # Single-qubit rotations
            for qubit in range(self.n_qubits):
                qml.RX(self.variational_weights[layer, qubit, 0], wires=qubit)
                qml.RY(self.variational_weights[layer, qubit, 1], wires=qubit)
                qml.RZ(self.variational_weights[layer, qubit, 2], wires=qubit)
            
            # Entangling layer
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
                qml.RY(self.entangling_weights[layer, qubit], wires=qubit + 1)
        
        # Output measurements
        measurements = []
        for i in range(self.n_outputs):
            if i < self.n_qubits:
                measurements.append(qml.expval(qml.PauliZ(i)))
            else:
                measurements.append(qml.expval(qml.PauliZ(0)))
        
        return measurements
    
    def forward(self, latent_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum generator.
        
        Args:
            latent_inputs: Latent noise inputs
            
        Returns:
            Generated outputs
        """
        # Ensure inputs are the right size
        if latent_inputs.shape[-1] != self.n_latent:
            raise ValueError(f"Expected {self.n_latent} latent inputs, got {latent_inputs.shape[-1]}")
        
        # Process through quantum circuit
        outputs = self.qnode(latent_inputs)
        
        # Convert to tensor
        if isinstance(outputs, (list, tuple)):
            outputs = torch.stack(outputs)
        else:
            outputs = torch.tensor(outputs)
        
        # Apply output weights
        outputs = outputs * self.output_weights[:len(outputs)]
        
        return outputs
    
    def generate_samples(self, n_samples: int, device: str = "cpu") -> torch.Tensor:
        """
        Generate synthetic samples.
        
        Args:
            n_samples: Number of samples to generate
            device: Device for computation
            
        Returns:
            Generated samples
        """
        # Generate random latent inputs
        latent_inputs = torch.randn(n_samples, self.n_latent, device=device)
        
        # Generate samples
        with torch.no_grad():
            samples = self.forward(latent_inputs)
        
        return samples


class ClassicalDiscriminator(nn.Module):
    """
    Classical Discriminator for Quantum GAN.
    
    Uses classical neural networks to distinguish between
    real and generated financial data.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        """Initialize Classical Discriminator."""
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build discriminator network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator."""
        return self.network(x)


class QuantumGAN:
    """
    Quantum Generative Adversarial Network for financial data.
    
    Combines quantum generator with classical discriminator
    for generating realistic financial time series.
    """
    
    def __init__(self, config: QGANConfig):
        """Initialize Quantum GAN."""
        self.config = config
        
        # Initialize generator and discriminator
        self.generator = QuantumGenerator(config)
        self.discriminator = ClassicalDiscriminator(config.n_outputs)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.generator_lr,
            betas=(config.beta1, config.beta2)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.discriminator_lr,
            betas=(config.beta1, config.beta2)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        self.g_accuracies = []
        self.d_accuracies = []
    
    def train(self, real_data: torch.Tensor, epochs: int = None) -> Dict[str, List[float]]:
        """
        Train Quantum GAN.
        
        Args:
            real_data: Real financial data
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        if epochs is None:
            epochs = self.config.epochs
        
        # Create data loader
        dataset = TensorDataset(real_data)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            epoch_g_acc = 0
            epoch_d_acc = 0
            num_batches = 0
            
            for batch_data in dataloader:
                real_batch = batch_data[0]
                batch_size = real_batch.size(0)
                
                # Train Discriminator
                d_loss, d_acc = self._train_discriminator(real_batch, batch_size)
                
                # Train Generator
                g_loss, g_acc = self._train_generator(batch_size)
                
                # Accumulate losses
                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                epoch_g_acc += g_acc
                epoch_d_acc += d_acc
                num_batches += 1
            
            # Average losses
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            avg_g_acc = epoch_g_acc / num_batches
            avg_d_acc = epoch_d_acc / num_batches
            
            # Store history
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            self.g_accuracies.append(avg_g_acc)
            self.d_accuracies.append(avg_d_acc)
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: G_Loss={avg_g_loss:.4f}, D_Loss={avg_d_loss:.4f}, "
                          f"G_Acc={avg_g_acc:.4f}, D_Acc={avg_d_acc:.4f}")
        
        return {
            "g_losses": self.g_losses,
            "d_losses": self.d_losses,
            "g_accuracies": self.g_accuracies,
            "d_accuracies": self.d_accuracies
        }
    
    def _train_discriminator(self, real_batch: torch.Tensor, batch_size: int) -> Tuple[float, float]:
        """Train discriminator on real and fake data."""
        # Real data
        real_labels = torch.ones(batch_size, 1)
        real_output = self.discriminator(real_batch)
        real_loss = self.criterion(real_output, real_labels)
        
        # Fake data
        fake_latent = torch.randn(batch_size, self.config.n_latent)
        fake_batch = self.generator(fake_latent)
        fake_labels = torch.zeros(batch_size, 1)
        fake_output = self.discriminator(fake_batch.detach())
        fake_loss = self.criterion(fake_output, fake_labels)
        
        # Total discriminator loss
        d_loss = real_loss + fake_loss
        
        # Update discriminator
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        
        # Calculate accuracy
        d_acc = (torch.mean(real_output) + torch.mean(1 - fake_output)) / 2
        
        return d_loss.item(), d_acc.item()
    
    def _train_generator(self, batch_size: int) -> Tuple[float, float]:
        """Train generator to fool discriminator."""
        # Generate fake data
        fake_latent = torch.randn(batch_size, self.config.n_latent)
        fake_batch = self.generator(fake_latent)
        
        # Try to fool discriminator
        fake_labels = torch.ones(batch_size, 1)  # Want discriminator to think it's real
        fake_output = self.discriminator(fake_batch)
        g_loss = self.criterion(fake_output, fake_labels)
        
        # Update generator
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        
        # Calculate accuracy
        g_acc = torch.mean(fake_output)
        
        return g_loss.item(), g_acc.item()
    
    def generate(self, n_samples: int) -> torch.Tensor:
        """Generate synthetic samples."""
        return self.generator.generate_samples(n_samples)
    
    def evaluate(self, real_data: torch.Tensor, n_samples: int = 1000) -> Dict[str, float]:
        """Evaluate QGAN performance."""
        # Generate samples
        fake_data = self.generate(n_samples)
        
        # Calculate statistics
        real_mean = torch.mean(real_data, dim=0)
        fake_mean = torch.mean(fake_data, dim=0)
        real_std = torch.std(real_data, dim=0)
        fake_std = torch.std(fake_data, dim=0)
        
        # Calculate metrics
        mean_error = torch.mean(torch.abs(real_mean - fake_mean))
        std_error = torch.mean(torch.abs(real_std - fake_std))
        
        # Discriminator accuracy
        real_output = self.discriminator(real_data)
        fake_output = self.discriminator(fake_data)
        d_acc = (torch.mean(real_output) + torch.mean(1 - fake_output)) / 2
        
        return {
            "mean_error": mean_error.item(),
            "std_error": std_error.item(),
            "discriminator_accuracy": d_acc.item()
        }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss plots
        axes[0, 0].plot(self.g_losses, label='Generator Loss')
        axes[0, 0].plot(self.d_losses, label='Discriminator Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plots
        axes[0, 1].plot(self.g_accuracies, label='Generator Accuracy')
        axes[0, 1].plot(self.d_accuracies, label='Discriminator Accuracy')
        axes[0, 1].set_title('Training Accuracies')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Loss ratio
        loss_ratio = [g / d for g, d in zip(self.g_losses, self.d_losses)]
        axes[1, 0].plot(loss_ratio)
        axes[1, 0].set_title('Generator/Discriminator Loss Ratio')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Ratio')
        axes[1, 0].grid(True)
        
        # Accuracy difference
        acc_diff = [g - d for g, d in zip(self.g_accuracies, self.d_accuracies)]
        axes[1, 1].plot(acc_diff)
        axes[1, 1].set_title('Generator - Discriminator Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Difference')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def save_model(self, path: str):
        """Save QGAN model."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'config': self.config,
            'training_history': {
                'g_losses': self.g_losses,
                'd_losses': self.d_losses,
                'g_accuracies': self.g_accuracies,
                'd_accuracies': self.d_accuracies
            }
        }, path)
    
    def load_model(self, path: str):
        """Load QGAN model."""
        checkpoint = torch.load(path)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        # Load training history
        history = checkpoint['training_history']
        self.g_losses = history['g_losses']
        self.d_losses = history['d_losses']
        self.g_accuracies = history['g_accuracies']
        self.d_accuracies = history['d_accuracies']


class FinancialQGAN:
    """
    Specialized QGAN for financial time series generation.
    
    Optimized for generating realistic financial data with
    proper statistical properties and correlations.
    """
    
    def __init__(self, config: QGANConfig):
        """Initialize Financial QGAN."""
        self.config = config
        self.qgan = QuantumGAN(config)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, financial_data: np.ndarray) -> "FinancialQGAN":
        """
        Fit QGAN to financial data.
        
        Args:
            financial_data: Financial time series data
            
        Returns:
            Fitted Financial QGAN
        """
        # Normalize data
        financial_data_scaled = self.scaler.fit_transform(financial_data)
        
        # Convert to tensor
        real_data = torch.FloatTensor(financial_data_scaled)
        
        # Train QGAN
        self.qgan.train(real_data)
        
        self.is_fitted = True
        return self
    
    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate synthetic financial data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Generated financial data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generation")
        
        # Generate samples
        fake_data = self.qgan.generate(n_samples)
        
        # Convert to numpy
        fake_data_np = fake_data.detach().numpy()
        
        # Inverse transform
        fake_data_original = self.scaler.inverse_transform(fake_data_np)
        
        return fake_data_original
    
    def evaluate(self, real_data: np.ndarray, n_samples: int = 1000) -> Dict[str, float]:
        """Evaluate Financial QGAN performance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Generate samples
        fake_data = self.generate(n_samples)
        
        # Calculate financial metrics
        real_returns = np.diff(real_data, axis=0)
        fake_returns = np.diff(fake_data, axis=0)
        
        # Statistical tests
        from scipy import stats
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(real_returns.flatten(), fake_returns.flatten())
        
        # Mean and variance comparison
        real_mean = np.mean(real_returns)
        fake_mean = np.mean(fake_returns)
        real_var = np.var(real_returns)
        fake_var = np.var(fake_returns)
        
        mean_error = abs(real_mean - fake_mean)
        var_error = abs(real_var - fake_var)
        
        return {
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "mean_error": mean_error,
            "variance_error": var_error,
            "real_mean": real_mean,
            "fake_mean": fake_mean,
            "real_variance": real_var,
            "fake_variance": fake_var
        }


# Example usage and testing
if __name__ == "__main__":
    # Test Quantum GAN
    config = QGANConfig(n_qubits=4, n_layers=2, n_latent=4, n_outputs=4)
    qgan = QuantumGAN(config)
    
    # Generate sample data
    real_data = torch.randn(100, 4)
    
    # Train QGAN
    history = qgan.train(real_data, epochs=50)
    
    # Generate samples
    fake_data = qgan.generate(50)
    print(f"Generated data shape: {fake_data.shape}")
    
    # Evaluate performance
    metrics = qgan.evaluate(real_data)
    print(f"Evaluation metrics: {metrics}")
    
    # Test Financial QGAN
    financial_config = QGANConfig(n_qubits=6, n_layers=3, n_latent=6, n_outputs=6)
    financial_qgan = FinancialQGAN(financial_config)
    
    # Generate sample financial data
    financial_data = np.random.randn(100, 6)
    
    # Fit and generate
    financial_qgan.fit(financial_data)
    synthetic_data = financial_qgan.generate(50)
    print(f"Synthetic financial data shape: {synthetic_data.shape}")
    
    # Evaluate financial performance
    financial_metrics = financial_qgan.evaluate(financial_data)
    print(f"Financial evaluation metrics: {financial_metrics}")
