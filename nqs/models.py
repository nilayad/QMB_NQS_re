"""Neural Quantum State architectures.

Implements various neural network architectures for representing quantum wavefunctions.
"""

import tensorflow as tf
from tensorflow import keras
from typing import Optional, List, Tuple
import numpy as np


class NQSBase(keras.Model):
    """Base class for Neural Quantum States.
    
    All NQS models should inherit from this and implement forward pass
    that returns log|ψ(s)|.
    """
    
    def __init__(self, num_sites: int, **kwargs):
        """Initialize base NQS.
        
        Args:
            num_sites: Number of lattice sites
        """
        super().__init__(**kwargs)
        self.num_sites = num_sites
    
    def call(self, spins: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Compute log|ψ(s)| for spin configurations.
        
        Args:
            spins: Spin configurations shape (batch_size, num_sites) with ±1 values
            training: Whether in training mode
            
        Returns:
            Log amplitude log|ψ(s)| shape (batch_size,)
        """
        raise NotImplementedError("Subclasses must implement call method")
    
    def amplitude(self, spins: tf.Tensor) -> tf.Tensor:
        """Compute amplitude |ψ(s)| (not log).
        
        Args:
            spins: Spin configurations
            
        Returns:
            Amplitude |ψ(s)|
        """
        return tf.exp(self.call(spins, training=False))
    
    def probability(self, spins: tf.Tensor) -> tf.Tensor:
        """Compute probability |ψ(s)|².
        
        Args:
            spins: Spin configurations
            
        Returns:
            Probability |ψ(s)|²
        """
        log_psi = self.call(spins, training=False)
        return tf.exp(2.0 * log_psi)


class RBM(NQSBase):
    """Restricted Boltzmann Machine for quantum states.
    
    ψ(s) = exp(Σᵢ aᵢsᵢ + Σⱼ log cosh(bⱼ + Σᵢ Wⱼᵢsᵢ))
    """
    
    def __init__(self, 
                 num_sites: int,
                 num_hidden: int = 2,
                 use_bias: bool = True,
                 dtype: tf.DType = tf.float32,
                 **kwargs):
        """Initialize RBM.
        
        Args:
            num_sites: Number of visible units (spins)
            num_hidden: Number of hidden units (default: 2x visible)
            use_bias: Include visible biases
            dtype: Data type for parameters
        """
        super().__init__(num_sites, **kwargs)
        
        if num_hidden <= 0:
            num_hidden = 2 * num_sites
        
        self.num_hidden = num_hidden
        self.use_bias = use_bias
        self._dtype = dtype
        
        # Initialize parameters
        self._build_parameters()
    
    def _build_parameters(self):
        """Initialize RBM parameters."""
        # Visible biases
        if self.use_bias:
            self.a = self.add_weight(
                name='visible_bias',
                shape=(self.num_sites,),
                initializer=keras.initializers.RandomNormal(stddev=0.01),
                dtype=self._dtype,
                trainable=True
            )
        
        # Hidden biases
        self.b = self.add_weight(
            name='hidden_bias',
            shape=(self.num_hidden,),
            initializer=keras.initializers.RandomNormal(stddev=0.01),
            dtype=self._dtype,
            trainable=True
        )
        
        # Weights
        self.W = self.add_weight(
            name='weights',
            shape=(self.num_sites, self.num_hidden),
            initializer=keras.initializers.GlorotNormal(),
            dtype=self._dtype,
            trainable=True
        )
    
    @tf.function
    def call(self, spins: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Compute log|ψ(s)|.
        
        Args:
            spins: Shape (batch_size, num_sites) with ±1 values
            training: Training mode flag
            
        Returns:
            log|ψ(s)| shape (batch_size,)
        """
        # Ensure correct dtype
        spins = tf.cast(spins, self._dtype)
        
        # Visible contribution: Σᵢ aᵢsᵢ
        if self.use_bias:
            visible_contrib = tf.reduce_sum(spins * self.a, axis=1)
        else:
            visible_contrib = 0.0
        
        # Hidden contribution: Σⱼ log cosh(bⱼ + Σᵢ Wⱼᵢsᵢ)
        hidden_input = self.b + tf.matmul(spins, self.W)
        
        # Use log_cosh for numerical stability
        hidden_contrib = tf.reduce_sum(self._log_cosh(hidden_input), axis=1)
        
        return visible_contrib + hidden_contrib
    
    @staticmethod
    @tf.function
    def _log_cosh(x: tf.Tensor) -> tf.Tensor:
        """Numerically stable log(cosh(x)).
        
        log(cosh(x)) = log((e^x + e^-x)/2) = |x| + log(1 + e^(-2|x|)) - log(2)
        """
        abs_x = tf.abs(x)
        return abs_x + tf.math.log1p(tf.exp(-2.0 * abs_x)) - tf.math.log(2.0)


class MLP(NQSBase):
    """Multi-layer perceptron for quantum states.
    
    ψ(s) = exp(MLP(s))
    """
    
    def __init__(self,
                 num_sites: int,
                 hidden_sizes: List[int] = [64, 64],
                 activation: str = 'tanh',
                 use_complex: bool = False,
                 dtype: tf.DType = tf.float32,
                 **kwargs):
        """Initialize MLP.
        
        Args:
            num_sites: Number of input sites
            hidden_sizes: List of hidden layer sizes
            activation: Activation function
            use_complex: Use complex-valued output (phase + amplitude)
            dtype: Data type
        """
        super().__init__(num_sites, **kwargs)
        
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.use_complex = use_complex
        self._dtype = dtype
        
        # Build layers
        self._build_layers()
    
    def _build_layers(self):
        """Construct MLP layers."""
        self.hidden_layers = []
        
        for i, size in enumerate(self.hidden_sizes):
            self.hidden_layers.append(
                keras.layers.Dense(
                    size,
                    activation=self.activation,
                    dtype=self._dtype,
                    name=f'hidden_{i}'
                )
            )
        
        # Output layer (log amplitude)
        self.output_layer = keras.layers.Dense(
            1, activation=None, dtype=self._dtype, name='log_amplitude'
        )
        
        if self.use_complex:
            # Additional output for phase
            self.phase_layer = keras.layers.Dense(
                1, activation=None, dtype=self._dtype, name='phase'
            )
    
    @tf.function
    def call(self, spins: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Compute log|ψ(s)|.
        
        Args:
            spins: Shape (batch_size, num_sites)
            training: Training mode
            
        Returns:
            log|ψ(s)| shape (batch_size,)
        """
        x = tf.cast(spins, self._dtype)
        
        # Forward pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        
        # Output
        log_psi = tf.squeeze(self.output_layer(x, training=training), axis=1)
        
        return log_psi
    
    def get_phase(self, spins: tf.Tensor) -> tf.Tensor:
        """Get phase of wavefunction (if complex).
        
        Args:
            spins: Spin configurations
            
        Returns:
            Phase angle
        """
        if not self.use_complex:
            return tf.zeros(tf.shape(spins)[0], dtype=self._dtype)
        
        x = tf.cast(spins, self._dtype)
        for layer in self.hidden_layers:
            x = layer(x, training=False)
        
        phase = tf.squeeze(self.phase_layer(x), axis=1)
        return phase


class ConvolutionalNQS(NQSBase):
    """Convolutional Neural Quantum State.
    
    Uses 1D convolutions to capture local correlations.
    """
    
    def __init__(self,
                 num_sites: int,
                 num_filters: List[int] = [16, 32],
                 kernel_sizes: List[int] = [3, 3],
                 activation: str = 'relu',
                 use_pooling: bool = False,
                 dtype: tf.DType = tf.float32,
                 **kwargs):
        """Initialize ConvNQS.
        
        Args:
            num_sites: Number of sites
            num_filters: Number of filters per conv layer
            kernel_sizes: Kernel sizes per conv layer
            activation: Activation function
            use_pooling: Use max pooling between conv layers
            dtype: Data type
        """
        super().__init__(num_sites, **kwargs)
        
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.use_pooling = use_pooling
        self._dtype = dtype
        
        self._build_layers()
    
    def _build_layers(self):
        """Build convolutional layers."""
        self.conv_layers = []
        self.pool_layers = []
        
        for i, (filters, kernel_size) in enumerate(zip(self.num_filters, self.kernel_sizes)):
            self.conv_layers.append(
                keras.layers.Conv1D(
                    filters,
                    kernel_size,
                    padding='same',
                    activation=self.activation,
                    dtype=self._dtype,
                    name=f'conv_{i}'
                )
            )
            
            if self.use_pooling:
                self.pool_layers.append(
                    keras.layers.MaxPooling1D(pool_size=2, padding='same')
                )
        
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(32, activation=self.activation, dtype=self._dtype)
        self.output_layer = keras.layers.Dense(1, activation=None, dtype=self._dtype)
    
    @tf.function
    def call(self, spins: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Compute log|ψ(s)|.
        
        Args:
            spins: Shape (batch_size, num_sites)
            training: Training mode
            
        Returns:
            log|ψ(s)|
        """
        # Reshape to (batch, sites, 1) for Conv1D
        x = tf.cast(spins, self._dtype)
        x = tf.expand_dims(x, axis=-1)
        
        # Convolutional layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, training=training)
            if self.use_pooling and i < len(self.pool_layers):
                x = self.pool_layers[i](x)
        
        # Flatten and dense
        x = self.flatten(x)
        x = self.dense(x, training=training)
        log_psi = tf.squeeze(self.output_layer(x, training=training), axis=1)
        
        return log_psi


class RecurrentNQS(NQSBase):
    """Recurrent Neural Quantum State.
    
    Uses autoregressive LSTM/GRU to model ψ(s) = Πᵢ p(sᵢ|s₁...sᵢ₋₁).
    """
    
    def __init__(self,
                 num_sites: int,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 cell_type: str = 'lstm',
                 dtype: tf.DType = tf.float32,
                 **kwargs):
        """Initialize RecurrentNQS.
        
        Args:
            num_sites: Number of sites
            hidden_size: RNN hidden state size
            num_layers: Number of RNN layers
            cell_type: 'lstm' or 'gru'
            dtype: Data type
        """
        super().__init__(num_sites, **kwargs)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        self._dtype = dtype
        
        self._build_layers()
    
    def _build_layers(self):
        """Build recurrent layers."""
        # Input embedding
        self.embedding = keras.layers.Dense(
            self.hidden_size, dtype=self._dtype, name='embedding'
        )
        
        # RNN layers
        self.rnn_layers = []
        for i in range(self.num_layers):
            if self.cell_type == 'lstm':
                rnn = keras.layers.LSTM(
                    self.hidden_size,
                    return_sequences=True,
                    return_state=False,
                    dtype=self._dtype,
                    name=f'lstm_{i}'
                )
            elif self.cell_type == 'gru':
                rnn = keras.layers.GRU(
                    self.hidden_size,
                    return_sequences=True,
                    return_state=False,
                    dtype=self._dtype,
                    name=f'gru_{i}'
                )
            else:
                raise ValueError(f"Unknown cell_type: {self.cell_type}")
            
            self.rnn_layers.append(rnn)
        
        # Output layer
        self.output_layer = keras.layers.Dense(1, dtype=self._dtype, name='output')
    
    @tf.function
    def call(self, spins: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Compute log|ψ(s)| using autoregressive factorization.
        
        log|ψ(s)| = Σᵢ log p(sᵢ|s₁...sᵢ₋₁)
        
        Args:
            spins: Shape (batch_size, num_sites)
            training: Training mode
            
        Returns:
            log|ψ(s)|
        """
        # Reshape to (batch, num_sites, 1)
        x = tf.cast(spins, self._dtype)
        x = tf.expand_dims(x, axis=-1)
        
        # Embed
        x = self.embedding(x, training=training)
        
        # RNN layers
        for rnn in self.rnn_layers:
            x = rnn(x, training=training)
        
        # Get conditional log probabilities
        log_probs = tf.squeeze(self.output_layer(x, training=training), axis=-1)
        
        # Sum over sites
        log_psi = tf.reduce_sum(log_probs, axis=1)
        
        return log_psi


def create_nqs_model(model_type: str, 
                     num_sites: int,
                     **kwargs) -> NQSBase:
    """Factory function to create NQS models.
    
    Args:
        model_type: One of 'rbm', 'mlp', 'conv', 'rnn'
        num_sites: Number of lattice sites
        **kwargs: Model-specific arguments
        
    Returns:
        NQS model instance
    """
    model_type = model_type.lower()
    
    if model_type == 'rbm':
        return RBM(num_sites, **kwargs)
    elif model_type == 'mlp':
        return MLP(num_sites, **kwargs)
    elif model_type == 'conv' or model_type == 'convolutional':
        return ConvolutionalNQS(num_sites, **kwargs)
    elif model_type == 'rnn' or model_type == 'recurrent':
        return RecurrentNQS(num_sites, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
