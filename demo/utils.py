import math
import torch
import random
import numpy as np

def set_seed(seed=42):
    """Initialize all random number generators for reproducibility
    
    Ensures consistent results across runs by controlling:
    - NumPy's random state
    - Python's built-in random module
    - PyTorch's CPU and CUDA random states
    - cuDNN's deterministic algorithms
    
    Args:
        seed (int): Random seed value (default: 42)
    """
    # NumPy random number generation
    np.random.seed(seed)
    # Python random module
    random.seed(seed)

    # PyTorch CUDA random states
    torch.cuda.manual_seed(seed)        # Current GPU
    torch.cuda.manual_seed_all(seed)    # All GPUs

    # PyTorch CPU random state
    torch.manual_seed(seed)
    # cuDNN configuration for deterministic algorithms
    torch.backends.cudnn.deterministic = True  # Reproducible convolution operations
    torch.backends.cudnn.benchmark = False     # Disable auto-tuner for consistency

def next_batch(X1, X2, batch_size):
    """Generate mini-batches from two aligned datasets
    
    Args:
        X1 (array-like): First dataset
        X2 (array-like): Second dataset (same sample count as X1)
        batch_size (int): Number of samples per batch
        
    Yields:
        tuple: (batch_x1, batch_x2, batch_index) where:
            - batch_x1: Batch from X1
            - batch_x2: Corresponding batch from X2
            - batch_index: 1-based batch counter
            
    Note: Final batch may contain fewer samples than batch_size
    """
    # Calculate total samples and required batches
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)  # Include partial final batch
    
    # Generate sequential batches
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)  # Handle final batch
        
        # Extract aligned batches from both datasets
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        
        # Yield batch data with 1-based index
        yield (batch_x1, batch_x2, (i + 1))

def normalize(x):
    """Perform min-max normalization to [0, 1] range
    
    Formula: (x - min(x)) / (max(x) - min(x))
    
    Args:
        x (array-like): Input data to normalize
        
    Returns:
        array-like: Normalized data with same shape as input
    """
    # Compute normalization parameters
    x_min = np.min(x)
    x_range = np.max(x) - x_min
    
    # Apply normalization formula
    return (x - x_min) / x_range