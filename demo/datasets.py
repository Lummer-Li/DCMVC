import numpy as np
import scipy.io as sio
from utils import normalize

def load_data(data_name):
    """Load and preprocess multi-view dataset from .mat file
    
    Args:
        data_name (str): Name identifier for the dataset
        
    Returns:
        X_list (list): List of normalized feature matrices per view
        Y (np.ndarray): Ground truth labels (int array)
    """
    X_list = []  # Stores normalized feature matrices for each view
    Y = None     # Placeholder for labels

    # Currently supports synthetic3d dataset
    if data_name in ['synthetic3d']:
        # Load MATLAB format data (assumes specific file structure)
        # Note: Path is hardcoded to '../../datasets/synthetic3d.mat'
        mat = sio.loadmat('../../datasets/synthetic3d.mat')
        
        # Extract multi-view features (3 views in this case)
        X = mat['X']  # Original data structure from .mat file
        
        # Process and normalize each view:
        # 1. Convert to float32 for numerical stability
        # 2. Apply sklearn.preprocessing.normalize (L2 normalization)
        # 3. Store in X_list
        X_list.append(normalize(X[0][0].astype('float32')))  # View 1
        X_list.append(normalize(X[1][0].astype('float32')))  # View 2
        X_list.append(normalize(X[2][0].astype('float32')))  # View 3
        
        # Process labels: squeeze to 1D and convert to integers
        Y = np.squeeze(mat['Y']).astype('int')  # Shape: (n_samples,)

    # Returns empty list and None for unsupported datasets
    return X_list, Y