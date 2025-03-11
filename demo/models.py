import torch.nn as nn
import torch.nn.functional as F

# Models    
class Encoder(nn.Module):
    """Neural network encoder with batch normalization and softmax output
    
    Architecture: Sequential layers of [Linear => BatchNorm => ReLU] 
    followed by final Softmax activation
    """
    
    def __init__(self, input_dim, embedding_dim):
        """
        Args:
            input_dim (int): Dimension of input features
            embedding_dim (list): List of hidden layer dimensions
        """
        super(Encoder, self).__init__()

        # Construct layer dimensions: input + hidden layers
        encoder_dim = [input_dim]          # Start with input dimension
        encoder_dim.extend(embedding_dim)   # Add hidden layer dimensions
        self._dim = len(encoder_dim) - 1   # Number of weight layers

        # Build sequential layers
        encoder_layers = []
        for i in range(self._dim):
            # Linear transformation layer
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            # Add BN + ReLU for all but final layer
            if i < self._dim - 1:
                # Batch normalization for stability
                encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))  
                # Non-linear activation
                encoder_layers.append(nn.ReLU())      
        # Final softmax activation (produces probability distribution)
        encoder_layers.append(nn.Softmax(dim=1))  
        # Full model container
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        """Process input through encoder layers
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor: Probability distribution over classes, shape (batch_size, final_dim)
        """
        return self.encoder(x)
    
class Decoder(nn.Module):
    """Neural decoder module with mirrored architecture of encoder
    
    Structure: Linear => BatchNorm => ReLU repeated for each layer
    Note: All layers including final output layer get BN + ReLU
    """
    
    def __init__(self, input_dim, embedding_dim):
        """
        Args:
            input_dim (int): Dimension of original input data
            embedding_dim (list): Hidden layer dimensions (reversed from encoder)
        """
        super(Decoder, self).__init__()

        # Construct decoder dimensions by reversing encoder structure
        # Example: embedding_dim [512, 256] becomes [256, 512, input_dim]
        decoder_dim = [i for i in reversed(embedding_dim)]  # Mirror encoder dimensions
        decoder_dim.append(input_dim)  # Add final reconstruction dimension
        self._dim = len(decoder_dim) - 1  # Number of weight layers

        # Build sequential decoder layers
        decoder_layers = []
        for i in range(self._dim):
            # Linear transformation layer
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            # Batch normalization for numerical stability
            decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            # Non-linear activation (present in all layers including final)
            decoder_layers.append(nn.ReLU())
        
        # Full decoder model
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Reconstruct input from latent space
        
        Args:
            x (Tensor): Latent features, shape (batch_size, latent_dim)
            
        Returns:
            Tensor: Reconstructed data, shape (batch_size, input_dim)
        """
        return self.decoder(x)

class DCMVC(nn.Module):
    """Deep Clustering Multi-View Architecture with Shared Clustering Head
    
    Key Features:
    - View-specific encoders/decoders
    - Shared cluster prediction module
    - L2-normalized latent representations
    - Multi-view feature fusion via shared clustering
    """
    
    def __init__(self, view, input_dim, embedding_dim, cluster_dim, n_clusters, device):
        """
        Args:
            view (int): Number of data views/modalities
            input_dim (list): Input dimensions for each view
            embedding_dim (list): Encoder hidden layer dimensions
            cluster_dim (int): Hidden dimension for clustering module
            n_clusters (int): Number of target clusters
            device: Computation device (CPU/GPU)
        """
        super(DCMVC, self).__init__()
        
        # View-specific components
        self.encoders = []  # List of view-specific encoders
        self.decoders = []  # List of view-specific decoders
        self.view = view    # Number of views
        self.cluster_dim = cluster_dim  # Cluster module hidden size
        self.n_clusters = n_clusters    # Target cluster count
        
        # Initialize encoder/decoder pairs for each view
        for v in range(self.view):
            # Encoder: input_dim -> embedding_dim
            self.encoders.append(Encoder(input_dim[v], embedding_dim).to(device))
            # Decoder: embedding_dim -> input_dim
            self.decoders.append(Decoder(input_dim[v], embedding_dim).to(device))
            
        # Register as ModuleList for proper parameter tracking
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        
        # Shared clustering module
        self.cluster = nn.Sequential(
            nn.Linear(embedding_dim[-1], self.cluster_dim),  # Feature projection
            nn.BatchNorm1d(self.cluster_dim),                # Normalization
            nn.ReLU(),                                        # Non-linearity
            nn.Linear(self.cluster_dim, n_clusters),          # Cluster logits
            nn.Softmax(dim=1)                                 # Probability output
        )
    
    def forward(self, xs):
        """
        Process multi-view inputs through the architecture
        
        Args:
            xs (list): List of view-specific tensors
            
        Returns:
            xrs (list): Reconstructed inputs per view
            zs (list): L2-normalized latent features per view
            cs (list): Cluster probability distributions per view
        """
        xrs = []  # Reconstructed views
        zs = []   # Normalized latent features
        cs = []   # Cluster assignments
        
        # Process each view independently
        for v in range(self.view):
            x = xs[v]  # Input for current view
            
            # Encoder: Input -> Latent space
            z = self.encoders[v](x)
            
            # Decoder: Latent space -> Reconstructed input
            xr = self.decoders[v](z)
            
            # Cluster prediction: Normalized latent -> Cluster probabilities
            c = self.cluster(F.normalize(z))  # L2-normalization before clustering
            
            # Store outputs
            zs.append(z)
            xrs.append(xr)
            cs.append(c)
            
        return xrs, zs, cs