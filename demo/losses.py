import math
import torch
import torch.nn.functional as F


def cluster_contrastive_loss(c_i, c_j, n_clusters, temperature=0.5):
    """Compute contrastive loss between cluster distributions with entropy regularization
    
    Args:
        c_i (Tensor): Cluster probability matrix for view i [batch_size, n_clusters]
        c_j (Tensor): Cluster probability matrix for view j [batch_size, n_clusters]
        n_clusters (int): Number of clusters
        temperature (float): Similarity scaling factor
    
    Returns:
        Tensor: Combined loss with contrastive term and entropy penalty
    
    Math:
        Loss = ContrastiveLoss + EntropyPenalty
        ContrastiveLoss = CrossEntropy( [pos_sim; neg_sims], zero_labels )
        EntropyPenalty = (H(c_i) + H(c_j)) where H(p) = log(n_clusters) + sum(p log p)
    """
    
    # 1. Initialize loss components
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    similarity_f = torch.nn.CosineSimilarity(dim=2)

    # 2. Entropy Regularization Term
    def compute_entropy(p):
        p_sum = p.sum(0).view(-1)  # Sum over batch dimension
        p_norm = p_sum / p_sum.sum()  # Normalize to probability distribution
        entropy = math.log(p_norm.size(0)) + (p_norm * torch.log(p_norm)).sum()
        return entropy
    
    ne_loss = compute_entropy(c_i) + compute_entropy(c_j)

    # 3. Contrastive Mask Preparation
    N = 2 * n_clusters
    mask = torch.ones((N, N), dtype=torch.bool, device=c_i.device)
    
    # Remove self-similarity and cross-view positives
    mask.fill_diagonal_(0)
    for i in range(n_clusters):
        mask[i, n_clusters + i] = False  # Block view1-view2 correspondences
        mask[n_clusters + i, i] = False

    # 4. Similarity Matrix Construction
    c = torch.cat([c_i.t(), c_j.t()], dim=0)  # Stack cluster centers [2n_clusters, batch_size]
    
    # Compute cosine similarity between all cluster pairs
    sim = similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / temperature  # [2n_clusters, 2n_clusters]

    # 5. Positive/Negative Sampling
    # Get cross-view positive pairs (i <-> n_clusters+i)
    pos_pairs = torch.cat([
        torch.diag(sim, diagonal=n_clusters),   # View1 to View2
        torch.diag(sim, diagonal=-n_clusters)    # View2 to View1
    ]).reshape(N, 1)  # [2n_clusters, 1]

    # Get all valid negative pairs using mask
    neg_pairs = sim[mask].reshape(N, -1)  # [2n_clusters, (2n_clusters-2)]

    # 6. Contrastive Loss Calculation
    logits = torch.cat([pos_pairs, neg_pairs], dim=1)  # [2n_clusters, 1+negatives]
    labels = torch.zeros(N, dtype=torch.long, device=logits.device)  # Positives are index 0
    
    contrastive_loss = criterion(logits, labels) / N  # Normalize by cluster count

    return contrastive_loss + ne_loss


def calcul_var(data, labels):
    """
    Calculate the sum of intra-cluster distance variances for all clusters
    
    Args:
        data (Tensor): Input data matrix of shape [num_samples, feature_dim]
        labels (Tensor): Cluster assignment labels of shape [num_samples]
        
    Returns:
        float: Total variance of Euclidean distances within all clusters
    
    Methodology:
        1. For each cluster:
           a. Compute cluster centroid (mean vector)
           b. Calculate Euclidean distances from samples to centroid
           c. Compute variance of these distances
        2. Sum variances across all clusters
    """
    # Ensure labels are 1D tensor
    labels = torch.squeeze(labels)
    
    # Get unique cluster identifiers
    clusters = torch.unique(labels)
    
    # Initialize total variance accumulator
    var_sum = 0.
    
    # Process each cluster
    for cluster in clusters:
        # Extract samples belonging to current cluster
        cluster_data = data[labels == cluster]
        # Compute cluster centroid (mean along feature dimension)
        cluster_center = torch.mean(cluster_data, dim=0)
        # Calculate Euclidean distances from samples to centroid
        distances = torch.norm(cluster_data - cluster_center, dim=1)
        # Compute variance of distances (measure of cluster compactness)
        variance = torch.var(distances)
        # Accumulate variance
        var_sum += variance

    return var_sum

def self_cluster_contrastive_loss(args, features, labels=None, mask=None, temperature=0.5, base_temperature=0.5, margin=0.1):
    """Self-supervised cluster contrastive loss with label-aware negative masking
    
    Args:
        args: Configuration parameters
        features (Tensor): Feature matrix of shape [batch_size, embedding_dim]
        labels (Tensor): Cluster labels of shape [batch_size]
        mask (Tensor): Optional precomputed mask (not used in current implementation)
        temperature (float): Softmax temperature for contrastive scaling
        base_temperature (float): Base temperature for loss scaling
        margin (float): Margin for decision boundary enhancement
        
    Returns:
        Tensor: Computed contrastive loss value
    """
    
    # Device configuration - maintain features' original device
    device = (torch.device(args.device) if features.is_cuda else torch.device('cpu'))
    
    # Get batch dimensions
    batch_size = features.shape[0]  # Number of samples in current batch

    # --------------------- Similarity Matrix Construction ---------------------
    # Compute pairwise cosine similarities between all samples
    anchor_dot_contrast = torch.div(
        torch.matmul(features, features.T),  # Raw dot products
        temperature  # Temperature scaling for contrast control
    )

    # --------------------- Numerical Stability Optimization -----------------
    # Subtract max logit for numerical stability (prevents exp() overflow)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()  # Centered logits

    # --------------------- Label-aware Mask Construction --------------------
    # Create binary mask where 1 indicates dissimilar clusters (negative pairs)
    labels = labels.view(-1, 1)  # Reshape labels to column vector
    mask = torch.ne(labels, labels.T).float().to(device)  # 1 where labels differ
    
    # Convert mask to 1 format: 
    # -1 for positive pairs (same cluster), 1 for negative pairs (different clusters)
    mask.masked_fill_(mask == 0, -1)  # Replace 0s with -1 (positive indicators)
    
    # Exclude self-comparisons from negative pairs
    mask = mask + torch.eye(batch_size, device=device)  # Add identity matrix to mask diagonal

    # --------------------- Contrastive Loss Calculation ----------------------
    # Softmax normalization
    exp_logits = torch.exp(logits)
    
    # Compute log probabilities using stable log-sum-exp
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    
    # --------------------- Masked Probability Aggregation -------------------
    # Calculate mean log probability for positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(1) / torch.abs(mask).sum(1)
    
    # Final loss composition
    loss = - (temperature / base_temperature) * mean_log_prob_pos  # Temperature scaled
    loss = loss.mean()  # Batch averaging

    # Add margin for enhanced cluster separation
    return loss + margin


