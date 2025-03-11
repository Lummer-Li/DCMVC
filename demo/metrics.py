import sys
import torch
import numpy as np
from sklearn import metrics
from munkres import Munkres
from sklearn.cluster import KMeans

def evaluate(model, X, y):
    """Evaluate model performance using clustering metrics
    
    Args:
        model: Trained model instance
        X: Input data tensor
        y: Ground truth labels
        
    Returns:
        Tuple: (NMI, ARI, FMI, ACC) metrics
    """
    model.eval()
    with torch.no_grad():
        # Forward pass through model
        xrs, zs, cs = model(X)
        
        # Combine cluster probabilities across views
        z = torch.stack(cs, dim=0)
        rs = torch.mean(z, dim=0)
        pred = torch.argmax(rs, dim=1).cpu().numpy()

        # Calculate metrics
        nmi = metrics.normalized_mutual_info_score(y, pred)
        ari = metrics.adjusted_rand_score(y, pred)
        f = metrics.fowlkes_mallows_score(y, pred)
        pred_adjusted = get_y_preds(y, pred, len(set(y)))
        acc = metrics.accuracy_score(pred_adjusted, y)

    return nmi, ari, f, acc

def evaluation(model, X, y, device):
    """Wrapper for clustering evaluation"""
    model.eval()
    with torch.no_grad():
        # Get latent representations
        xrs, zs, cs = model(X)
        # Concatenate multi-view features
        latent_fusion = torch.cat(zs, dim=1).cpu().numpy()

        # Perform clustering evaluation
        scores = clustering([latent_fusion], y)

    return scores

def clustering(x_list, y):
    """Perform clustering and calculate metrics
    
    Args:
        x_list: List of feature matrices
        y: True labels
        
    Returns:
        dict: Clustering quality metrics
    """
    global fig_name  # Preserved for compatibility (not used in current scope)
    n_clusters = np.size(np.unique(y))
    
    # Concatenate features from all views
    x_final_concat = np.concatenate(x_list[:], axis=1)
   
    # Get KMeans cluster assignments
    kmeans_assignments, km = get_cluster_sols(x_final_concat, 
                                            ClusterClass=KMeans, 
                                            n_clusters=n_clusters,
                                            init_args={'n_init': 10})
                                            
    # Adjust label indices if needed
    if np.min(y) == 1:
        y = y - 1
        
    # Calculate clustering metrics
    scores, _ = clustering_metric(y, kmeans_assignments, n_clusters)

    return {'kmeans': scores}

def calculate_cost_matrix(C, n_clusters):
    """Create cost matrix for Hungarian algorithm
    
    Args:
        C: Confusion matrix
        n_clusters: Number of clusters
        
    Returns:
        ndarray: Cost matrix for cluster alignment
    """
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # Total samples in predicted cluster j
        for i in range(n_clusters):
            t = C[i, j]      # Samples of true class i in predicted cluster j
            cost_matrix[j, i] = s - t  # Cost = total in cluster - correct matches
    return cost_matrix

def get_cluster_labels_from_indices(indices):
    """Map Hungarian algorithm result to cluster labels
    
    Args:
        indices: Output from Munkres algorithm
        
    Returns:
        ndarray: Remapped cluster labels
    """
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]  # Get column index from assignment
    return cluster_labels

def get_y_preds(y_true, cluster_assignments, n_clusters):
    """Align cluster assignments with true labels using Hungarian algorithm
    
    Args:
        y_true: True labels
        cluster_assignments: Predicted cluster indices
        n_clusters: Number of clusters
        
    Returns:
        ndarray: Adjusted predictions aligned with true labels
    """
    # Create confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments)
    
    # Calculate cost matrix for optimal assignment
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    
    # Solve assignment problem
    indices = Munkres().compute(cost_matrix)
    
    # Get label mapping
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    
    # Adjust cluster indices if needed
    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
        
    # Map predictions using optimal assignment
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred

def classification_metric(y_true, y_pred, average='macro', verbose=True, decimals=4):
    """Calculate classification metrics
    
    Returns:
        tuple: (metrics_dict, confusion_matrix)
    """
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    
    # Core metrics calculation
    accuracy = np.round(metrics.accuracy_score(y_true, y_pred), decimals)
    precision = np.round(metrics.precision_score(y_true, y_pred, average=average), decimals)
    recall = np.round(metrics.recall_score(y_true, y_pred, average=average), decimals)
    f_score = np.round(metrics.f1_score(y_true, y_pred, average=average), decimals)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f_measure': f_score
    }, confusion_matrix

def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
    """Get cluster solutions with robustness handling
    
    Args:
        x: Input data
        cluster_obj: Pre-fit cluster object
        ClusterClass: Cluster algorithm class
        n_clusters: Number of clusters
        init_args: Initialization arguments
        
    Returns:
        tuple: (assignments, cluster_object)
    """
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None))
    
    cluster_assignments = None
    if cluster_obj is None:
        cluster_obj = ClusterClass(n_clusters, **init_args)
        # Attempt clustering with retry mechanism
        for _ in range(10):
            try:
                cluster_obj.fit(x)
                break
            except:
                print("Unexpected error:", sys.exc_info())
        else:  # Executed if loop completes normally (no break)
            return np.zeros((len(x),)), cluster_obj

    cluster_assignments = cluster_obj.predict(x)
    return cluster_assignments, cluster_obj

def clustering_metric(y_true, y_pred, n_clusters, verbose=True, decimals=4):
    """Calculate comprehensive clustering metrics
    
    Returns:
        tuple: (metrics_dict, confusion_matrix)
    """
    # Align predicted and true labels
    y_pred_adjusted = get_y_preds(y_true, y_pred, n_clusters)
    
    # Get classification metrics
    classification_metrics, confusion_matrix = classification_metric(y_true, y_pred_adjusted)
    
    # Clustering-specific metrics
    ami = np.round(metrics.adjusted_mutual_info_score(y_true, y_pred), decimals)
    nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), decimals)
    ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), decimals)
    pur = np.round(calculate_purity(y_true, y_pred), decimals)

    # Combine all metrics
    return dict({
        'AMI': ami,
        'NMI': nmi,
        'ARI': ari,
        'Purity': pur
    }, **classification_metrics), confusion_matrix

def calculate_purity(y_true, y_pred):
    """Calculate clustering purity through majority voting
    
    Args:
        y_true: True labels
        y_pred: Cluster assignments
        
    Returns:
        float: Purity score between 0-1
    """
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    
    # Remap labels to 0-indexed
    ordered_labels = np.arange(len(labels))
    for k in range(len(labels)):
        y_true[y_true == labels[k]] = ordered_labels[k]
        
    # Create histogram bins
    bins = np.concatenate((ordered_labels, [np.max(ordered_labels)+1]))
    
    # Assign majority class for each cluster
    for cluster_index in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster_index], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster_index] = winner

    return metrics.accuracy_score(y_true, y_voted_labels)
