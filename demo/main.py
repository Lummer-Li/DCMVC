import json
import torch
import utils
import configs
import metrics
import argparse
import numpy as np
from models import DCMVC
import torch.optim as optim
from datasets import load_data
import torch.nn.functional as F
from sklearn.utils import shuffle
from losses import cluster_contrastive_loss, self_cluster_contrastive_loss

# ----------------------- Argument Parser Setup -----------------------
parser = argparse.ArgumentParser(description='DCMVC Super Parameters')

# Define command-line arguments with default values
parser.add_argument('--batch-size', type=int, default=512, metavar='N', 
                   help='Training batch size (default: 512)')
parser.add_argument('--dataset', type=str, default='synthetic3d', metavar='N',
                   help='Dataset name to load (default: synthetic3d)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                   help='Total training epochs (default: 500)')
parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                   help='Learning rate (default: 0.0003)')
parser.add_argument('--momentum', type=float, default=0, metavar='M',
                   help='SGD momentum (default: 0)')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                   help='Random seed (default: 0)')
parser.add_argument('--weight_decay', type=float, default=0, metavar='M',
                   help='Weight decay (L2 penalty) (default: 0)')
parser.add_argument('--temperature', type=float, default=0.5, metavar='M',
                   help='Contrastive loss temperature (default: 0.5)')
parser.add_argument('--device', type=str, default='cuda:1', metavar='M',
                   help='Computation device (default: cuda:1)')

args = parser.parse_args()

if __name__ == '__main__':
    # ----------------------- Initialization -----------------------
    # Set temperature from arguments
    temperature = args.temperature
    
    # Load dataset-specific configuration
    args = configs.get_config(args)
    
    # Device configuration
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Set random seeds for reproducibility
    utils.set_seed(args.seed)

    # ----------------------- Data Loading & Preparation -----------------------
    print('='*40)
    print(args)  # Display final configuration
    print('='*40)
    
    # Load dataset features and labels
    X, Y = load_data(args.dataset)
    
    # Get view count and cluster numbers
    view = len(X)
    n_clusters = np.unique(Y).size
    print(f'Number of clusters: {n_clusters}')
    
    # Convert numpy arrays to PyTorch tensors
    for i in range(len(X)):
        print(f'View {i} shape: {X[i].shape}')
        X[i] = torch.from_numpy(X[i]).float().to(device)

    # ----------------------- Model & Optimizer Setup -----------------------
    # Initialize multi-view clustering model
    model = DCMVC(view, args.input_dim, args.embedding_dims, 
                 args.cluster_dims, n_clusters, device).to(device)

    # Configure Adam optimizer
    optimizer = optim.Adam(model.parameters(), 
                         lr=args.lr, 
                         weight_decay=args.weight_decay)

    # ----------------------- Training Loop -----------------------
    # Initialize loss trackers
    loss_rc_list, loss_cc_list, loss_cl_list, loss_loss_list = [], [], [], []
    acc_list, nmi_list, ari_list = [], [], []
    
    # Epoch loop
    for epoch in range(args.epochs):
        # Reset loss accumulators
        loss_rc, loss_cc, loss_cl, loss_loss = 0, 0, 0, 0
        loss_list = []
        
        # Zero gradients before forward pass
        optimizer.zero_grad()
        
        # Process all view pairs
        for i in range(0, view-1):
            for j in range(i+1, view):
                # Shuffle views for decorrelation
                X[i], X[j] = X[i].to(device), X[j].to(device)
                X1, X2 = shuffle(X[i], X[j])
                
                # Batch processing
                for batch_x_i, batch_x_j, batch_i in utils.next_batch(X1, X2, args.batch_size):
                    # ----------------------- Forward Pass -----------------------
                    # View i processing
                    z_i = model.encoders[i](batch_x_i)
                    d_i = model.decoders[i](z_i)
                    
                    # View j processing
                    z_j = model.encoders[j](batch_x_j)
                    d_j = model.decoders[j](z_j)
                    
                    # ----------------------- Loss Calculations -----------------------
                    # Reconstruction loss (MSE)
                    rc_loss_i = F.mse_loss(d_i, batch_x_i) 
                    rc_loss_j = F.mse_loss(d_j, batch_x_j) 
                    rc_loss = rc_loss_i + rc_loss_j
                    
                    # Cluster prediction (normalized features)
                    cl_i = model.cluster(F.normalize(z_i))
                    cl_j = model.cluster(F.normalize(z_j))
                    
                    # Cluster contrastive loss
                    cl_loss = cluster_contrastive_loss(cl_i, cl_j, n_clusters, temperature)
                    
                    # Self-cluster consistency loss
                    combined_cl = torch.argmax(torch.mean(torch.stack([cl_i, cl_j]), dim=0), dim=1)
                    s_i = self_cluster_contrastive_loss(args, z_i, combined_cl)
                    s_j = self_cluster_contrastive_loss(args, z_j, combined_cl)
                    intra_view = s_i + s_j
                    
                    # ----------------------- Loss Accumulation -----------------------
                    loss_rc += rc_loss
                    loss_cc += intra_view
                    loss_cl += cl_loss
                    total_loss = rc_loss + cl_loss + intra_view
                    loss_loss += total_loss
                    
                    # Collect weighted losses
                    loss_list.extend([intra_view * args.alpha, 
                                    rc_loss * args.beta, 
                                    cl_loss * args.gamma])

        # ----------------------- Backpropagation -----------------------
        # Calculate total loss
        loss = sum(loss_list)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()

        # ----------------------- Evaluation & Logging -----------------------
        if (epoch + 1) % 100 == 0:
            # Print training metrics
            print(f"Epoch: {epoch+1}, "
                 f"RC Loss: {loss_rc:.6f}, "
                 f"Cross-View Loss: {loss_cc:.6f}, "
                 f"Cluster Loss: {loss_cl:.6f}, "
                 f"Total Loss: {loss:.6f}")
            
            # Store losses
            loss_rc_list.append(loss_rc.item())
            loss_cc_list.append(loss_cc.item())
            loss_cl_list.append(loss_cl.item())
            loss_loss_list.append(loss_loss.item())
            
            # Evaluate clustering performance
            score = metrics.evaluation(model, X, Y, device)
            print(score)
            print('-'*80)

    # ----------------------- Results Saving -----------------------
    # Prepare results dictionary
    save_dict = {
        'rc_loss': loss_rc_list,
        'cc_loss': loss_cc_list,
        'cl_loss': loss_cl_list,
        'loss_loss': loss_loss_list,
        'ACC': acc_list,
        'NMI': nmi_list,
        'ARI': ari_list
    }
    
    # Write results to file
    with open('./loss.txt', "w") as file:
        file.write(str(save_dict))