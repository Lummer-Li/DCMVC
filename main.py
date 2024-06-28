import torch
import argparse
import utils
import configs
import metrics
import numpy as np
from models import DCMVC
import torch.optim as optim
from datasets import load_data
import torch.nn.functional as F
from sklearn.utils import shuffle
from losses import cluster_contrastive_loss, self_cluster_contrastive_loss

parser = argparse.ArgumentParser(description='DCCL Super Parameters')

parser.add_argument('--batch-size', type=int, default=512, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--dataset', type=str, default='synthetic3d', metavar='N', help='input dataset name to choose dataset (default: Caltech101-20)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=3e-4, metavar='LR', help='learning rate (default: 1e-4)')
parser.add_argument('--momentum', type=float, default=0, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
parser.add_argument('--weight_decay', type=float, default=0, metavar='M',help='weight decay (default: 0)')
parser.add_argument('--temperature', type=float, default=0.5, metavar='M',help='temperature (default: 0.5)')
parser.add_argument('--device', type=str, default='cuda:1', metavar='M',help='device (default: 0)')

args = parser.parse_args()

if __name__ == '__main__':
        temperature = args.temperature
        args = configs.get_config(args)
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        utils.set_seed(args.seed)

        print('=========================================')
        print(args)
        print('=========================================')
        
        X, Y = load_data(args.dataset)
        view = len(X)
        n_clusters = np.unique(Y).size
        print('The clusters of datasets:', n_clusters)
        for i in range(len(X)):
            print(X[i].shape)
            X[i] = torch.from_numpy(X[i]).float().to(device)

        
        loss_rc_list, loss_cc_list, loss_cl_list, loss_loss_list = [], [], [], []
        acc_list, nmi_list, ari_list = [], [], []
        
        model = DCMVC(view, args.input_dim, args.embedding_dims, 
                    args.cluster_dims, n_clusters, device).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(args.epochs):
            loss_rc, loss_cc, loss_cl, loss_loss = 0, 0, 0, 0
            loss_list = []
            optimizer.zero_grad()
            for i in range(0, view-1):
                for j in range(i+1, view):
                    X[i], X[j] = X[i].to(device), X[j].to(device)
                    X1, X2 = shuffle(X[i], X[j])
                    
                    for batch_x_i, batch_x_j, batch_i in utils.next_batch(X1, X2, args.batch_size):
                        z_i = model.encoders[i](batch_x_i)
                        z_j = model.encoders[j](batch_x_j)
                        d_i = model.decoders[i](z_i)
                        d_j = model.decoders[j](z_j)

                        rc_loss_i = F.mse_loss(d_i, batch_x_i) 
                        rc_loss_j = F.mse_loss(d_j, batch_x_j) 
                        rc_loss = rc_loss_i + rc_loss_j

                        cl_i = model.cluster(F.normalize(z_i))
                        cl_j = model.cluster(F.normalize(z_j))
                        cl_loss = cluster_contrastive_loss(cl_i, cl_j, n_clusters, temperature)

                        cl = torch.argmax(torch.mean(torch.stack([cl_i, cl_j]), dim=0), dim=1)
                        s_i = self_cluster_contrastive_loss(args, z_i, cl)
                        s_j = self_cluster_contrastive_loss(args, z_j, cl)
                        intra_view = s_i + s_j

                        loss_rc += rc_loss
                        loss_cc += intra_view
                        loss_cl += cl_loss
                        loss_loss += rc_loss + cl_loss + intra_view
                        loss_list.extend([intra_view * args.alpha, rc_loss * args.beta, cl_loss * args.gamma])

            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print("epoch: %.0f, rc loss: %.6f, cross view loss: %.6f, cluster loss: %.6f, loss: %.6f"
                    % (epoch+1, loss_rc, loss_cc, loss_cl, loss))
                loss_rc_list.append(loss_rc.item())
                loss_cc_list.append(loss_cc.item())
                loss_cl_list.append(loss_cl.item())
                loss_loss_list.append(loss_loss.item())
                score = metrics.evaluation(model, X, Y, device)
                print(score)
                print('------------------------------------------------------------------------------------')

        save_dict= {'rc_loss': loss_rc_list, 'cc_loss': loss_cc_list, 'cl_loss': loss_cl_list, 'loss_loss': loss_loss_list,
                    'ACC': acc_list, 'NMI': nmi_list, 'ARI': ari_list}
        with open('./loss.txt', "w") as file:
            file.write(str(save_dict))
