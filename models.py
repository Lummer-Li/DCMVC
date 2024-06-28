import torch
import torch.nn as nn
import torch.nn.functional as F

# Models
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(Encoder, self).__init__()

        encoder_dim = [input_dim]
        encoder_dim.extend(embedding_dim)
        self._dim = len(encoder_dim) - 1
        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Softmax(dim=1))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(Decoder, self).__init__()

        decoder_dim = [i for i in reversed(embedding_dim)]
        decoder_dim.append(input_dim)
        self._dim = len(decoder_dim) - 1

        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(x)

class DCMVC(nn.Module):
    def __init__(self, view, input_dim, embedding_dim, cluster_dim, n_clusters, device):
        super(DCMVC, self).__init__()

        self.encoders = []
        self.decoders = []
        self.view = view
        self.cluster_dim = cluster_dim
        for v in range(self.view):
            self.encoders.append(Encoder(input_dim[v], embedding_dim).to(device))
            self.decoders.append(Decoder(input_dim[v], embedding_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.n_clusters = n_clusters

        self.cluster = nn.Sequential(
            nn.Linear(embedding_dim[-1], self.cluster_dim),
            nn.BatchNorm1d(self.cluster_dim),
            nn.ReLU(),
            nn.Linear(self.cluster_dim, n_clusters),
            nn.Softmax(dim=1)
        )
    
    def forward(self, xs):
        xrs = []
        zs = []
        cs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            c = self.cluster(F.normalize(z))
            zs.append(z)
            xrs.append(xr)
            cs.append(c)
        return xrs, zs, cs