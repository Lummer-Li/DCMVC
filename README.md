# DCMVC: Dual Contrastive Multi-view Clustering
This repo contains the demo code and data of our paper DCMVC: Dual Contrastive Multi-view Clustering (Neurocomputing).

Please contact us (pengyuanli@bjtu.edu.cn) if you require other datasets.
> URL: [DCMVC: Dual Contrastive Multi-view Clustering](xx.com)
<img src="https://github.com/Lummer-Li/DCMVC/blob/main/DCMVC.png">

## Requirements
```python
torch >= 2.0.1
numpy >= 1.24.4
scikit-learn >= 1.3.0
```

## Reference
Please consider citing if our work is beneficial for your research.
```
@article{LI2025129889,
title = {DCMVC: Dual contrastive multi-view clustering},
journal = {Neurocomputing},
pages = {129889},
year = {2025},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2025.129889},
url = {https://www.sciencedirect.com/science/article/pii/S0925231225005612},
author = {Pengyuan Li and Dongxia Chang and Zisen Kong and Yiming Wang and Yao Zhao},
keywords = {Multi-view clustering, Deep clustering, Contrastive learning},
abstract = {Multi-view clustering, which aims to divide data into different categories that are unsupervised in respect to information from different views, plays an important role in the field of computer vision. Contrastive learning is widely used in deep multi-view clustering methods to learn more discriminative representations. However, most existing multi-view clustering methods based on contrastive learning use only a single positive sample and do not fully utilize the category information in the learning process. To address the above issues, we propose a novel dual contrastive multi-view clustering (DCMVC) method, which uses pseudo-labels to refine the embedded features to make them more suitable for clustering tasks. Specifically, an inter-view correlation contrastive module is designed to learn more compact clustering assignments through a shared clustering prediction layer. Then, on the basis of the clustering predictions, we propose an intra-view consistency contrastive module, which dynamically selects the samples with the same pseudo label as positive samples and sets the other samples as negative samples to construct contrastive learning. The proposed model can alleviate the constraints of a single positive sample on contrastive learning by fully considering the latent category information to regularize the representation structure. Extensive experiments conducted on nine real datasets demonstrate the superiority of the proposed method.}
}
```
