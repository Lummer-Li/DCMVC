<h2 align="center">✨DCMVC: Dual Contrastive Multi-view Clustering</h2>


<p align="center">
  <b>Pengyuan Li<sup>1</sup>, Dongxia Chang<sup>1</sup>, Zisen Kong<sup>1</sup>, Yiming Wang<sup>2</sup>, Yao Zhao<sup>1</sup></b>
</p>

<p align="center">
  <sup>1</sup>Institute of Information Science, Beijing Jiaotong University, Beijing, China<br>
  <sup>2</sup>School of Computer Science, Nanjing University of Posts and Telecommunications, Nanjing, China<br>
</p>

<p align="center">
  <!-- Neurocomputing Badge -->
  <a href="https://www.sciencedirect.com/science/article/abs/pii/S0925231225005612" target="_blank">
    <img src="https://img.shields.io/badge/Neurocomputing-2025-blueviolet.svg?style=flat-square" alt="Neurocomputing">
  </a>
  <!-- arXiv Badge -->
  <!-- <a href="https://arxiv.org/abs/2412.08345" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2412.08345-b31b1b.svg?style=flat-square" alt="arXiv Paper">
  </a> -->
  <!-- Contact Badge -->
  <a href="pengyuanli@bjtu.edu.cn" target="_blank">
    <img src="https://img.shields.io/badge/Email-pengyuanli%40bjtu.edu.cn-blue.svg" alt="Contact Author">
  </a>
</p>

<p align="center">
  🔥 Our work has been accepted by Neurocomputing 2025!<br>
</p>

## Overview🔍
<div>
    <img src="https://github.com/Lummer-Li/DCMVC/blob/main/DCMVC.png" width="90%" height="90%">
</div>

**Figure 1. The framework of the proposed DCMVC.**


**_Abstract -_** Multi-view clustering, which aims to divide data into different categories that are unsupervised with respect to information from different views, plays an important role in the field of computer vision. Contrastive learning is widely used in deep multi-view clustering methods to learn more discriminative representations. However, most existing multi-view clustering methods based on contrastive learning use only a single positive sample and do not fully utilize the category information in the learning process. To address the above issues, we propose a novel dual contrastive multi-view clustering (DCMVC) method, which uses pseudo-labels to refine the embedded features to make them more suitable for clustering tasks. Specifically, an inter-view correlation contrastive module is designed to learn more compact clustering assignments through a shared clustering prediction layer. Then, on the basis of the clustering predictions, we propose an intra-view consistency contrastive module, which dynamically selects the samples with the same pseudo label as positive samples and sets the other samples as negative samples to construct contrastive learning. The proposed model can alleviate the constraints of a single positive sample on contrastive learning by fully considering the latent category information to regularize the representation structure. Extensive experiments conducted on nine real datasets demonstrate the superiority of the proposed method.

## Datasets📚
To comprehensively evaluate the performance of the proposed DCMVC method, we conducted experiments using nine publicly available multi-view datasets, as shown below. 


| Dataset      | Samples  | Views  | View Dimensions         | Clusters  |
|--------------|----------|--------|-------------------------|-----------|
| MSRC_v1      | 210      | 6      | 1302/48/512/100/256/210 | 7         |
| Synthetic3d  | 600      | 3      | 3/3/3                   | 3         |
| Handwritten  | 2000     | 6      | 216/76/64/6/240/47      | 10        |
| UCI-Digit    | 2000     | 3      | 216/76/64               | 10        |
| NoisyMNIST   | 20000    | 2      | 784/784                 | 10        |
| Fashion      | 10000    | 3      | 784/784/784             | 10        |
| BDGP         | 2500     | 2      | 1750/79                 | 5         |
| Mfeat        | 2000     | 6      | 216/76/64/6/240/47      | 10        |
| NUSWIDEOBJ   | 30000    | 5      | 65/226/145/74/129       | 31        |



## Experimental Results🏆


**Table 1. Clustering average results and standard deviations for nine multi-view datasets. The optimal and suboptimal results are highlighted in red and blue. The notation O/M signifies an out-of-memory error encountered during the training process.**
<div>
    <img src="https://github.com/Lummer-Li/DCMVC/blob/main/assets/tab1.png" width="80%" height="96%">
</div>

<br> </br>

**Table 2. Ablation study on MSRC_v1, UCI-Digit, and NoisyMNIST, respectively.**
<div>
    <img src="https://github.com/Lummer-Li/DCMVC/blob/main/assets/tab2.png" width="30%" height="96%">
</div>



## Getting Started🚀
### Data Preparation
The dataset should be organised as follows, taking MSRC_v1 as an example:
```text
MSRC_v1
├── X
│   ├── X1
│   ├── X2
│   ├── X3
│   ├── ...
├── Y
```

### Training and Evaluation
- To train the DCMVC, run: `main.py`. The prediction results obtained using the K-Means algorithm.



## Cite our work📝
```bibtex
@article{li2025dcmvc,
  title={DCMVC: Dual contrastive multi-view clustering},
  author={Li, Pengyuan and Chang, Dongxia and Kong, Zisen and Wang, Yiming and Zhao, Yao},
  journal={Neurocomputing},
  volume={635},
  pages={129889},
  year={2025},
  publisher={Elsevier}
}
```

## License📜
The source code is free for research and educational use only. Any commercial use should get formal permission first.



