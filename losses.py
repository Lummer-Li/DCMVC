import sys
import math
import torch
import torch.nn.functional as F


def cluster_contrastive_loss(c_i, c_j, n_clusters, temperature=0.5):
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    similarity_f = torch.nn.CosineSimilarity(dim=2)

    # entropy
    p_i = c_i.sum(0).view(-1)
    p_i /= p_i.sum()
    ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
    p_j = c_j.sum(0).view(-1)   
    p_j /= p_j.sum()
    ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
    ne_loss = ne_i + ne_j

    N = 2 * n_clusters                  
    mask = torch.ones((N, N))           
    mask = mask.fill_diagonal_(0)       
    for i in range(n_clusters):
        mask[i, n_clusters + i] = 0
        mask[n_clusters + i, i] = 0
    mask = mask.bool()           

    c_i, c_j = c_i.t(), c_j.t()                       
    c = torch.cat((c_i, c_j), dim=0)    

    # print((c.unsqueeze(1).shape, c.unsqueeze(0).shape))
    sim = similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / temperature    
    sim_i_j = torch.diag(sim, n_clusters)                               
    sim_j_i = torch.diag(sim, -n_clusters)                              

    positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) 
    negative_clusters = sim[mask].reshape(N, -1)                           

    labels = torch.zeros(N).to(positive_clusters.device).long()            
    logits = torch.cat((positive_clusters, negative_clusters), dim=1)      
    loss = criterion(logits, labels) / N

    return loss + ne_loss


def calcul_var(data, labels):
    labels = torch.squeeze(labels)
    # 获取不同聚类簇的索引
    clusters = torch.unique(labels)

    var_sum = 0.
    # 计算每个聚类簇的中心和方差
    for cluster in clusters:
        cluster_data = data[labels == cluster]
        cluster_center = torch.mean(cluster_data, dim=0)
        distances = torch.norm(cluster_data - cluster_center, dim=1)
        variance = torch.var(distances)
        var_sum += variance
        # print(f'Cluster {cluster.item() + 1} Center: {cluster_center}, Variance: {variance.item()}')

    return var_sum

# temperature=0.5, base_temperature=0.5
def self_cluster_contrastive_loss(args, features, labels=None, mask=None, temperature=0.5, base_temperature=0.5, margin=0.1):
    # features = target_distribution(features)

    device = (torch.device(args.device) if features.is_cuda else torch.device('cpu'))
    batch_size = features.shape[0]  # 获取批次大小，即数据点的数量

    # 计算对比损失
    anchor_dot_contrast = torch.div(torch.matmul(features, features.T),
                                    temperature)  # 计算特征之间的点积，并除以温度

    # 为了数值稳定性
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 计算点积的最大值
    logits = anchor_dot_contrast - logits_max.detach()  # 对点积减去最大值

    # logits_mean = torch.mean(anchor_dot_contrast, dim=1, keepdim=True)  # 计算点积的最大值
    # logits_std = torch.std(anchor_dot_contrast, dim=1, keepdim=True)  # 计算点积的最大值
    # logits = torch.div((anchor_dot_contrast - logits_mean.detach()), logits_std.detach())  # 对点积减去最大值

    # 创建标签之间的匹配掩码
    labels = labels.view(-1, 1)  # 将标签变形为列向量
    mask = torch.ne(labels, labels.T).float().to(device)  # 创建标签之间的匹配掩码

    # 区分正样本和负样本
    mask.masked_fill_(mask == 0, -1)

    # 掩盖自我对比情况
    # mask = mask - torch.eye(batch_size, device=device)
    mask = mask + torch.eye(batch_size, device=device)

    # 计算交叉熵
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # 计算对数概率
    # log_prob = torch.log(exp_logits / exp_logits.sum(1, keepdim=True) + margin)
    mean_log_prob_pos = (mask * log_prob).sum(1) / torch.abs(mask).sum(1)      # 计算负样本上的对数概率的平均值

    # 计算损失
    loss = - (temperature / base_temperature) * mean_log_prob_pos  # 计算对比损失
    loss = loss.mean()  # 计算平均损失

    return loss + margin
    # return loss + ne_i



def contrastive_loss(z, temperature=0.5):
    """
    To get contrastive loss

    Parameters
    ----------
    z: sample features
    temperature: float optional(default 0.5)
        super parameter

    Returns 
    -------
    loss: contrastive loss
    """
    batch_size = z.size(0) // 2

    similarity_martix = torch.matmul(z, z.T) / temperature
    mask = torch.eye(batch_size * 2, dtype=torch.bool).to(z.device)
    logits = similarity_martix.masked_select(~mask).view(batch_size * 2, -1)
    targets = torch.arange(batch_size).repeat(2).T.flatten().to(z.device)

    loss = torch.nn.CrossEntropyLoss()(logits, targets.long())

    return loss



def similarity_measure(x1, x2):
    # 计算相似度度量，可以使用欧氏距离、余弦相似度等方法
    # 这里使用余弦相似度作为示例
    cos_sim = F.cosine_similarity(x1, x2, dim=-1)
    return cos_sim

def loss_function(x1, x2, y1, y2, margin=0.1):
    # x1, x2: 样本向量表示
    # y1, y2: 伪标签，指示样本的相似性

    p_i = x1.sum(0).view(-1)
    p_i /= p_i.sum()
    ne_i = (p_i * torch.log(p_i)).sum()
    p_j = x2.sum(0).view(-1)
    p_j /= p_j.sum()
    ne_j = (p_j * torch.log(p_j)).sum()
    entropy = ne_i + ne_j

    # 计算相似度度量
    sim = similarity_measure(x1, x2)

    # 相同标签样本的损失函数
    loss_same = F.mse_loss(sim[y1 == y2], torch.ones_like(sim[y1 == y2]))

    # 不同标签样本的损失函数
    loss_diff = F.relu(margin - sim[y1 != y2]).mean()

    # 总体损失函数
    loss = loss_same - loss_diff

    # return loss + entropy
    return loss

