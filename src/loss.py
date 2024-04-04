from torch import einsum, logsumexp, no_grad
import torch.nn.functional as F
import torch.nn as nn

def InfoNCE(ref, pos, neg, tau = 1.0):
    pos_dist = einsum("nd, nd->n", ref, pos)/tau
    neg_dist = einsum("nd, md->nm", ref, neg)/tau

    with no_grad():
        c, _ = neg_dist.max(dim = 1)
    pos_dist = pos_dist - c.detach()
    neg_dist = neg_dist - c.detach()
    pos_loss = -pos_dist.mean()
    neg_loss = logsumexp(neg_dist, dim = 1).mean()

    return pos_loss + neg_loss 

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # Euclidean distance
        distance_negative = (anchor - negative).pow(2).sum(1)  # Euclidean distance
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()