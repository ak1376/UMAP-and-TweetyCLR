from torch import einsum, logsumexp, no_grad

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