import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graph.gat.layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    """Dense version of GAT."""
    def __init__(self, nin, nhid, nout, alpha, nheads):
        super(GAT, self).__init__()

        self.attentions = [GraphAttentionLayer(nin, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nout, alpha=alpha, concat=False)
        self.bn1 = nn.BatchNorm1d(nout)

    def forward(self, x, adj):
        # self.attentions = [att.cuda() for att in self.attentions]
        x = torch.cat([v(x, adj) for k, v in self._modules.items() if k.startswith("attention")], dim=1)#torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.out_att(x, adj)
        x = self.bn1(x)
        return F.tanh(x)


class SpGAT(nn.Module):
    """Sparse version of GAT."""
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)