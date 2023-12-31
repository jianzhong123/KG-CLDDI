import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import ones_



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):#100,100,0.4,0.2
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.num_heads = 1

        self.layers = nn.ModuleList([
            GraphAttentionLayer(nfeat,
                                nhid,
                                dropout=dropout,
                                alpha=alpha,
                                concat=True) for _ in range(self.num_heads)
        ])

        #self.out1 = nn.Linear(nhid * self.num_heads, nhid)
        self.out = nn.Linear(100, 100)
    def forward(self, item_embs, entity_embs, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.out(x, y, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def forward_relation(self, item_embs, entity_embs, w_r, adj):#item_embs, entity_embs, relation_embs,padding_mask
        x = F.dropout(entity_embs, self.dropout, training=self.training)

        x = torch.cat([att.forward_relation(item_embs, x, w_r, adj) for att in self.layers], dim=1)
        x = self.out(x)
        x = x + item_embs
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def forward_relation1(self, item_embs, entity_embs, w_r, adj):  # item_embs, entity_embs, relation_embs,padding_mask
        x = F.dropout(entity_embs, self.dropout, training=self.training)

        x = torch.cat([att.forward_relation(item_embs, x, w_r, adj) for att in self.layers], dim=1)
        x = self.out1(x)
        x = x + item_embs
        x = F.dropout(x, self.dropout, training=self.training)
        return x

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.Tensor(size=(in_features, out_features)))#torch.empty,55, 2*out_features
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
        #nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.Tensor(size=(1,2307)))#(1,3 *out_features)
        nn.init.xavier_uniform_(self.a, gain=nn.init.calculate_gain('relu'))
        #self.fc = nn.Linear(in_features + out_features, out_features)#200,100
        self.fc = nn.Linear(2*out_features, out_features)  # 200,100
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward_relation(self, item_embs, entity_embs, relations, adj):
        # item_embs: N, dim
        # entity_embs: N, e_num, dim
        # relations: N, e_num, r_dim
        # adj: N, e_num
        # item_embs:[48957, 64];entity_embs:[48957, 10, 64];relation_embs:[48957, 10, 64];padding_mask:[48957, 10]
        # N, e_num, dim

        #Wh = item_embs.unsqueeze(1).expand(entity_embs.shape[0], entity_embs.shape[1], -1)  # [48957, 10, 64]

        # N, e_num, dim
        We = entity_embs
        #a_input = torch.cat((Wh, We), dim=-1)  # (N, e_num, 2*dim)
        a_input = We
        # N,e,2dim -> N,e,dim
        e_input = torch.multiply(a_input, relations).sum(-1)  # N,e#self.fc(a_input)
        We = torch.multiply(a_input, relations)
        e = self.leakyrelu(e_input)  # (N, e_num)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        #attention = F.dropout(attention, self.dropout,training=self.training)  # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        #entity_emb_weighted = torch.bmm(attention.unsqueeze(1),entity_embs).squeeze()
        attention = attention.unsqueeze(dim=2)
        #print("attention",attention)
        entity_emb_weighted = attention*We#.squeeze()
        #print("entity_emb_weighted",entity_emb_weighted)
        entity_emb_weighted=torch.sum(entity_emb_weighted,dim=1)
        #print("entity_emb_weighted",entity_emb_weighted.shape)
        h_prime = entity_emb_weighted
        return h_prime

    def forward_relation11(self, item_embs, entity_embs, relations, adj):
        # item_embs: N, dim
        # entity_embs: N, e_num, dim
        # relations: N, e_num, r_dim
        # adj: N, e_num
        # item_embs:[48957, 64];entity_embs:[48957, 10, 64];relation_embs:[48957, 10, 64];padding_mask:[48957, 10]
        # N, e_num, dim

        Wh = item_embs.unsqueeze(1).expand(entity_embs.shape[0], entity_embs.shape[1], -1)  # [48957, 10, 64]

        # N, e_num, dim
        We = entity_embs
        #a_input = torch.cat((Wh, We), dim=-1)  # (N, e_num, 2*dim)
        a_input = We
        # N,e,2dim -> N,e,dim
        e_input = torch.multiply(a_input, relations).sum(-1)  # N,e#self.fc(a_input)
        e = self.leakyrelu(e_input)  # (N, e_num)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout,
                              training=self.training)  # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1),entity_embs).squeeze()

        h_prime = entity_emb_weighted
        return h_prime
        # if self.concat:
        #     return F.elu(h_prime)
        # else:
        #     return h_prime
    #GATv2
    def forward_relation1(self, item_embs, entity_embs, relations, adj):
        # item_embs: N, dim
        # entity_embs: N, e_num, dim
        # relations: N, e_num, r_dim
        # adj: N, e_num
        # item_embs:[48957, 64];entity_embs:[48957, 10, 64];relation_embs:[48957, 10, 64];padding_mask:[48957, 10]
        # N, e_num, dim

        #item_embs = torch.mm(item_embs, self.W)
        #item_embs在第一维进行扩展
        Wh = item_embs.unsqueeze(1).expand(entity_embs.shape[0],entity_embs.shape[1], -1)#[48957, 10, 64]

        # N, e_num, dim
        #We = entity_embs


        We = torch.multiply(entity_embs, relations)
        a_input = torch.cat((Wh, We), dim=-1)  # (N, e_num, 2*dim)

        #a_input = self.fc(a_input)#2*dim->dim
        #a_input = self.fc(We)  # 2*dim->dim
        # N,e,2dim -> N,e,dim
        #e_input = torch.multiply(a_input, self.W).sum(-1)  # N,e
        #e = self.leakyrelu(e_input)  # (N, e_num)
        #e_input = torch.multiply(a_input, self.a).sum(-1)  # N,e
        #a_input =torch.multiply(a_input)
        e = self.leakyrelu(a_input).sum(-1)  # (N, e_num)
        #e=self.fc(e)
        #print("e",e.shape,self.a.shape)
        e=torch.matmul(self.a,e) # N,e,,

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout,training=self.training)  # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()

        h_prime = entity_emb_weighted

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def forward(self, item_embs, entity_embs,relations, adj):
        Wh = torch.mm(item_embs,self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        We = torch.matmul(entity_embs, self.W)  # entity_embs: (N, e_num, in_features), We.shape: (N, e_num, out_features)
        a_input = self._prepare_cat(Wh, We)  # (N, e_num, 2*out_features)
        #e = self.leakyrelu(torch.matmul(a_input,self.a).squeeze(2))  # (N, e_num)
        e = self.leakyrelu(torch.matmul(a_input, relations).squeeze(2))  # (N, e_num)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout,
                              training=self.training)  # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1),
                                        entity_embs).squeeze()
        h_prime = entity_emb_weighted + item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_cat(self, Wh, We):
        Wh = Wh.unsqueeze(1).expand(We.size())  # (N, e_num, out_features)
        return torch.cat((Wh, We), dim=-1)  # (N, e_num, 2*out_features)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(
            self.in_features) + ' -> ' + str(self.out_features) + ')'
