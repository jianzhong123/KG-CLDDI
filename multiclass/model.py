import math

from torch.nn import Parameter
from torch.nn.init import xavier_normal_

import world
import torch
from dataloader import BasicDataset
from torch import nn
from GAT import GAT


import torch.nn.functional as F

import numpy as np

import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem import DataStructs
# from mycode.data_preprocessing import get_mol_edge_list_and_feat_mtx
from sklearn.decomposition import PCA

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class KGCL(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset, kg_dataset):
        super(KGCL, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset
        self.kg_dataset = kg_dataset

        transE_entity_path = '/home/zhongjian/KG-CLDDI-main/data/TwoSides/DRKG_TransE_l2_entity.npy'#_300
        transE_entity_data = np.load(transE_entity_path)
        #transE_entity_data = transE_entity_data / np.linalg.norm(transE_entity_data, axis=1)[:, np.newaxis]
        zero_vec = np.zeros((1, transE_entity_data.shape[1]))
        transE_entity_data = np.append(transE_entity_data, zero_vec, axis=0)  # maxvalue:96766
        self.transemb_entity = torch.tensor(transE_entity_data).to(world.device).float()

        transE_r_path = '/home/zhongjian/KG-CLDDI-main/data/TwoSides/DRKG_TransE_l2_relation.npy'
        transE_r_data = np.load(transE_r_path)
        # transE_r_data = transE_r_data / np.linalg.norm(transE_r_data, axis=1)[:, np.newaxis]
        transE_r_data = np.append(transE_r_data, zero_vec, axis=0)
        self.transemb_r = torch.tensor(transE_r_data).to(world.device).float()
        self.embedding_relation = nn.Embedding.from_pretrained(self.transemb_r)
        self.embedding_relation.weight.requires_grad =True#False#

        self.embedding_entity = nn.Embedding.from_pretrained(self.transemb_entity)
        self.embedding_entity.weight.requires_grad = True#False#

        # mol_path = '/home/zhongjian/KGCL-SIGIR22-main/data/TwoSides/Drug_Mol2Vec300.npy'
        # np_molvec = np.load(mol_path)
        # self.molvec = torch.tensor(np_molvec).to(world.device).float()

        self.__init_weight()
        self.gat = GAT(self.latent_dim,
                       self.latent_dim,
                       dropout=0.2,
                       alpha=0.2).train()
        #self.ConvE=ConvE(96766,96766)
        self.layer1 = nn.Sequential(nn.Linear(600, 2048), nn.BatchNorm1d(2048), nn.ReLU(True))#nn.Dropout(0.1),
        self.layer11 = nn.Sequential(nn.Linear(400, 2048), nn.BatchNorm1d(2048), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(2048, 200))

        self.layer4 = nn.Sequential(nn.Linear(2048, 81))
        self.sigmoid = torch.nn.Sigmoid()

        self.layercon1 = nn.Sequential(nn.Linear(600, 300),  nn.ReLU(True))  # 768nn.Dropout(0.5),
        self.layercon2 = nn.Sequential(nn.Linear(300, 200), nn.ReLU(True))#n.Dropout(0.1),

        self.activation = nn.LeakyReLU()
        self.linear1 = nn.Linear(100, 100)  # self.latent_dim, self.latent_dim
        self.linear2 = nn.Linear(100, 100)  # W2 in Equation (8)

        self.fc = nn.Sequential(nn.Linear(300, 300), nn.Dropout(0.1),nn.ReLU(True))#nn.BatchNorm1d(2048)

    def __init_weight(self):
        #self.num_users = self.dataset.n_users#70679
        #self.num_items = self.dataset.m_items#24915
        self.num_entities = self.kg_dataset.entity_count#4799
        self.num_relations = self.kg_dataset.relation_count

        #xavier_normal_(self.embedding_relation.weight.data)
        self.latent_dim = self.config['latent_dim_rec']#64
        self.trans_M = nn.Parameter(torch.Tensor(self.num_relations, self.latent_dim, self.latent_dim))
        self.n_layers = self.config['lightGCN_n_layers']#3
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.AllGraph = self.dataset.getAllSparseGraph()
        self.kg_dict, self.item2relations = self.kg_dataset.get_kg_dict(2307)#2322
        self.A_in = nn.Parameter(torch.FloatTensor(512,512))
        self.A_all = nn.Parameter(torch.FloatTensor(2307, 2307))
        self.reset_parameters()
        identity_matrix = torch.eye(*self.A_in.shape)

        # 将 weight 与单位矩阵相加
        self.Adj = nn.Parameter(self.A_in + identity_matrix)
    def reset_parameters(self):
        stdv = 1. / math.sqrt(512)
        self.A_in.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(2307)
        self.A_all.data.uniform_(-stdv1, stdv1)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()

        #random_index = torch.rand(len(values)) + keep_prob#len(values):157322,random_index[:]>1
        random_index=torch.ones(len(values)).to(world.device)

        #random_index = random_index.int().bool()
        dropnum=int((1-keep_prob)*len(values))
        ran=torch.randperm(len(values)).numpy()

        drops=ran[:dropnum]
        # for idx in drops:
        #     random_index[idx]=0
        random_index[drops]=0
        random_index = random_index.bool()
        index = index[random_index]
        values = values[random_index]#/keep_prob#bigger
        #print(len(values))
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        #S1
        graph = self.__dropout_x(self.Graph, keep_prob)
        return graph



    def computer(self,drop_prob=0.95):
        """
        propagate methods for lightGCN
        """
        #users_emb = self.cal_item_embedding_from_kg(self.kg_dict)#self.transE_entity_data
        items_emb = self.cal_item_embedding_from_kg(self.kg_dict)#self.transE_entity_data
        all_emb = torch.cat([items_emb, items_emb])
        embs = []#all_emb
        embs1=[]
        if self.config['dropout']:
            if self.training:
                #g_droped = self.Graph
                g_droped = self.__dropout(drop_prob)#self.keep_prob
            else:
                #S1
                #g_droped = self.AllGraph
                g_droped = self.Graph
        else:
            # S1
            #g_droped = self.AllGraph
            g_droped = self.Graph

        for layer in range(self.n_layers):#3
            side_embeddings = torch.sparse.mm(g_droped, all_emb)
            side_embeddings=F.dropout(side_embeddings,0.1, training=self.training)
            sum_embeddings = self.activation(self.linear1(all_emb + side_embeddings))#
            bi_embeddings = self.activation(self.linear2(all_emb * side_embeddings))#
            embeddings = bi_embeddings + sum_embeddings
            all_emb = F.normalize(embeddings, p=2, dim=1)
            #side_embeddings = F.normalize(side_embeddings, p=2, dim=1)
            embs.append(all_emb)
            embs1.append(side_embeddings)
        embs = torch.stack(embs, dim=1)#[95594, 4, 64]
        light_out = torch.mean(embs, dim=1)#[95594, 64]

        users, items = torch.split(light_out, [items_emb.shape[0], items_emb.shape[0]])#//users:[70679, 64],items:[24915, 64]
        embs1 = torch.stack(embs1, dim=1)  # [95594, 4, 64]
        light_out1 = torch.mean(embs1, dim=1)  # [95594, 64]
        users1, items1 = torch.split(light_out1, [items_emb.shape[0], items_emb.shape[0]])#//users:[70679, 64],items:[24915, 64]
        return users,items,users1, items1,items_emb

    def dropoedge(self, adj, keep_prob):
        B = np.random.binomial(n=1, p=keep_prob, size=adj.shape)
        #D = np.diag(np.sum(adj, axis=1))
        # 将A和B逐元素相乘，得到新的邻接矩阵A'
        A_prime = adj.cpu().data.numpy() * B
        A_prime=torch.FloatTensor(A_prime).to(world.device)
        # 计算规范化的邻接矩阵
        #D_sqrt_inv = np.linalg.inv(np.sqrt(D))
        #adj_norm = np.dot(np.dot(D_sqrt_inv, A_prime), D_sqrt_inv)
        return A_prime

    def getDDIEmbedding(self,DDIone,DDItwo,neg,drop_prob=0.95):
        #DDIone_conv, DDItwo_conv,all_entitis = self.computer1(DDIone,DDItwo,drop_prob)
        DDIone_emb, DDItwo_emb, DDIone_conv, DDItwo_conv,all_entitis = self.computer(drop_prob)

        neg_entitis = all_entitis[neg.long()]
        neg_emb = DDItwo_emb[neg.long()]
        neg_conv = DDItwo_conv[neg.long()]

        #DDIone_conv = self.embedding_entity.weight[DDIone.long()]  # all_entitis
        #DDItwo_conv = self.embedding_entity.weight[DDItwo.long()]
        DDIone_entitis = all_entitis[DDIone.long()]
        DDItwo_entitis = all_entitis[DDItwo.long()]

        DDIone_emb = DDIone_emb[DDIone.long()]
        DDItwo_emb = DDItwo_emb[DDItwo.long()]

        DDIone_conv = DDIone_conv[DDIone.long()]
        DDItwo_conv = DDItwo_conv[DDItwo.long()]



        return DDIone_conv, DDIone_entitis,DDIone_emb, DDItwo_conv, DDItwo_entitis,DDItwo_emb,neg_conv,neg_entitis,neg_emb
        #return DDIone_conv,DDIone_entitis,DDIone_emb, DDItwo_conv, DDItwo_entitis,DDItwo_emb

    def getDDIEmbedding1(self, DDIone, DDItwo, neg, drop_prob=0.95):
        # DDIone_conv, DDItwo_conv,all_entitis = self.computer1(DDIone,DDItwo,drop_prob)
        DDIone_emb, DDItwo_emb, DDIone_conv, DDItwo_conv, all_entitis = self.computer(drop_prob)

        DDIone_entitis = all_entitis[DDIone.long()]
        DDItwo_entitis = all_entitis[DDItwo.long()]

        DDIone_emb = DDIone_emb[DDIone.long()]
        DDItwo_emb = DDItwo_emb[DDItwo.long()]

        DDIone_conv = DDIone_conv[DDIone.long()]
        DDItwo_conv = DDItwo_conv[DDItwo.long()]

        return DDIone_conv, DDIone_entitis, DDIone_emb, DDItwo_conv, DDItwo_entitis, DDItwo_emb
        # return DDIone_conv,DDIone_entitis,DDIone_emb, DDItwo_conv, DDItwo_entitis,DDItwo_emb
    def cal_item_embedding_rgat(self,kg: dict):
        item_embs = self.embedding_entity(torch.LongTensor(list(kg.keys())).to(world.device))  # item_num, emb_dim

        item_entities = torch.stack(list(kg.values())).long()  # item_num, entity_num_each
        item_relations = torch.stack(list(self.item2relations.values())).long()

        entity_embs = self.embedding_entity(item_entities)  # item_num, entity_num_each, emb_dim
        relation_embs = self.embedding_relation(item_relations)  # item_num, entity_num_each, emb_dim

        padding_mask = torch.where(item_entities != self.num_entities,#item_entities里的每一个元素进行比较
                                   torch.ones_like(item_entities),#填充了值1的张量，其大小与item_entities相同
                                   torch.zeros_like(item_entities)).float()

        return self.gat.forward_relation(item_embs, entity_embs, relation_embs,padding_mask)#GAT 系数：a,forward_relation

    def cal_item_embedding_from_kg(self,kg: dict = None):
        if kg is None:
            kg = self.kg_dict

        if world.kgcn == "RGAT":#yes
            return self.cal_item_embedding_rgat(kg)
        elif (world.kgcn == "NO"):
            return self.embedding_entity.weight

    #def branch1_forward(self, DDIone_emb,DDIone_entitis,DDItwo_emb,DDItwo_entitis,drop=0.1):#

    def branch1_forward(self, DDIone_conv, DDIone_entitis, DDIone_emb, DDItwo_conv, DDItwo_entitis, DDItwo_emb,drop=0.1):  #
            # labelemb = self.fcl(Labelemb1)
            inner_pro = torch.cat([DDIone_conv, DDIone_entitis, DDIone_emb, DDItwo_conv, DDItwo_entitis, DDItwo_emb],dim=1)
            inner_pro = F.dropout(inner_pro, drop, training=self.training)
            #con_result = self.layercon1(inner_pro)
            #con_result = self.layercon2(con_result)

            inner_pro = self.layer1(inner_pro)
            result = self.layer4(inner_pro)

            return result#, con_result

    def branch2_forward(self, DDIone_conv, DDIone_entitis, DDIone_emb, DDItwo_conv,drop=0.1):  #

        inner_pro = torch.cat([DDIone_conv, DDIone_entitis, DDIone_emb, DDItwo_conv], dim=1)
        inner_pro = self.layer11(inner_pro)
        inner_pro = F.dropout(inner_pro, drop, training=self.training)
        result = self.layer4(inner_pro)
        return result

    def forward(self, *input,mode):
        if mode == 'branch1':
            return self.branch1_forward(*input)
        if mode == 'branch2':
            return self.branch2_forward(*input)
