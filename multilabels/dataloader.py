import collections
import os
from os.path import join
import pickle
import sys
import random

import pandas
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world as world
from world import cprint
from time import time


class KGDataset(Dataset):
    def __init__(self,kg_path=join(world.DATA_PATH,"TwoSides", "kg.txt")):#world.dataset,bi_kg,
        kg_data = pd.read_csv(kg_path,
                              sep=' ',
                              names=['h', 'r', 't'],
                              engine='python')
        self.kg_data = kg_data.drop_duplicates()
        self.kg_dict, self.heads = self.generate_kg_data(kg_data=self.kg_data)
        self.item_net_path = join(world.DATA_PATH, world.dataset)

    @property
    def entity_count(self):
        # max entity id + 2 (starts with 0)
        return self.kg_data['t'].max() +1

    @property
    def relation_count(self):
        return self.kg_data['r'].max() + 1

    def get_kg_dict(self, item_num):
        entity_num = world.entity_num_per_item
        i2es = dict()
        i2rs = dict()
        for item in range(item_num):
            rts = self.kg_dict.get(item, False)
            if rts:
                tails = list(map(lambda x: x[1], rts))
                relations = list(map(lambda x: x[0], rts))

                if (len(tails) > entity_num):
                    count = random.sample(rts, entity_num)
                    tails = list(map(lambda x: x[1], count))
                    relations = list(map(lambda x: x[0], count))
                    i2es[item] = torch.IntTensor(tails).to(world.device)#[:entity_num]
                    i2rs[item] = torch.IntTensor(relations).to(world.device)#[:entity_num]
                else:
                    # last embedding as padding idx
                    tails.extend([self.entity_count] *
                                 (entity_num - len(tails)))
                    relations.extend([self.relation_count] *
                                     (entity_num - len(relations)))
                    i2es[item] = torch.IntTensor(tails).to(world.device)
                    i2rs[item] = torch.IntTensor(relations).to(world.device)
            else:
                i2es[item] = torch.IntTensor([self.entity_count] *
                                             entity_num).to(world.device)
                i2rs[item] = torch.IntTensor([self.relation_count] *
                                             entity_num).to(world.device)
        return i2es, i2rs

    def generate_kg_data(self, kg_data):
        # construct kg dict
        kg_dict = collections.defaultdict(list)
        for row in kg_data.iterrows():
            h, r, t = row[1]
            kg_dict[h].append((r, t))
        heads = list(kg_dict.keys())
        return kg_dict, heads

    def __len__(self):
        return len(self.kg_dict)

    def __getitem__(self, index):
        head = self.heads[index]
        relation, pos_tail = random.choice(self.kg_dict[head])
        while True:
            neg_head = random.choice(self.heads)
            neg_tail = random.choice(self.kg_dict[neg_head])[1]
            if (relation, neg_tail) in self.kg_dict[head]:
                continue
            else:
                break
        return head, relation, pos_tail, neg_tail


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        raise NotImplementedError

    def getSparseGraph(self):
        raise NotImplementedError

def load_DDI_data(filename):
    train_X_data = []
    train_Y_data = []
    test_X_data = []
    test_Y_data = []

    traindf = pandas.read_csv(filename, delimiter=',', header=None)
    data = traindf.values
    DDI = data[:, 0:2]
    label = data[:, 2:]
    #label = np.array(list(map(int, Y)))

    # print(DDI.shape)
    # print(label.shape)

    kfold = KFold(n_splits=5, shuffle=True, random_state=3)

    for train, test in kfold.split(DDI, label):
        train_X_data.append(DDI[train])
        train_Y_data.append(label[train])
        test_X_data.append(DDI[test])
        test_Y_data.append(label[test])

    print('Loading DDI data down!')

    return train_X, train_Y, test_X, test_Y

def load_DDI_data_SPartition():
    # S1
    train_file = '../data/TwoSides/MUltiLabel_TrainAll_S1.csv'
    test_file = '../data/TwoSides/MUltiLabel_Test_S1.csv'

    # S2
    # train_file = '../data/TwoSides/MUltiLabel_TrainAll_S2.csv'
    # test_file = '../data/TwoSides/MUltiLabel_Test_S2.csv'

    traindf = pandas.read_csv(train_file, delimiter=',', header=None)
    data = traindf.values
    train_X_data= data[:, 0:2]
    train_Y_data = data[:, 2:]

    testf = pandas.read_csv(test_file, delimiter=',', header=None)
    data = testf.values
    test_X_data = data[:, 0:2]
    test_Y_data = data[:, 2:]


    print('Loading DDI data down!')

    #return train_X, train_Y, test_X, test_Y
    return train_X_data,train_Y_data,test_X_data,test_Y_data

class UILoader(BasicDataset):
    def __init__(self, config=world.config, path="../data/TwoSides"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0

        #transductive setting
        # train_X, train_Y, test_X, test_Y = load_DDI_data("../data/TwoSides/TWOSIDES_refine_Multilabels_top600.txt")
        # self.train_X= train_X[0]
        # self.test_X = test_X[0]
        # self.train_Y = train_Y[0]
        # self.test_Y = test_Y[0]

        #inductive setting
        train_X, train_Y, test_X, test_Y = load_DDI_data_SPartition()
        self.train_X = train_X
        self.test_X = test_X
        self.train_Y = train_Y
        self.test_Y = test_Y

        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        validUniqueUsers, validItem, validUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0
        self.validDataSize = 0
        self.alldataSize = 0
        self.all_user = 0
        self.all_item = 0

        #trainLabels_file = path + '/zhong_All_multilabels.csv'
        trainLabels_file = path + '/TWOSIDES_refine_Multilabels_top600.txt'
        traindf = pandas.read_csv(trainLabels_file, delimiter=',', header=None)

        self.Item_labels = {}

        Item_labels = traindf.values
        for ele in Item_labels:
            key = str(str(ele[0]) + "," + str(ele[1]))
            if world.multi_labels:
                value = ele[2:]
            else:
                value = ele[2]
            self.Item_labels[key] = value

        allUniqueUsers, allItem, allUser = [], [], []
        # with open(all_file) as f:
        #     for l in f.readlines():
        #         if len(l) > 0:
        #             l = l.strip('\n').split(' ')
        #             items = [int(i) for i in l[1:]]
        #             uid = int(l[0])
        #             allUniqueUsers.append(uid)
        #             allUser.extend([uid] * len(items))
        #             allItem.extend(items)
        #             self.all_item = max(self.all_item, max(items))
        #             self.all_user = max(self.all_user, uid)
        #             self.alldataSize += len(items)
        # self.allUniqueUsers = np.array(allUniqueUsers)
        # self.allUser = np.array(allUser)  # 652514
        # self.allItem = np.array(allItem)  # 652514


        for ddi in self.train_X:
            items = ddi[1]
            uid = int(ddi[0])
            trainUniqueUsers.append(uid)
            trainUser.extend([uid])
            trainItem.extend([items])
            self.m_item = max(self.m_item, items)
            self.n_user = max(self.n_user, uid)
            self.traindataSize += 1
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)#652514
        self.trainItem = np.array(trainItem)#652514



        self.m_item += 1
        self.n_user += 1
        #self.all_item =2307
        #self.all_user=2307

        for ddi in self.test_X:
            items = ddi[1]
            uid = int(ddi[0])
            testUniqueUsers.append(uid)
            testUser.extend([uid])
            testItem.extend([items])
            self.testm_item = max(self.m_item, items)
            self.testn_user = max(self.n_user, uid)
            self.testDataSize += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)  # 193920
        self.testItem = np.array(testItem)  # 193920

        #self.testm_item +=1
        #self.testn_user +=1

        self.Graph = None
        self.AllGraph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(
            f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}"
        )

        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item))

        #self.allUser=np.concatenate([self.trainUser, self.testUser])
        #self.allItem = np.concatenate([self.trainItem, self.testItem])

        # self.allNet = csr_matrix(
        #     (np.ones(len(self.allUser)), (self.allUser, self.allItem)),
        #     shape=(self.all_user, self.all_item))
        print("train",self.m_item,self.n_user)
        print("test",self.testm_item, self.testn_user)
        self.TestUserItemNet = csr_matrix(
            (np.ones(len(self.testUser)), (self.testUser, self.testItem)),
            shape=(self.testm_item, self.testn_user))

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def item_groups(self):
        with open(self.path + "/item_groups.pkl", 'rb') as f:
            g = pickle.load(f)
        return g

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(
                self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(
                    world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    # Inductive Test graph
    def getAllSparseGraph(self):
        print("loading adjacency matrix")
        if self.AllGraph is None:

            print("generating adjacency matrix")
            s = time()
            adj_mat = sp.dok_matrix(
                (self.testn_user + self.testm_item, self.testn_user + self.testm_item),
                dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.TestUserItemNet.tolil()
            adj_mat[:self.testn_user, self.testn_user:] = R  # 右上角部分
            adj_mat[self.testn_user:, :self.testn_user] = R.T  # 左下角部分
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time()
            print(f"costing {end - s}s, saved norm_mat...")
            #sp.save_npz(self.path + '/allgraph.npz', norm_adj)


            self.AllGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.AllGraph = self.AllGraph.coalesce().to(world.device)
            print("don't split the matrix")
        return self.AllGraph

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:

            print("generating adjacency matrix")
            s = time()
            adj_mat = sp.dok_matrix(
                (self.n_users + self.m_items, self.n_users + self.m_items),
                dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.UserItemNet.tolil()
            adj_mat[:self.n_users, self.n_users:] = R#右上角部分
            adj_mat[self.n_users:, :self.n_users] = R.T#左下角部分
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time()
            print(f"costing {end-s}s, saved norm_mat...")
            #sp.save_npz(self.path + '/graph.npz', norm_adj)


            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(world.device)
            print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users,
                                         items]).astype('uint8').reshape(
                                             (-1, ))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # train loader and sampler part
    def __len__(self):
        return self.traindataSize

    def __getitem__(self, idx):
        user = self.trainUser[idx]
        pos = random.choice(self._allPos[user])
        key = str(user) + "," + str(pos)
        label = self.Item_labels[key]
        while True:
            neg = np.random.randint(0, self.m_item)
            if neg in self._allPos[user]:
                continue
            else:
                break
        return user, pos, neg,label
