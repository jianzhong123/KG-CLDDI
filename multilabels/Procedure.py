
import pandas
from sklearn.preprocessing import label_binarize

from torch.utils.data.dataloader import DataLoader
import world
import numpy as np
import torch

import dataloader

from tqdm import tqdm
import model
import copy
import multiprocessing
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,average_precision_score
from torch.nn import functional as F

CORES = multiprocessing.cpu_count() // 2

def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = torch.nn.functional.normalize(z1)
    z2 = torch.nn.functional.normalize(z2)
    return torch.mm(z1, z2.t())

def calculate_loss(A_embedding, B_embedding):
    # first calculate the sim rec
    tau = 0.6    # default = 0.8
    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(A_embedding, A_embedding))
    between_sim = f(sim(A_embedding, B_embedding))

    loss_1 = -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    ret = loss_1
    if world.multi_labels:
        ret = ret.sum()
    else:
        ret = ret.mean()
    return ret

def calculate_loss_1(A_embedding, B_embedding):
    # first calculate the sim rec
    tau = 0.6    # default = 0.8
    f = lambda x: torch.exp(x / tau)

    refl_sim = f(sim(A_embedding, A_embedding))
    between_sim = f(sim(A_embedding, B_embedding))

    loss_1 = -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    refl_sim_1 = f(sim(B_embedding, B_embedding))
    between_sim_1 = f(sim(B_embedding, A_embedding))
    loss_2 = -torch.log(
        between_sim_1.diag()
        / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
    ret = (loss_1 + loss_2) * 0.5
    ret = ret.sum()
    return ret

def calculate_loss_2(A_embedding, B_embedding):
    # first calculate the sim rec
    tau = 0.6    # default = 0.8
    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(A_embedding, A_embedding))
    between_sim = f(sim(A_embedding, B_embedding))

    loss_1 = -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    refl_sim_1 = f(sim(B_embedding, B_embedding))
    between_sim_1 = f(sim(B_embedding, A_embedding))
    loss_2 = -torch.log(
        between_sim_1.diag()
        / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
    ret = (loss_1 + loss_2) * 0.5
    ret = ret.sum()
    return ret
#SimCSE

def sup_compute_loss(y_pred,lamda=0.6):#0.05
    row = torch.arange(0,y_pred.shape[0],3).to(world.device)
    col = torch.arange(y_pred.shape[0]).to(world.device)
    col = torch.where(col % 3 != 0)[0].to(world.device)
    y_true = torch.arange(0,len(col),2).to(world.device)
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    #torch自带的快速计算相似度矩阵的方法
    similarities = torch.index_select(similarities,0,row)
    similarities = torch.index_select(similarities,1,col)
    #屏蔽对角矩阵即自身相等的loss
    similarities = similarities / lamda
    #论文中除以 temperature 超参 0.05
    loss = F.cross_entropy(similarities,y_true)
    return torch.sum(loss)

def TransR_train(recommend_model, opt):
    recommend_model.train()
    kgdataset = dataloader.KGDataset()
    kgloader = DataLoader(kgdataset,batch_size=4096,drop_last=False)#True
    trans_loss = 0.

    for data in tqdm(kgloader, total=len(kgloader), disable=True):#400000
        heads = data[0].to(world.device)
        relations = data[1].to(world.device)
        pos_tails = data[2].to(world.device)
        neg_tails = data[3].to(world.device)
        #print("heads, relations, pos_tails, neg_tails",heads.shape, relations.shape, pos_tails.shape, neg_tails.shape)
        kg_batch_loss = recommend_model(heads, relations, pos_tails, neg_tails)#
        trans_loss += kg_batch_loss / len(kgloader)

        opt.zero_grad()
        kg_batch_loss.backward()
        opt.step()
    return trans_loss

def DDItrain(recommend_model,optimizer,train_X, train_Y,Labelfeatures,scheduler):
    Recmodel: model.KGCL = recommend_model
    Recmodel.train()

    #if world.multi_labels:
        #traindf = pandas.read_csv("../data/TwoSides/DDITrainOrgin.csv", delimiter=',',header=None)  #
        # S1
        #traindf = pandas.read_csv("../data/TwoSides/MUltiLabel_TrainAll_S1.csv", delimiter=',', header=None)#DDITrainOrgin
        # S2
        #traindf = pandas.read_csv("../data/TwoSides/MUltiLabel_TrainAll_S2.csv", delimiter=',', header=None)

    data = train_X#traindf.values

    DDIone = data[:, 0]
    DDItwo = data[:, 1]

    DDIone = torch.tensor(DDIone).to(world.device)
    DDItwo = torch.tensor(DDItwo).to(world.device)

    label=train_Y
    label = torch.tensor(label).to(world.device)

    u_batch_size = 256
    torch_dataset_train = Data.TensorDataset(DDIone, DDItwo,label)

    loader_train = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=u_batch_size,
        shuffle=True,
        #drop_last=True
    )
    loader_train2 = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=u_batch_size,
        shuffle=True,
        #drop_last=True
    )
    print("len",len(loader_train))
    train_loss=0
    if world.multi_labels:
        criterion = torch.nn.BCELoss(reduction='sum')
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    criterion = criterion.to(world.device)

    # 循环迭代两个 DataLoader，每个迭代器中包含 256 个样本
    for data, data3 in zip(loader_train, loader_train2):
        optimizer.zero_grad()

        users, items, batch_label = data
        users3, items3, batch_label3 = data3
        '''
        Labelemb1 = []
        npusers=users.cpu().detach().numpy()
        npitems=items.cpu().detach().numpy()
        for  idx in range (0,npusers.shape[0]):
            drug1=npusers[idx]
            drug2=npitems[idx]
            key=str(drug1) + "," + str(drug2)
            Labelemb1.append(Labelfeatures[key])
        Labelemb1=torch.tensor(Labelemb1).to(world.device).float()

        Labelemb3 = []
        npusers3=users3.cpu().detach().numpy()
        npitems3=items3.cpu().detach().numpy()
        for  idx in range (0,npusers3.shape[0]):
            drug1=npusers3[idx]
            drug2=npitems3[idx]
            key=str(drug1) + "," + str(drug2)
            Labelemb3.append(Labelfeatures[key])
        Labelemb3=torch.tensor(Labelemb3).to(world.device).float()
        '''
        DDIone_conv,DDIone_entitis,DDIone_emb, DDItwo_conv, DDItwo_entitis,DDItwo_emb = Recmodel.getDDIEmbedding(users, items,0.95)
        out1 = Recmodel(DDIone_conv,DDIone_entitis,DDIone_emb, DDItwo_conv, DDItwo_entitis,DDItwo_emb, 0.1, mode='branch1')
        out1 = out1.squeeze(-1)
        DDIone_conv2,DDIone_entitis2,DDIone_emb2, DDItwo_conv2, DDItwo_entitis2,DDItwo_emb2 = Recmodel.getDDIEmbedding(users, items, 0.95)
        #out = Recmodel(Labelemb1,DDItwo_conv1, DDItwo_items1,DDItwo_emb1, DDIone_conv1, DDIone_items1,DDIone_emb1, 0.2, mode='branch2')
        out = Recmodel(DDItwo_conv2, DDItwo_entitis2,DDItwo_emb2,DDIone_conv2,DDIone_entitis2,DDIone_emb2, 0.4,  mode='branch1')
        out = out.squeeze(-1)

        DDIone_conv3,DDIone_entitis3,DDIone_emb3, DDItwo_conv3, DDItwo_entitis3,DDItwo_emb3 = Recmodel.getDDIEmbedding(users3, items3, 1)
        out3 = Recmodel(DDIone_conv3,DDIone_entitis3,DDIone_emb3, DDItwo_conv3, DDItwo_entitis3,DDItwo_emb3, 0.1, mode='branch1')
        out3 = out3.squeeze(-1)

        mul_loss = criterion(out, batch_label.float())
        mul_loss1 = criterion(out1, batch_label.float())
        out = out.cpu().tolist()
        out1 = out1.cpu().tolist()
        out3 = out3.cpu().tolist()
        y_pred = []
        for index in range(len(out)):
            y_pred.append(out[index])
            y_pred.append(out1[index])
            y_pred.append(out3[index])

        y_pred = torch.FloatTensor(y_pred).to(world.device)
        con_loss = sup_compute_loss(y_pred)  # compute_loss
        con_loss = con_loss.requires_grad_(True)
        loss=(mul_loss+mul_loss1)+0.1*con_loss#+0.01*con_loss

        train_loss+=loss
        loss.backward()
        optimizer.step()
        #if  world.multi_labels:
        #scheduler.step()
    return train_loss/500#len(loader_train)

def multiclass_calc_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_precision = precision_score(y_true, y_pred, average='micro')
    micro_recall = recall_score(y_true, y_pred, average='micro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    #print("Test,",acc, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1)
    return acc, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1


def multilabels_calc_metrics(y_true, y_pred, pred_score):
    acc = accuracy_score(y_true.data.cpu().numpy(), pred_score.data.cpu().numpy())
    auc = roc_auc_score(y_true.data.cpu().numpy(), y_pred.data.cpu().numpy())
    prc_auc = average_precision_score(y_true.data.cpu().numpy(), y_pred.data.cpu().numpy())

    #precision = precision_score(y_true, y_pred)
    #recall = recall_score(y_true, y_pred)
    #f1 = f1_score(y_true, y_pred)
    return auc,prc_auc,acc


def DDITest_Bin_MultiLabels(Recmodel,epoch,test_X, test_Y,Labelfeatures):
    data =test_X

    DDIone = data[:,0]
    DDItwo = data[:,1]

    label = test_Y

    DDIone = np.array(DDIone,dtype=int)
    DDItwo = np.array(DDItwo,dtype=int)

    DDIone = torch.tensor(DDIone).to(world.device)
    DDItwo = torch.tensor(DDItwo).to(world.device)
    label = torch.tensor(label).to(world.device)
    Recmodel: model.KGCL

    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    auc_list = []
    pr_list = []
    acc_list = []

    macro_precision_list = []
    macro_recall_list = []
    macro_f1_list = []
    micro_precision_list = []
    micro_recall_list = []
    micro_f1_list = []

    u_batch_size = 256
    torch_dataset_test = Data.TensorDataset(DDIone, DDItwo, label)

    loader_test = Data.DataLoader(
        dataset=torch_dataset_test,
        batch_size=u_batch_size,
        shuffle=True,
        drop_last=True
    )

    outs= np.empty((0, 200))
    out1s=np.empty((0, 200))
    results=np.empty((0, 200))
    batch_labels=np.empty((0))
    predictions=np.empty((0))
    for epoch,data in enumerate(loader_test):
        with torch.no_grad():
            drugs1, drugs2, batch_label = data
            '''
            Labelemb1 = []
            npdrug1 = drugs1.cpu().detach().numpy()
            npdrug2 = drugs2.cpu().detach().numpy()
            for idx in range(0, npdrug1.shape[0]):
                drugone = npdrug1[idx]
                drugtwo = npdrug2[idx]
                key = str(drugone) + "," + str(drugtwo)
     
                Labelemb1.append(Labelfeatures[key])
            
            Labelemb1 = torch.tensor(Labelemb1).to(world.device).float()
            '''
            DDIone_conv, DDIone_entitis, DDIone_emb, DDItwo_conv, DDItwo_entitis, DDItwo_emb = Recmodel.getDDIEmbedding(drugs1, drugs2, 1)
            out1 = Recmodel(DDIone_conv, DDIone_entitis, DDIone_emb, DDItwo_conv, DDItwo_entitis,DDItwo_emb, 0.0, mode='branch1')
            out1 = out1.squeeze(-1)
            #DDIone_conv2, DDIone_entitis2, DDIone_emb2, DDItwo_conv2, DDItwo_entitis2, DDItwo_emb2 = Recmodel.getDDIEmbedding(drugs1, drugs2,1)
            # out = Recmodel(Labelemb1,DDItwo_conv1, DDItwo_items1,DDItwo_emb1, DDIone_conv1, DDIone_items1,DDIone_emb1, 0.2, mode='branch2')
            out = Recmodel( DDItwo_conv, DDItwo_entitis,DDItwo_emb, DDIone_conv, DDIone_entitis, DDIone_emb, 0.0, mode='branch1')
            out = out.squeeze(-1)
            out1s = np.concatenate((out1s, out1.data.cpu().numpy()), axis=0)
            outs = np.concatenate((outs, out.data.cpu().numpy()), axis=0)
            #if world.multi_labels:
            out=0.5*out+0.5*out1#out1#
            results = np.concatenate((results, out.data.cpu().numpy()), axis=0)
            prediction = copy.deepcopy(out)
            prediction[prediction >= 0.5] = 1
            prediction[prediction < 0.5] = 0
            if world.multi_labels==True:
                auc, pr,acc = multilabels_calc_metrics(batch_label, out,prediction)
                auc_list.append(auc)
                pr_list.append(pr)
                acc_list.append(acc)
            else:
                prediction = torch.max(out, 1)[1]
                prediction = prediction.data.cpu().numpy()
                batch_label = batch_label.data.cpu().numpy()

                y_score=torch.sigmoid(out)
                y_score = y_score.data.cpu().numpy()
                y_score = (y_score.T / y_score.sum(axis=1)).T
                y_one_hot = label_binarize(batch_label, np.arange(81))
                y_scores = np.concatenate((y_scores, y_score), axis=0)
                y_one_hots = np.concatenate((y_one_hots, y_one_hot), axis=0)

                #acc, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = multiclass_calc_metrics(batch_label,prediction)
                batch_labels=np.concatenate((batch_labels, batch_label), axis=0)
                predictions = np.concatenate((predictions, prediction), axis=0)
                # acc_list.append(acc)
                # macro_precision_list.append(macro_precision)
                # macro_recall_list.append(macro_recall)
                # macro_f1_list.append(macro_f1)
                # micro_precision_list.append(micro_precision)
                # micro_recall_list.append(micro_recall)
                # micro_f1_list.append(micro_f1)

    if world.multi_labels == True:
        auc = np.mean(auc_list)
        pr = np.mean(pr_list)
        acc = np.mean(acc_list)
        print("Test", epoch, auc, pr, acc)
        return auc,outs,out1s,results
    else:
        # macro_precision = np.mean(macro_precision_list)
        # macro_recall = np.mean(macro_recall_list)
        # macro_f1 = np.mean(macro_f1_list)
        # micro_precision = np.mean(micro_precision_list)
        # micro_recall = np.mean(micro_recall_list)
        # micro_f1 = np.mean(micro_f1_list)
        # acc = np.mean(acc_list)
        acc, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = multiclass_calc_metrics(batch_labels,predictions)

        #auc = roc_auc_score(y_one_hots, y_scores, multi_class='ovr')
        print("Test:", epoch, macro_precision, macro_recall, macro_f1, acc)#,auc
        return acc

