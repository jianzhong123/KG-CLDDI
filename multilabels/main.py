import csv

import pandas
from sklearn.model_selection import KFold
from tqdm import tqdm

from torch import optim
import world
import utils as utils
import Procedure, register
from world import cprint
import torch
import numpy as np
import time


# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

from register import dataset, kg_dataset



Recmodel = register.MODELS[world.model_name](world.config, dataset, kg_dataset)
Recmodel = Recmodel.to(world.device)#GPU or CPU


weight_file = utils.getFileName()#/home/zhongjian/KGCL-SIGIR22-main/code/checkpoints/kgc-MIND-64.pth.tar
print(f"will save to {weight_file}")
Neg_k = 0.5

least_loss = 1e5
best_result = 0
stopping_step = 0


if world.multi_labels:
    optimizer = optim.AdamW(Recmodel.parameters(), lr=0.01)#
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, cycle_momentum=False,step_size_up=292,step_size_down=307)#
    #S1
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, cycle_momentum=False,step_size_up=238,step_size_down=238)#
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30,gamma=0.1)  #

Labelfeatures={}
# traindf = pandas.read_csv("../data/TwoSides/zhong_All_Labelvectors.csv", delimiter=',', header=None)
# data=traindf.values
# for row in data:
#     key = str(row[0]) + "," + str(row[1])
#     Labelfeatures[key] = eval(row[2])

#traindf = pandas.read_csv("../data/TwoSides/zhong_All_Labelvectors.csv", delimiter=',', header=None)
# tqdm长循环中添加一个进度提示信息
for epoch in tqdm(range(150), disable=True):#world.TRAIN_epochs:1000
    start = time.time()
    cprint("[Trans]")

    #print("out1", transmodel.ent_embedding.weight[0])
    train_loss = Procedure.DDItrain(Recmodel, optimizer,dataset.train_X, dataset.train_Y,Labelfeatures,scheduler)#trans_loss
    print("train",epoch,train_loss)

    #test
    # if world.multi_labels:
    #     traindf = pandas.read_csv("../data/TwoSides/DDITestOrgin.csv", delimiter=',', header=None)#
        #S1
        #traindf = pandas.read_csv("../data/TwoSides/MUltiLabel_Test_S1.csv", delimiter=',', header=None)#DDITestOrgin
        # S2
        #traindf = pandas.read_csv("../data/TwoSides/MUltiLabel_Test_S2.csv", delimiter=',', header=None)#DDITestOrgin

    auc,outs,out1s,results=Procedure.DDITest_Bin_MultiLabels(Recmodel,epoch,dataset.test_X, dataset.test_Y,Labelfeatures)
    scheduler.step()
    if auc > best_result:
        stopping_step = 0
        best_result = auc
        print("find a better model")
        if world.SAVE:
            print("save...")
            #pretrain, att, DDIone_emb,DDIone_conv = Recmodel.test_getDDIEmbedding(1)
            #light=torch.cat([DDIone_emb,DDIone_conv],dim=1)
            # np.savetxt('mutilabel-pretrain.csv', pres, delimiter=',')
            # np.savetxt('multilabel-att.csv', atts, delimiter=',')
            # np.savetxt('multilabel-light.csv', lights, delimiter=',')

            np.savetxt('multilabel-out.csv', outs, delimiter=',')
            np.savetxt('multilabel-out1.csv', out1s, delimiter=',')
            np.savetxt('multilabel-result.csv', results, delimiter=',')
            torch.save(Recmodel.state_dict(), weight_file)

    else:
        stopping_step += 1
        if stopping_step >= world.early_stop_cnt:
            print(f"early stop triggerd at epoch {epoch}")
            break
