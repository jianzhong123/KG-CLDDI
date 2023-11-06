import pandas
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
#print(f"will save to {weight_file}")
Neg_k = 0.5

least_loss = 1e5
best_result = 0
stopping_step = 0


print("multi_class")
optimizer = optim.AdamW(Recmodel.parameters(), lr=0.01)#0.001
#scheduler =torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, cycle_momentum=False,step_size_down=539,step_size_up=539)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
#S1
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, cycle_momentum=False,step_size_up=544, step_size_down=544)


# tqdm长循环中添加一个进度提示信息
for epoch in tqdm(range(150), disable=True):#world.TRAIN_epochs:1000
    start = time.time()
    cprint("[Trans]")

    #print("out1", transmodel.ent_embedding.weight[0])
    train_loss = Procedure.DDItrain(Recmodel, optimizer,dataset.train_X, dataset.train_Y,scheduler)#trans_loss
    print("train",epoch,train_loss)

    #test
    #traindf = pandas.read_csv("../data/processdata/DDITestOrgin0.txt", delimiter=' ', header=None)

    out1s,outs,results,auc=Procedure.DDITest_Bin_MultiLabels(Recmodel,epoch,dataset.test_X, dataset.test_Y)
    scheduler.step()
    if auc > best_result:
        stopping_step = 0
        best_result = auc
        print("find a better model")
        if world.SAVE:
            print("save...")



            torch.save(Recmodel.state_dict(), weight_file)

    else:
        stopping_step += 1
        if stopping_step >= world.early_stop_cnt:
            print(f"early stop triggerd at epoch {epoch}")
            break
