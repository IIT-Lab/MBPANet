import  os
import torch as t
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader 
from dataset import PADataset
from network import *
from utils import *
from loss import *

K = 10
var_noise = 1.0
Pmax = 1.0
Pc = 0.5

nEpochs = 200
dir_data = os.path.join(os.getcwd(),"dataset")
print("=============>Loading training datasets")
train_dataset = PADataset(dir_data, "train", K)
train_dataloader = DataLoader(train_dataset,500)

print("=============>Loading test datasets")
test_dataset = PADataset(dir_data, "test", K)
test_dataloader = DataLoader(test_dataset, 1000)
for i, data in enumerate(test_dataloader):
    vaild_csi, vaild_target_power = data
    vaild_csi = vaild_csi.float()
    target_power = vaild_target_power.float()
    # just 1000 samples for vaild
    break


print("=============>Building model")

model =  MBPANet(K) 

criterion_rate = RateLoss(var_noise,K,)
criterion_ee = EELoss(var_noise,K,Pc)
criterion_mse = nn.MSELoss(reduce=True, size_average=True) 



optimizer = optim.Adam(model.parameters(), lr = 0.001) 
optimizer_ae = optim.Adam(model.parameters(), lr = 0.001)
print("=============>Start Training")
global_step = 0

for epoch in range(nEpochs):
    for i, data in enumerate(train_dataloader):
        global_step += 1
        csi, target_power = data
        csi = csi.float()
        target_power = target_power.float() #
        batch_size = csi.shape[0]
        if global_step % 3 ==1: # 
            alg_id = 0
            train_mode = 1
            optimizer.zero_grad()
            predict_power = model(csi,alg_id)

            loss = criterion_ee(csi,predict_power)# 
            loss_ee = loss.item()
            loss.backward()
            optimizer.step()
        if global_step % 3 ==2:
            alg_id = 1
            train_mode = 2
            optimizer.zero_grad()
            predict_power = model(csi,alg_id)
            loss = criterion_rate(csi, predict_power)  #
            loss_rate = loss.item()
            loss.backward()
            optimizer.step()

        if global_step % 3 ==0:
            optimizer_ae.zero_grad() 
            predict_power = model(csi)
            loss = criterion_mse(predict_power,target_power) 
            loss_mse = loss.item()
            loss.backward()
            optimizer_ae.step()
        if global_step % 100 == 0:
            # 
            sum_rate,sum_ee = perf_eval(vaild_csi,model,var_noise,K,Pc)
                        
            # print(f"Global_Step:{global_step} MSE_Loss:{loss_mse:0.4f} EE_Loss:{loss_ee:0.4f} RATE_Loss:{loss_rate:0.4f}")
            print(f"Epoch: [{epoch}]/[{nEpochs}] sum_rate: {sum_rate:0.4f}  sum_ee: {sum_ee:0.4f}")
            
            # save model
            t.save(model.state_dict(),f'checkpoint/model/with_{K}BS_model.pth')


print("==================>End Training")