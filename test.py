import os
import torch as t
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader 
from dataset import PADataset
from network import *
from utils import *

K = 10
var_noise = 1.0
Pmax = 1.0

Pc = 0.5
dir_data = os.path.join(os.getcwd(),"dataset")

print("=============>Loading test datasets")
test_dataset = PADataset(dir_data, "test", K)
test_dataloader = DataLoader(test_dataset, 1000)

model = MBPANet(K)
path_model = os.path.join(os.getcwd(),"checkpoint","pretrained_model",f"with_{K}BS_model.pth")
model.load_state_dict(t.load(path_model))
print("network model loaded")
for i, data in enumerate(test_dataloader):
    vaild_csi, vaild_target_power = data
    vaild_csi = vaild_csi.float()
    vaild_target_power = vaild_target_power.float()
    predict_power = model(vaild_csi,0)
    sum_ee = ee_perf_eval(vaild_csi, predict_power, var_noise, K, Pc)
    predict_power = model(vaild_csi, 1)
    sum_rate = rate_perf_eval(vaild_csi, predict_power,var_noise,K)
    print(f"sum_ee: {sum_ee.item():0.5f} sum_rate: {sum_rate.item():0.5f}")