import torch as t
import torch.nn as nn


class RateLoss(nn.Module):
    def __init__(self, var_noise=1, k=15):
        super(RateLoss, self).__init__()
        self.var_noise = var_noise
        self.k = k
    def forward(self, csi, power):
        power = t.unsqueeze(power, -1)
        csi = csi.permute(0, 2, 1)
        csi = t.pow(csi, 2)
        rx_power = t.mul(csi, power)
        mask = t.eye(self.k)
        valid_rx_power = t.sum(t.mul(rx_power, mask), 1)
        interference = t.sum(t.mul(rx_power, 1 - mask), 1) + self.var_noise
        rate = t.log2(1 + t.div(valid_rx_power, interference)) 
        sum_rate = t.mean(t.sum(rate, 1))
        loss = t.neg(sum_rate)
        
        return loss

class EELoss(nn.Module):
    def __init__(self, noise_power, user_num,pc):
        super(EELoss, self).__init__()
        self.noise_power = noise_power
        self.user_num = user_num
        self.pc = pc
    def forward(self, csi, power):
        power = t.unsqueeze(power, -1)
        csi = csi.permute(0,2,1) 
        csi = t.pow(csi, 2)
        rx_power = t.mul(csi, power)
        mask = t.eye(self.user_num)
        valid_rx_power = t.sum(t.mul(rx_power, mask), 1)
        interference = t.sum(t.mul(rx_power, 1 - mask), 1) + self.noise_power
        rate = t.log2(1 + t.div(valid_rx_power, interference))
        ee = t.div(rate,(t.squeeze(power)+t.FloatTensor([self.pc])))
        sum_ee = t.mean(t.sum(ee, 1))
        loss = t.neg(sum_ee)

        return loss
