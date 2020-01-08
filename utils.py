import numpy as np 
import torch as t 
import scipy.io as sio

def rate_perf_eval(csis, power, var_noise, k):

    power = t.unsqueeze(power, -1)
    csis = csis.permute(0, 2, 1)
    csis = t.pow(csis, 2)
    rx_power = t.mul(csis, power)
    mask = t.eye(k)
    valid_rx_power = t.sum(t.mul(rx_power, mask), 1)
    interference = t.sum(t.mul(rx_power, 1 - mask), 1) + var_noise
    rate = t.log2(1 + t.div(valid_rx_power, interference))
    sum_rate = t.mean(t.sum(rate, 1))

    return sum_rate

def ee_perf_eval(csi, power, var_noise, k,Pc):

    power = t.unsqueeze(power, -1)
    csi = csi.permute(0,2,1)
    csi = t.pow(csi, 2)
    rx_power = t.mul(csi, power)
    mask = t.eye(k)
    valid_rx_power = t.sum(t.mul(rx_power, mask), 1)
    interference = t.sum(t.mul(rx_power, 1 - mask), 1) + var_noise
    rate = t.log2(1 + t.div(valid_rx_power, interference))
    ee = t.div(rate,(t.squeeze(power)+t.FloatTensor([Pc])))
    sum_ee = t.mean(t.sum(ee, 1))

    return sum_ee

def perf_eval(vaild_csi, model,var_noise,K,Pc):
    power_rate = model(vaild_csi, 1)
    power_ee = model(vaild_csi, 0)
    sum_rate = rate_perf_eval(vaild_csi, power_rate, var_noise, K)
    sum_ee = ee_perf_eval(vaild_csi, power_ee, var_noise, K,Pc)
    
    return (sum_rate, sum_ee)
    

