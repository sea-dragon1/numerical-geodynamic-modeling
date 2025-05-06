# 2025-03-26 Hailong Liu
# 已知左右两侧进入速度，与流出速度

import numpy as np

vin_left = 0.08
vin_right = 0.08


hin_left = 120.0
hin_right = 80.0
hout_left = 540.0
hout_right = 580.0

vout = (vin_left * hin_left + vin_right * hin_right) / (hout_left + hout_right)
print(vout)