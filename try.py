import os
import pandas as pd
import numpy as np

# a=np.array([1,2,3,4,5,6,7,8,9,0,11,12])
# print(a)
# print(a.shape)
# a1=a.reshape(-1,6).T
# print(a1)
# print(a1.shape)

a = np.arange(1, 85)
print(a)
print(a.shape)
N=68
print(a[-N:])
print(a[-N:].shape)
print(a[:-N])
print(a[:-N].shape)