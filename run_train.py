import os
import conf

for ii in range(1,conf.BAGGING_COUNT+1):
 train_file = ("train_%d.csv" % ii)
 out_pth = ("model_%d.pth" % ii)
 os.system("python3 test.py " + train_file + " " + out_pth)
