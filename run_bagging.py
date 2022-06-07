import sys
import os
import conf

for ii in range(1,conf.BAGGING_COUNT+1):
 in_file = sys.argv[1]
 out_file = ("train_%d.csv" % ii)
 os.system("python3 bagging.py " + in_file + " " + out_file)

