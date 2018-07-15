import os
import sys

for i in range(90):
    cmd = "python eval_cls.py --epoch " + str(i)
    print cmd
    os.system(cmd)