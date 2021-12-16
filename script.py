import os

for i in range(25):
    file = str((i+1)*2000)
    print(file)
    os.system('python Trained-model.py --model actor-r-14-gamma-0.99-init-1-eval-%s --findex 6 > tm.log 2>&1 &' % (file))
