import numpy as np
import random as rd
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

initpolicy = []
ND = []
NDa = []
perf = []

# file1 = 'EnsembleQ/temp-folder/initialized-actor-1.txt'
# file1 = 'EnsembleQ/5-links/arrival-2-qos-1/initialized-actor-1.txt'
# file1 = 'EnsembleQ/objective-with-penalty/5-links/initialized-actor-4.txt'
file1 = 'EnsembleQ/maxmin-ratio/2-links/arrival-2-qos-1/initialized-actor-6.txt'
p1 = np.loadtxt(file1)
p1 = np.mean(p1[-100:])

# # file2 = 'EnsembleQ/temp-folder/ND-f-1-adjust-0.txt'
# file2 = 'EnsembleQ/2-links/arrival-2-qos-1/ND-f-1-adjust-0.txt'
# p2 = np.loadtxt(file2)
# p2 = np.mean(p2[-100:])

# # file3 = 'EnsembleQ/temp-folder/ND-f-1-adjust-1.txt'
# file3 = 'EnsembleQ/2-links/arrival-2-qos-1/ND-f-1-adjust-1.txt'
# p3 = np.loadtxt(file3)
# p3 = np.mean(p3[-100:])

for i in range(25):
    # file = 'EnsembleQ/5-links/arrival-2-qos-1/actor-r-3-gamma-0.99-init-1-eval-'+str((i+1)*2000)+'-1.txt'
    # file = 'EnsembleQ/objective-with-penalty/5-links/actor-r-10-gamma-0.99-init-1-eval-'+str((i+1)*2000)+'-4.txt'
    file = 'EnsembleQ/maxmin-ratio/2-links/arrival-2-qos-1/actor-r-13-gamma-0.99-init-1-eval-'+str((i+1)*2000)+'-6.txt'
    p = np.loadtxt(file)
    p = p[-100:]
    perf.append(np.mean(p))
    initpolicy.append(p1)
    # ND.append(p2)
    # NDa.append(p3)
# perf = perf[:20]
# initpolicy = initpolicy[:20]
plt.plot(perf, color = 'red', label = 'RL', linewidth = 2)
# plt.plot(ND, color = 'black', label = 'ND', linewidth = 2)
# plt.plot(NDa, color = 'blue', label = 'NDa', linewidth = 2)
plt.plot(initpolicy, color = 'orange', label = 'Initial policy(ND)', linewidth = 2)

plt.legend(loc='best', prop=font_manager.FontProperties(size=12))
plt.tick_params(labelsize=12)
plt.xlabel('training steps(x2000)', size = 12)
plt.ylabel('performance', size = 12)
plt.show()
