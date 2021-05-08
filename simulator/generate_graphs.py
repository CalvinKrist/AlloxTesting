import numpy as np
import matplotlib.pyplot as plt

x_alex=[7438.5945,7438.5945,7438.5945,7438.5945,7438.5945,7438.5945,7438.5945]
x_alex_high=[7515.637,8076.256,7779.649,8389.779,8051.057,7752.327,8127.235]
x_alex_low=[7361.552,6839.999,7197.504,6807.414,6893.424,6990.27,6689.054]
alex_errors = [x_alex_high[k]-x_alex_low[k] for k in range(len(x_alex)) ]

x_lenet=[1082.3471,1108.4605,1080.174,1084.6546,1081.0413,1088.0304,1090.4647]
x_lenet_high=[1207.612,1255.99,1191.454,1124.783,1271.569,1188.743,1227.64]
x_lenet_low=[978.645,973.859,922.636,1010.145,920.081,975.166,952.089]
lenet_errors = [x_lenet_high[k]-x_lenet_low[k] for k in range(len(x_lenet)) ]

x_dist=[113.6119,113.6119,113.392,113.6119,113.6119,113.6119,113.6119,113.6119,113.6119]
x_dist_high=[117.977,118.165,117.753,118.905,118.528,119.933,118.313,119.011,118.215]
x_dist_low=[106.587,106.136,106.652,107.273,106.941,107.039,106.972,106.539,105.511]
dist_errors = [x_dist_high[k]-x_dist_low[k] for k in range(len(x_dist)) ]

#Names for AlexNet,LeNet

names = ["Baseline","LinReg_0","LinReg_1","LinReg_2","Detailed_0","Detailed_1","Detailed_2"]
"""

names = ["\u03BC=0, \u03C3=0.01",
         "\u03BC=0, \u03C3=0.05","\u03BC=0, \u03C3=0.1,","\u03BC=0.05, \u03C3=0.01","\u03BC=0.05, \u03C3=0.05","\u03BC=0.05, \u03C3=0.1","\u03BC=-0.05, \u03C3=0.01","\u03BC=-0.05, \u03C3=0.05","\u03BC=-0.05, \u03C3=0.1"]
"""
x = range(len(x_lenet))
fig, ax = plt.subplots(figsize=(len(names)-2, 3))
for i,method in enumerate(names):
    ax.errorbar(i, x_lenet[i] , xerr=0, yerr=lenet_errors[i], ls='none',marker='o',label=method)
ax.set_xlabel("Estimation Configurations")
ax.set_title("LeNet Simulation Completion Times")
ax.set_ylabel("Average Job Completion Time")

# Area plot
fig.tight_layout()
plt.legend(bbox_to_anchor=(0.05, -.85),loc='lower left',ncol=3,labelspacing=0.05)
fig.subplots_adjust(bottom=0.4)
plt.show()