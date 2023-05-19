import json
import math
import os
import numpy as np
from scipy.interpolate import CubicSpline, interp1d, make_interp_spline, UnivariateSpline, splev, splrep

import matplotlib.pyplot as plt

path = r'C:\Users\ledro\StageE23\Data\Ete_2022\Participant06\autocorrection\Prise02'

jsonfile = open(path+'/Positions/positions_corrigees.json')
dict_coordo_labels = json.load(jsonfile)

labels = list(dict_coordo_labels['image1'].keys())

x_axis = range(1, len(list(dict_coordo_labels.keys()))+1)
x, y = {}, {}
cs, spl, s1d, interp = {}, {}, {}, {}
for l in labels:
    y.update({l : [[], []]})
    x.update({l: []})
    cs.update({l: [[], []]})
    spl.update({l: [[], []]})
    s1d.update({l: [[], []]})
    interp.update({l: [[], []]})
for im, coordos in dict_coordo_labels.items():
    for l, c in coordos.items():
        if not np.isnan(c[0]):
            y[l][0].append(c[0])
            y[l][1].append(c[1])
            x[l].append(int(im[5:]))

for l in labels:
    m = len(x[l])
    sm = m-math.sqrt(2*m)
    print(sm)
    std_0 = np.std(y[l][0])
    std_1 = np.std(y[l][1])
    print(std_0, std_1)
    w = np.ones(m) #poids de 1 à tous les points
    w[0] = 5
    w[-1] = 5
    cs[l][0] = CubicSpline(x[l], y[l][0], bc_type='clamped')
    cs[l][1] = CubicSpline(x[l], y[l][1], bc_type='clamped')
    s1d[l][0] = splrep(x[l], y[l][0], w, k=3, s=sm/3)
    s1d[l][1] = splrep(x[l], y[l][1], w, k=3, s=sm/3)
    interp[l][0] = interp1d(x[l], y[l][0], 'cubic')
    interp[l][1] = interp1d(x[l], y[l][1], 'cubic')
    #mss[l][0] = make_smoothing_spline(x[l], y[l][0])

fig, ax = plt.subplots()
for l in labels:
    ax.scatter(x[l], y[l][0], label=f'{l} data')
    ax.plot(x_axis, cs[l][0](x_axis), label=f'{l} cubic')
    ax.plot(x_axis, splev(x_axis, s1d[l][0]), label=f'{l}spl')
    #ax.plot(x_axis, interp[l][0](x_axis), label=f'{l}interp')
ax.legend()
plt.show()

fig, ax = plt.subplots()
for l in labels:
    ax.scatter(x[l], y[l][1], label=f'{l} data')
    ax.plot(x_axis, cs[l][1](x_axis), label=f'{l} cubic')
    ax.plot(x_axis, splev(x_axis, s1d[l][1]), label=f'{l}spl')
    #ax.plot(x_axis, interp[l][1](x_axis), label=f'{l}interp')
ax.legend()
plt.show()

def interpolate_spline(dict_coordo_labels, labels):
    x, y = {}, {}
    splines, spl = {}, {}
    
    for l in labels:
        y.update({l : [[], []]})
        x.update({l: []})
        splines.update({l: [[], []]})
        spl.update({l: [[], []]})
    for im, coordos in dict_coordo_labels.items():
        for l, c in coordos.items():
            if not math.isnan(c[0]):
                y[l][0].append(c[0])
                y[l][1].append(c[1])
                x[l].append(im[5:])

    for l in labels:
        m = len(x[l])
        sm = m-math.sqrt(2*m)
        w = np.ones(m) #poids de 1 à tous les points
        w[0] = 5
        w[-1] = 5
        cs_x = CubicSpline(x[l], y[l][0])
        cs_y = CubicSpline(x[l], y[l][1])
        splines[l][0] = cs_x
        splines[l][1] = cs_y
        
    return splines

'''
def smooth_splines(splines):
    splines_smooth = {}
    if m > 5:
        smooth_x = splrep(x[l], y[l][0], w, k=5, s=sm)
        smooth_y = splrep(x[l], y[l][1], w, k=5, s=sm)
        spl[l][0] = smooth_x
        spl[l][1] = smooth_y
        splines_smooth[l][0] = splev(x_axis, spl[l][0])
        splines_smooth[l][1] = splev(x_axis, spl[l][1])
'''

def graph(splines, splines_smooth):
    x_axis = np.arange(len(list(dict_coordo_labels.keys())))
    if not 'interpolate' in os.listdir(path):
        os.mkdir(path+ '/interpolate/')
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.scatter(x_axis, splines['G'][0](x_axis), s=5, label=f'G x')
    ax1.plot(x_axis, splines_smooth['G'][0], label=f'G x smooth')
    ax1.legend()
    ax2.scatter(x_axis, splines['D'][0](x_axis), s=5, label=f'D x')
    ax2.plot(x_axis, splines_smooth['D'][0], label=f'D x smooth')
    ax2.legend()
    for l in ['C', 'T', 'L']:
        ax3.scatter(x_axis, splines[l][0](x_axis), s=5, label=f'{l} x')
        ax3.plot(x_axis, splines_smooth[l][0], label=f'{l} x smooth')
    ax3.legend()
    plt.show()
    plt.close() 

    fig, (ax4, ax5, ax6) = plt.subplots(1,3)
    ax4.scatter(x_axis, splines['C'][0](x_axis), s=5, label=f'C x')
    ax4.plot(x_axis, splines_smooth['C'][0], label=f'C Y smooth')
    ax4.legend()
    ax5.scatter(x_axis, splines['L'][0](x_axis), s=5, label=f'L x')
    ax5.plot(x_axis, splines_smooth['L'][0], label=f'L Y smooth')
    ax5.legend()
    for l in ['G', 'D', 'C']:
        ax6.scatter(x_axis, splines[l][0](x_axis), s=5, label=f'{l} Y')
        ax6.plot(x_axis, splines_smooth[l][0], label=f'{l} Y smooth')
    ax6.legend()
    plt.show()
    plt.close()