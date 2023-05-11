#!/usr/bin/env python
# coding: utf-8

from types import CodeType
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


import matplotlib
from numpy.lib.function_base import cov
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
figsz = {
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (6, 3),
}
plt.rcParams.update(figsz)
hb = '\\\\//\\\\//'

color_def = [
    '#f4b183',
    '#ffd966',
    '#c5e0b4',
    '#bdd7ee',
    "#8dd3c7",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#cccccc",
    "#fccde5",
    "#b3de69",
    "#ffd92f",
    '#fc8d59',
    '#74a9cf',
    '#66c2a4',
    '#f4a143',
    '#ffc936',
    '#78c679',
]

hatch_def = [
    '//',
    '\\\\',
    'xx',
    '++',
    '--',
    '||',
    '..',
    'oo',
    '',
]

marker_def = [
    'o',
    'D',
    'H',
    'v',
    '*',
    '+',
    '^',
    'x',
]

ls_def = [
    '--',
    '-.',
    ':',
    'densely dashed'
    '-',
]
