import os
import pdb
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

fsize = 16; lwidth = 2.5

ash = np.load('logs/2023-02-27-12:40:59.npy')
sh = np.load('logs/2023-02-27-12:40:47.npy')
pdf = PdfPages('figs/mnist.pdf')
plt.plot(sh[:, 1], sh[:, 0], label='SH', linewidth=lwidth)
plt.plot(ash[:, 1], ash[:, 0], label='ASH', linewidth=lwidth)
plt.ylabel('ASR', fontsize=fsize); plt.xlabel('Avg. Queries', fontsize=fsize)
plt.xticks(fontsize=fsize); plt.yticks(fontsize=fsize)
plt.grid(); plt.legend(fontsize=fsize); plt.tight_layout()
pdf.savefig()
plt.close(); pdf.close()

sh = np.load('logs/2023-02-27-21:49:41.npy')
ash = np.load('logs/2023-02-27-21:50:05.npy')
pdf = PdfPages('figs/mnist.pdf')
plt.plot(sh[:, 1], sh[:, 0], label='SH', linewidth=lwidth)
plt.plot(ash[:, 1], ash[:, 0], label='ASH', linewidth=lwidth)
plt.ylabel('ASR', fontsize=fsize); plt.xlabel('Avg. Queries', fontsize=fsize)
plt.xticks(fontsize=fsize); plt.yticks(fontsize=fsize)
plt.grid(); plt.legend(fontsize=fsize); plt.tight_layout()
pdf.savefig()
plt.close(); pdf.close()

clean_acc = 1.0
sh = np.load('logs/2023-03-04-15:02:44.npy')
sh_accuracy = ((1 - sh[:, 0]) * clean_acc * 1000) / 1000
ash = np.load('logs/2023-03-04-15:12:04.npy')
ash_accuracy = ((1 - ash[:, 0]) * clean_acc * 1000) / 1000
pdf = PdfPages('figs/mnist_robust.pdf')
plt.plot(sh[:, 1], sh_accuracy, label='SH', linewidth=lwidth)
plt.plot(ash[:, 1], ash_accuracy, label='ASH', linewidth=lwidth)
plt.ylabel('Accuracy', fontsize=fsize); plt.xlabel('Avg. Queries', fontsize=fsize)
plt.xlim(0, 500); plt.xticks(fontsize=fsize); plt.yticks(fontsize=fsize)
plt.grid(); plt.legend(fontsize=fsize); plt.tight_layout()
pdf.savefig()
plt.close(); pdf.close()