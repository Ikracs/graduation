import os
import pdb
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

fsize = 16; lwidth = 2.5

ash = np.load('logs/2023-02-26-23:26:07.npy')
sh = np.load('logs/2023-02-27-00:31:27.npy')
pdf = PdfPages('figs/cifar10.pdf')
plt.plot(sh[:, 1], sh[:, 0], label='SH', linewidth=lwidth)
plt.plot(ash[:, 1], ash[:, 0], label='ASH', linewidth=lwidth)
plt.ylabel('ASR', fontsize=fsize); plt.xlabel('Avg. Queries', fontsize=fsize)
plt.xticks(fontsize=fsize); plt.yticks(fontsize=fsize)
plt.grid(); plt.legend(fontsize=fsize); plt.tight_layout()
pdf.savefig()
plt.close(); pdf.close()

ash = np.load('logs/2023-02-27-21:51:17.npy')
sh = np.load('logs/2023-02-28-00:52:22.npy')
pdf = PdfPages('figs/cifar10_t.pdf')
plt.plot(sh[:, 1], sh[:, 0], label='SH', linewidth=lwidth)
plt.plot(ash[:, 1], ash[:, 0], label='ASH', linewidth=lwidth)
plt.ylabel('ASR', fontsize=fsize); plt.xlabel('Avg. Queries', fontsize=fsize)
plt.xticks(fontsize=fsize); plt.yticks(fontsize=fsize)
plt.grid(); plt.legend(fontsize=fsize); plt.tight_layout()
pdf.savefig()
plt.close(); pdf.close()

clean_acc = 1.0
ash = np.load('logs/2023-02-27-21:51:17.npy')
sh_accuracy = ((1 - sh[:, 0]) * clean_acc * 1000) / 1000
sh = np.load('logs/2023-02-28-00:52:22.npy')
ash_accuracy = ((1 - ash[:, 0]) * clean_acc * 1000) / 1000
pdf = PdfPages('figs/cifar10_robust.pdf')
plt.plot(sh[:, 1], sh[:, 0], label='SH', linewidth=lwidth)
plt.plot(ash[:, 1], ash[:, 0], label='ASH', linewidth=lwidth)
plt.ylabel('Accuracy', fontsize=fsize); plt.xlabel('Avg. Queries', fontsize=fsize)
plt.xticks(fontsize=fsize); plt.yticks(fontsize=fsize)
plt.grid(); plt.legend(fontsize=fsize); plt.tight_layout()
pdf.savefig()
plt.close(); pdf.close()
