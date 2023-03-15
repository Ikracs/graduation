import os
import pdb
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

fsize = 16; lwidth = 2.5

sh = np.load('logs/2023-03-04-23:21:12.npy')
bps = np.load('logs/2023-03-05-13:53:58.npy')
ghs = np.load('logs/2023-03-05-10:49:52.npy')
ash = np.load('logs/2023-03-04-23:21:20.npy')
pdf = PdfPages('figs/ablation_resnet.pdf')
plt.plot(sh[:, 1], sh[:, 0], label='SH', linewidth=lwidth)
plt.plot(bps[:, 1], bps[:, 0], label='SH + BPS', linewidth=lwidth)
plt.plot(ghs[:, 1], ghs[:, 0], label='SH + GHS', linewidth=lwidth)
plt.plot(ash[:, 1], ash[:, 0], label='ASH', linewidth=lwidth)
plt.ylabel('ASR', fontsize=fsize); plt.xlabel('Avg. Queries', fontsize=fsize)
plt.xticks(fontsize=fsize); plt.yticks(fontsize=fsize)
plt.grid(); plt.legend(fontsize=fsize); plt.tight_layout()
pdf.savefig()
plt.close(); pdf.close()

sh = np.load('logs/2023-03-05-01:07:55.npy')
bps = np.load('logs/2023-03-05-14:31:31.npy')
ghs = np.load('logs/2023-03-05-11:28:55.npy')
ash = np.load('logs/2023-03-04-23:56:06.npy')
pdf = PdfPages('figs/ablation_inception.pdf')
plt.plot(sh[:, 1], sh[:, 0], label='SH', linewidth=lwidth)
plt.plot(bps[:, 1], bps[:, 0], label='SH + BPS', linewidth=lwidth)
plt.plot(ghs[:, 1], ghs[:, 0], label='SH + GHS', linewidth=lwidth)
plt.plot(ash[:, 1], ash[:, 0], label='ASH', linewidth=lwidth)
plt.ylabel('ASR', fontsize=fsize); plt.xlabel('Avg. Queries', fontsize=fsize)
plt.xticks(fontsize=fsize); plt.yticks(fontsize=fsize)
plt.grid(); plt.legend(fontsize=fsize); plt.tight_layout()
pdf.savefig()
plt.close(); pdf.close()

sh = np.load('logs/2023-03-05-00:29:22.npy')
bps = np.load('logs/2023-03-05-14:22:00.npy')
ghs = np.load('logs/2023-03-05-11:18:26.npy')
ash = np.load('logs/2023-03-04-23:46:31.npy')
pdf = PdfPages('figs/ablation_vgg.pdf')
plt.plot(sh[:, 1], sh[:, 0], label='SH', linewidth=lwidth)
plt.plot(bps[:, 1], bps[:, 0], label='SH + BPS', linewidth=lwidth)
plt.plot(ghs[:, 1], ghs[:, 0], label='SH + GHS', linewidth=lwidth)
plt.plot(ash[:, 1], ash[:, 0], label='ASH', linewidth=lwidth)
plt.ylabel('ASR', fontsize=fsize); plt.xlabel('Avg. Queries', fontsize=fsize)
plt.xticks(fontsize=fsize); plt.yticks(fontsize=fsize)
plt.grid(); plt.legend(fontsize=fsize); plt.tight_layout()
pdf.savefig()
plt.close(); pdf.close()