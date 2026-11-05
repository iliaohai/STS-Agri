import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 3))
plt.subplots_adjust(left=0.08, bottom=0.15, right=0.9, top=0.95, hspace=0.23)
ax1 = plt.gca()
ax1.tick_params(width=0.5, length=2)
ax1.spines['right'].set_linewidth(0.5)
ax1.spines['top'].set_linewidth(0.5)
ax1.spines['left'].set_linewidth(0.5)
ax1.spines['bottom'].set_linewidth(0.5)

# ax1.grid(color='#8f8f8f', linestyle='--', linewidth=0.3)
ax1.set_xlim(0, 6)
ax1.set_ylim(0, 20)

plt.xticks([1, 2, 3, 4, 5],
           ('0.5', '1.0', '1.5', '2.0', '2.5'), fontsize=8)
plt.yticks([0, 4, 8, 12, 16, 20], ('0', '4', '8', '12', '16', '20'), fontsize=8)
ax1.set_xlabel('Grid size (m)', fontsize=8)
ax1.set_ylabel('Error (m)', fontsize=8, labelpad=0)

X = np.arange(len(avg_errors)) + 1
bar1 = ax1.bar(X - 0.24, avg_errors, width=0.24, color='#F7DA8E', edgecolor="k", label='Vocabulary(K)', linewidth=0.5)
bar2 = ax1.bar(X, max_errors, width=0.24, color='#F7F6BE', edgecolor="k", label='Params(M)', linewidth=0.5)

ax2 = plt.twinx()
ax2.tick_params(width=0.5, length=2, bottom=False)
ax2.set_ylim(0, 5)
ax2.set_ylabel('Time (s)', fontsize=8, labelpad=8)
ax2.set_yticks([0, 1, 2, 3, 4, 5])
ax2.set_yticklabels(('0', '1', '2', '3', '4', '5'), fontsize=8)
bar3 = ax2.bar(X + 0.24, run_time, width=0.24, color='#4873B8', edgecolor="k", label='F1', linewidth=0.5)
# bar2 = ax2.plot(X, run_time, color='#4873B8', label='Run times', marker='.', linewidth=0.8)
ax2.legend([bar1, bar2, bar3], ["Mean error", "Max error", "Run time"], fontsize=8)

plt.savefig("grid.png", dpi=800)
plt.show()
