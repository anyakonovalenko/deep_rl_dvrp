import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import ScalarFormatter, FuncFormatter


def smooth_data(data, window_size):
    """Apply moving average smoothing to data."""
    padding_size = window_size // 2
    padded_data = np.pad(data, (padding_size, padding_size), mode='edge')
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(padded_data, window, mode='same')[padding_size:-padding_size]
    return smoothed_data


# Configuration examples for comparing different state-space components:
# One-hot encoding comparison:
#   csv_path = ['results/csv/PPO_10.csv', 'results/csv/PPO_15.csv', 'results/csv/PPO_28_1h.csv']
#   tags = ['basic state', 'with one-hot encoding', 'with one-hot + locations']
#
# Location features comparison:
#   csv_path = ['results/csv/PPO_10.csv', 'results/csv/PPO_1.csv']
#
# Current configuration - ratios comparison:
csv_path = ['results/csv/PPO_10.csv', 'results/csv/PPO_18.csv', 'results/csv/PPO_22.csv']
tags = ['basic state', 'with reward ratio', 'with capacity ratio']
class CustomScalarFormatter(ScalarFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_scientific(True)
        self.set_powerlimits((-3, 3))

    def _set_format(self):
        super()._set_format()
        self.format = r'%1.1f'

    def get_offset(self):
        return r'1$\times 10^{8}$'


matplotlib.use('TkAgg')

def read_and_plot(window_size, csv_path):
    """Plot smoothed learning curves from TensorBoard CSV exports."""
    plt.figure(figsize=(10, 6))
    rs = 0
    for i in csv_path:
        df = pd.read_csv(i)
        smoothed_values = smooth_data(df['Value'], window_size)
        sns.lineplot(x=df['Step'], y=smoothed_values, label=f'{tags[rs]}')
        rs += 1

    plt.ylim(50, 270)
    plt.xlim(500000, 110000000)
    plt.xlabel('Steps')
    plt.ylabel('Average episodic reward')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(CustomScalarFormatter())

    plt.legend()
    plt.show()


read_and_plot(100, csv_path)
