
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import ScalarFormatter, FuncFormatter

def smooth_data(data, window_size):
    padding_size = window_size // 2
    padded_data = np.pad(data, (padding_size, padding_size), mode='edge')
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(padded_data, window, mode='same')[padding_size:-padding_size]
    return smoothed_data


# 'csv_files/PPO_28_1h.csv' one hot second experiment
# csv_path = ['csv_files/PPO_7.csv', 'csv_files/PPO_8.csv', 'csv_files/PPO_9.csv', 'csv_files/PPO_10.csv', 'csv_files/PPO_11.csv', 'csv_files/PPO_12.csv']
# csv_path = ['csv_files/PPO_10.csv', 'csv_files/PPO_15.csv', 'csv_files/PPO_28_1h.csv'] onehot encoding
# csv_path = ['csv_files/PPO_10.csv', 'csv_files/PPO_1.csv'] locations
csv_path = ['csv_files/PPO_10.csv', 'csv_files/PPO_18.csv', 'csv_files/PPO_22.csv'] #ratios
# csv_path = ['csv_files/PPO_30.csv', 'csv_files/PPO_28.csv', 'csv_files/PPO_29.csv', 'csv_files/PPO_24.csv', 'csv_files/PPO_25.csv', 'csv_files/PPO_26.csv']
# tags = ['basic state', 'with one hot encoding (statuses)', 'with one hot encoding (statuses & locations)']
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
    plt.figure(figsize=(10, 6))
    rs = 0
    for i in csv_path:
        df = pd.read_csv(i)
        smoothed_values = smooth_data(df['Value'], window_size)
        # sns.lineplot(x=df['Step'], y=df['Value'], color='#eaeaea')
        sns.lineplot(x=df['Step'], y=smoothed_values, label=f'{tags[rs]}')
        rs += 1
        # Set labels and title
        plt.ylim(50, 270)
        plt.xlim(500000, 110000000)
        plt.xlabel('Steps')
        plt.ylabel('Average episodic reward')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(CustomScalarFormatter())

    plt.legend()
    plt.show()

read_and_plot(100, csv_path)


