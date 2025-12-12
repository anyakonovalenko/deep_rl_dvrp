from packaging import version
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
import numpy as np

csv_path = 'csv_files/PPO_7.csv'
matplotlib.use('TkAgg')
df = pd.read_csv(csv_path)
filtered_df = df.loc[(df['run'] == 'PPO_7') & (df['tag'].str.contains('rollout/ep_rew_mean'))]

sns.lineplot(x='step', y='value', data=filtered_df)

# Set the labels and title for the plot
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('PPO_7 - rollout/ep_len_mean')
plt.show()