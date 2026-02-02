import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for SCI paper
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 1.5

# File paths
base_dir = "/devdata/yaojinyu/Chat4Seniors/out/chat4seniors_rlvr_model"
files = [
    {
        "path": os.path.join(base_dir, "run-.-tag-actor_entropy.csv"),
        "title": "Actor Entropy",
        "ylabel": "Entropy",
        "color": "#1f77b4" # Blue
    },
    {
        "path": os.path.join(base_dir, "run-.-tag-critic_rewards_mean.csv"),
        "title": "Critic Rewards Mean",
        "ylabel": "Reward",
        "color": "#ff7f0e" # Orange
    },
    {
        "path": os.path.join(base_dir, "run-.-tag-val-core_dpo_trainset_reward_mean@1.csv"),
        "title": "Validation Reward Mean@1",
        "ylabel": "Mean Reward",
        "color": "#2ca02c" # Green
    }
]

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, config in enumerate(files):
    ax = axes[i]
    if os.path.exists(config["path"]):
        try:
            df = pd.read_csv(config["path"])
            
            # Plot
            sns.lineplot(data=df, x='Step', y='Value', ax=ax, color=config["color"], linewidth=1.5, alpha=0.9)
            
            # Optional: Add rolling mean for noisy data (except validation which is sparse)
            if i != 2 and len(df) > 50:
                df['Smoothed'] = df['Value'].rolling(window=10).mean()
                sns.lineplot(data=df, x='Step', y='Smoothed', ax=ax, color='black', linewidth=1, linestyle='--', alpha=0.6, label='Smoothed (MA10)')
                ax.legend()
            
            # Formatting
            ax.set_title(config["title"], fontweight='bold', pad=10)
            ax.set_xlabel("Training Steps")
            ax.set_ylabel(config["ylabel"])
            
            # Add a slight transparency fill below the line
            ax.fill_between(df['Step'], df['Value'], alpha=0.1, color=config["color"])
            
            # Spines adjustment for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        except Exception as e:
            print(f"Error reading {config['path']}: {e}")
            ax.text(0.5, 0.5, 'Error reading data', ha='center', va='center')
    else:
        print(f"File not found: {config['path']}")
        ax.text(0.5, 0.5, 'File not found', ha='center', va='center')

plt.tight_layout()
output_path = "/devdata/yaojinyu/Chat4Seniors/paper/training_process.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")
