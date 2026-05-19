import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from data.dataset_config import dataset_config
from util.directory_manager import DirectoryManager


class Logger:
    def __init__(self, dataset_names):
        self.exp_dir = Path(DirectoryManager().exp_dir)
        self.dataset_names = dataset_names
        print('=' * 100)

    def _load_data(self, folder_path):
        labels_data = np.load(f'{folder_path}/labels.npz')
        preds_data = np.load(f'{folder_path}/preds.npz')
        labels = labels_data[labels_data.files[0]]
        preds = preds_data[preds_data.files[0]]
        return labels, preds

    def _compute_metrics(self, labels, preds):
        return {
            'accuracy': accuracy_score(labels, preds),
            'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
            'recall_macro': recall_score(labels, preds, average='macro', zero_division=0),
            'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
            'precision_micro': precision_score(labels, preds, average='micro', zero_division=0),
            'recall_micro': recall_score(labels, preds, average='micro', zero_division=0),
            'f1_micro': f1_score(labels, preds, average='micro', zero_division=0),
            'classification_report': classification_report(labels, preds, zero_division=0),
        }

    def _save_report(self, metrics, folder_name, path):
        path.mkdir(parents=True, exist_ok=True)
        report_path = path / f'report_{folder_name}.txt'
        
        with report_path.open('w') as f:
            f.write('Metrics Report\n')
            f.write('=========================\n')
            for key, value in metrics.items():
                if key == 'classification_report':
                    f.write(f'\n{key}:\n{value}\n')
                else:
                    f.write(f'{key}: {value}\n')
        print(f'Report saved => {report_path}')

    def _generate_confusion_matrices(self, labels, preds, folder_name, path, dataset_name):
        path.mkdir(parents=True, exist_ok=True)
        
        cm = confusion_matrix(labels, preds)
        np.savetxt(path / f'confusion_matrix_{folder_name}.csv', cm, delimiter=',', fmt='%d')
        cm_norm = confusion_matrix(labels, preds, normalize='true')

        dc = dataset_config[dataset_name]
        label_column = dc.get('label_column', 'label').lower()
        label_conv_path = Path(dc['path']).parent / f'{label_column}_conv.json'
        with label_conv_path.open('r') as f:
            label_conv = json.load(f)

        classes = label_conv.keys()
        n_classes = len(classes)

        plt.figure(figsize=(max(6, n_classes * 0.8), max(4, n_classes * 0.6))) # Dynamic fig size
        sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='viridis')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Normalized Confusion Matrix')
        plt.savefig(path / f'confusion_matrix_{folder_name}.pdf', bbox_inches='tight')
        plt.close()

    def _save_aggregated_report(self, episode_metrics, path):
        """
        Save mean ± std of scalar metrics across episodes.
        """
        path.mkdir(parents=True, exist_ok=True)
        scalar_keys = [k for k in episode_metrics[0] if k != 'classification_report']
        n = len(episode_metrics)

        report_path = path / 'report_test_aggregated.txt'
        with report_path.open('w') as f:
            f.write(f'Aggregated Metrics Report ({n} episodes)\n')
            f.write('=========================\n')
            for key in scalar_keys:
                vals = [m[key] for m in episode_metrics]
                f.write(f'{key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}\n')

            f.write('\n--- Per-episode breakdown ---\n')
            for i, m in enumerate(episode_metrics):
                f.write(f'\nEpisode {i}:\n')
                for key in scalar_keys:
                    f.write(f'  {key}: {m[key]:.4f}\n')

        print(f'Aggregated report saved => {report_path}')


    def _generate_mean_confusion_matrix(self, episode_cms, path, dataset_name):
        """
        Generate and save a mean normalized confusion matrix across episodes.
        """
        path.mkdir(parents=True, exist_ok=True)

        mean_cm = np.mean(np.stack(episode_cms), axis=0)
        np.savetxt(path / 'confusion_matrix_test_mean.csv', mean_cm, delimiter=',', fmt='%.4f')

        dc = dataset_config[dataset_name]
        label_column = dc.get('label_column', 'label').lower()
        label_conv_path = Path(dc['path']).parent / f'{label_column}_conv.json'
        with label_conv_path.open('r') as f:
            label_conv = json.load(f)

        classes = label_conv.keys()
        n_classes = len(classes)

        plt.figure(figsize=(max(6, n_classes * 0.8), max(4, n_classes * 0.6)))
        sns.heatmap(mean_cm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='viridis')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(f'Mean Normalized Confusion Matrix ({len(episode_cms)} episodes)')
        plt.savefig(path / 'confusion_matrix_test_mean.pdf', bbox_inches='tight')
        plt.close()

    def process_results(self):
        """
        Process results for each dataset used.
        - Loads labels and predictions.
        - Computes metrics and saves them to a text file.
        - Generates confusion matrix CSV and a normalized confusion matrix plot.
        """
        for dataset_name in self.dataset_names:
            base_path = self.exp_dir / dataset_name
            if not base_path.exists():
                continue

            # Single episode metrics (train/val/test)
            for phase in ['train', 'val', 'test']:
                path = base_path / phase
                if not path.exists():
                    continue

                labels, preds = self._load_data(path)
                metrics = self._compute_metrics(labels, preds)
                self._save_report(metrics, folder_name=phase, path=base_path / 'detailed')
                self._generate_confusion_matrices(
                    labels, preds, folder_name=phase, path=base_path / 'detailed', 
                    dataset_name=dataset_name
                )
            
            test_episode_dirs = sorted(
                (p for p in base_path.glob('test_*') if p.is_dir()),
                key=lambda p: int(p.name.split('_')[1])
            )
            if not test_episode_dirs:
                continue

            # Multiple episode metrics (test episodes)
            episode_metrics = []
            episode_cms = []

            for ep_dir in test_episode_dirs:
                ep_idx = ep_dir.name
                labels, preds = self._load_data(ep_dir)
                metrics = self._compute_metrics(labels, preds)
                cm = confusion_matrix(labels, preds, normalize='true')
                episode_metrics.append(metrics)
                episode_cms.append(cm)

            # Aggregate metrics across episodes (mean ± std)
            self._save_aggregated_report(episode_metrics, path=base_path / 'detailed')
            # Mean confusion matrix across episodes
            self._generate_mean_confusion_matrix(
                episode_cms, path=base_path / 'detailed', dataset_name=dataset_name
            )
                
    def _split_into_episodes(self, df):
        """Split dataframe into episodes based on epoch resets."""
        episodes = []
        current = []
        prev_epoch = None

        for _, row in df.iterrows():
            if prev_epoch is not None and row['epoch'] <= prev_epoch:
                episodes.append(pd.DataFrame(current).reset_index(drop=True))
                current = []
            current.append(row)
            prev_epoch = row['epoch']

        if current:
            episodes.append(pd.DataFrame(current).reset_index(drop=True))

        return episodes

    def _plot_metric_group(self, df, group_cols, ylabel, path, output_filename):
        episodes = self._split_into_episodes(df)
        n_episodes = len(episodes)
        max_epoch = max(ep['epoch'].max() for ep in episodes)

        plt.figure(figsize=(max_epoch * 0.20 + 2, 5))

        colors = [p['color'] for p in plt.rcParams['axes.prop_cycle']]

        if n_episodes == 1:
            ep = episodes[0]
            for i, col in enumerate(group_cols):
                plt.plot(ep['epoch'], ep[col], marker='o', label=col,
                        color=colors[i % len(colors)])
        else:
            for i, col in enumerate(group_cols):
                col_color = colors[i % len(colors)]

                # Thin transparent lines for each individual episode
                for ep in episodes:
                    plt.plot(ep['epoch'], ep[col],
                            color=col_color, alpha=0.25, linewidth=1,
                            marker='o', markersize=2,
                            label=None)

                # Bold mean line across episodes (only if all have the same length)
                ep_lengths = [len(ep) for ep in episodes]
                if len(set(ep_lengths)) == 1:
                    mean_vals = np.mean(
                        np.stack([ep[col].values for ep in episodes]), axis=0
                    )
                    base_epochs = episodes[0]['epoch'].values
                    plt.plot(base_epochs, mean_vals,
                            color=col_color, linewidth=2.5,
                            marker='o', markersize=4,
                            label=f'{col} (mean, {n_episodes} ep.)')
                else:
                    # Episodes have unequal lengths: skip mean, just add a legend entry
                    plt.plot([], [], color=col_color, linewidth=2,
                            label=f'{col} ({n_episodes} ep.)')

        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.grid(linestyle='--', color='gray')
        plt.xticks(np.arange(1, max_epoch + 1, 1), fontsize=8, rotation=90)
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.savefig(path / output_filename, bbox_inches='tight')
        plt.close()

    def plot_per_epoch_metrics(self, filename='epoch_metrics'):
        """
        Plot per-epoch metrics for each dataset used.
        The data should be stored in a Parquet file named `filename` (default: 'epoch_metrics').
        """
        for dataset_name in self.dataset_names:
            base_path = self.exp_dir / dataset_name
            if not base_path.exists():
                continue
        
            file_path = Path(base_path) / f'{filename}.csv'
            if not file_path.exists():
                continue
            
            df = pd.read_csv(file_path)

            metrics_columns = [col for col in df.columns if col != 'epoch']
            loss_cols = [col for col in metrics_columns if 'loss' in col.lower()]
            other_cols = [col for col in metrics_columns if col not in loss_cols]

            self._plot_metric_group(
                df, loss_cols, ylabel='Loss Value', 
                path=base_path / 'detailed', output_filename='epoch_loss.pdf'
            )
            self._plot_metric_group(
                df, other_cols, ylabel='Metric Value', 
                path=base_path / 'detailed', output_filename='epoch_metrics.pdf'
            )
