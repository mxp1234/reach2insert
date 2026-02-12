#!/usr/bin/env python3
"""Visualize training metrics from JSONL file.

Separates learner metrics (x-axis: learner_step) and actor metrics (x-axis: env_steps).
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse


def load_metrics(filepath, start_line=0):
    """Load metrics from JSONL file starting from a specific line.

    Returns:
        learner_metrics: dict with learner_step as x-axis
        actor_metrics: dict with env_steps as x-axis
    """
    learner_metrics = defaultdict(list)
    actor_metrics = defaultdict(list)

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            try:
                data = json.loads(line.strip())

                # Get step values
                learner_step = data.get('learner_step', data.get('step', 0))
                env_steps = data.get('env_steps', data.get('progress/total_steps', learner_step))

                # ========== Actor metrics (use env_steps) ==========
                if 'environment/episode' in data:
                    ep = data['environment/episode']
                    if 'length' in ep:
                        actor_metrics['episode_length'].append((env_steps, ep['length']))
                    if 'intervention_rate' in ep:
                        actor_metrics['intervention_rate'].append((env_steps, ep['intervention_rate']))

                # ========== Learner metrics (use learner_step) ==========
                # Critic metrics
                if 'critic/q_mean' in data:
                    learner_metrics['q_mean'].append((learner_step, data['critic/q_mean']))
                if 'critic/critic_loss' in data:
                    learner_metrics['critic_loss'].append((learner_step, data['critic/critic_loss']))
                if 'critic/disagreement_mean' in data:
                    learner_metrics['critic_disagreement'].append((learner_step, data['critic/disagreement_mean']))

                # Actor metrics
                if 'actor/actor_loss' in data:
                    learner_metrics['actor_loss'].append((learner_step, data['actor/actor_loss']))
                if 'actor/entropy' in data:
                    learner_metrics['entropy'].append((learner_step, data['actor/entropy']))
                if 'actor/temperature' in data:
                    learner_metrics['temperature'].append((learner_step, data['actor/temperature']))

                # Buffer metrics
                if 'buffer/demo_size' in data:
                    learner_metrics['demo_size'].append((learner_step, data['buffer/demo_size']))
                if 'buffer/online_size' in data:
                    learner_metrics['online_size'].append((learner_step, data['buffer/online_size']))

                # Sampler metrics
                if 'sampler/offline_ratio' in data:
                    learner_metrics['offline_ratio'].append((learner_step, data['sampler/offline_ratio']))

            except json.JSONDecodeError:
                continue

    return learner_metrics, actor_metrics


def smooth(values, window=5):
    """Apply moving average smoothing."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')


def plot_metrics(learner_metrics, actor_metrics, save_path=None):
    """Plot key training metrics with separate x-axes."""
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle('SERL Training Metrics', fontsize=14)

    # ========== Row 1: Actor metrics (env_steps) ==========
    # 1. Episode Length
    ax = axes[0, 0]
    if actor_metrics['episode_length']:
        steps, values = zip(*actor_metrics['episode_length'])
        ax.scatter(steps, values, alpha=0.5, s=10, c='blue')
        if len(values) > 5:
            smoothed = smooth(values, 5)
            ax.plot(steps[2:-2], smoothed, 'r-', linewidth=2, label='smoothed')
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel('Episode Length')
        ax.set_title('Episode Length')
        ax.legend()

    # 2. Intervention Rate
    ax = axes[0, 1]
    if actor_metrics['intervention_rate']:
        steps, values = zip(*actor_metrics['intervention_rate'])
        ax.scatter(steps, values, alpha=0.5, s=10, c='blue')
        if len(values) > 5:
            smoothed = smooth(values, 5)
            ax.plot(steps[2:-2], smoothed, 'r-', linewidth=2, label='smoothed')
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel('Intervention Rate')
        ax.set_title('Intervention Rate')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.2, color='g', linestyle='--', alpha=0.5, label='target 20%')
        ax.legend()

    # 3-4: Empty or summary
    ax = axes[0, 2]
    ax.axis('off')
    ax.text(0.5, 0.5, 'Actor Metrics\n(x-axis: env_steps)',
            ha='center', va='center', fontsize=12, transform=ax.transAxes)

    ax = axes[0, 3]
    ax.axis('off')
    if actor_metrics['intervention_rate']:
        steps, values = zip(*actor_metrics['intervention_rate'])
        final_rate = values[-1] if values else 0
        min_rate = min(values) if values else 0
        ax.text(0.5, 0.7, f'Final Intervention: {final_rate:.1%}',
                ha='center', va='center', fontsize=11, transform=ax.transAxes)
        ax.text(0.5, 0.5, f'Min Intervention: {min_rate:.1%}',
                ha='center', va='center', fontsize=11, transform=ax.transAxes)
        ax.text(0.5, 0.3, f'Total Episodes: {len(values)}',
                ha='center', va='center', fontsize=11, transform=ax.transAxes)

    # ========== Row 2: Critic metrics (learner_step) ==========
    # 5. Q Mean
    ax = axes[1, 0]
    if learner_metrics['q_mean']:
        steps, values = zip(*learner_metrics['q_mean'])
        ax.plot(steps, values, 'b-', linewidth=1)
        ax.set_xlabel('Learner Steps')
        ax.set_ylabel('Q Mean')
        ax.set_title('Critic Q Mean')

    # 6. Critic Loss
    ax = axes[1, 1]
    if learner_metrics['critic_loss']:
        steps, values = zip(*learner_metrics['critic_loss'])
        ax.plot(steps, values, 'b-', linewidth=1)
        ax.set_xlabel('Learner Steps')
        ax.set_ylabel('Critic Loss')
        ax.set_title('Critic Loss')
        ax.set_yscale('log')

    # 7. Critic Disagreement
    ax = axes[1, 2]
    if learner_metrics['critic_disagreement']:
        steps, values = zip(*learner_metrics['critic_disagreement'])
        ax.plot(steps, values, 'orange', linewidth=1)
        ax.set_xlabel('Learner Steps')
        ax.set_ylabel('Disagreement')
        ax.set_title('Critic Disagreement')

    # 8. Offline Ratio
    ax = axes[1, 3]
    if learner_metrics['offline_ratio']:
        steps, values = zip(*learner_metrics['offline_ratio'])
        ax.plot(steps, values, 'purple', linewidth=1)
        ax.set_xlabel('Learner Steps')
        ax.set_ylabel('Offline Ratio')
        ax.set_title('Offline Ratio (Annealing)')
        ax.set_ylim(0, 1)

    # ========== Row 3: Actor/Policy metrics (learner_step) ==========
    # 9. Actor Loss
    ax = axes[2, 0]
    if learner_metrics['actor_loss']:
        steps, values = zip(*learner_metrics['actor_loss'])
        ax.plot(steps, values, 'g-', linewidth=1)
        ax.set_xlabel('Learner Steps')
        ax.set_ylabel('Actor Loss')
        ax.set_title('Actor Loss')

    # 10. Entropy
    ax = axes[2, 1]
    if learner_metrics['entropy']:
        steps, values = zip(*learner_metrics['entropy'])
        ax.plot(steps, values, 'm-', linewidth=1)
        ax.set_xlabel('Learner Steps')
        ax.set_ylabel('Entropy')
        ax.set_title('Policy Entropy')

    # 11. Temperature
    ax = axes[2, 2]
    if learner_metrics['temperature']:
        steps, values = zip(*learner_metrics['temperature'])
        ax.plot(steps, values, 'c-', linewidth=1)
        ax.set_xlabel('Learner Steps')
        ax.set_ylabel('Temperature')
        ax.set_title('SAC Temperature')

    # 12. Buffer Sizes
    ax = axes[2, 3]
    if learner_metrics['demo_size'] and learner_metrics['online_size']:
        steps_d, values_d = zip(*learner_metrics['demo_size'])
        steps_o, values_o = zip(*learner_metrics['online_size'])
        ax.plot(steps_d, values_d, 'b-', linewidth=1, label='demo')
        ax.plot(steps_o, values_o, 'r-', linewidth=1, label='online')
        ax.set_xlabel('Learner Steps')
        ax.set_ylabel('Buffer Size')
        ax.set_title('Buffer Sizes')
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize SERL training metrics')
    parser.add_argument('--file', '-f', type=str,
                        default="/home/pi-zero/Documents/see_to_reach_feel_to_insert/task/peg_in_hole_square_III/checkpoints_2-7/metrics.jsonl",
                        help='Path to metrics.jsonl file')
    parser.add_argument('--start-line', '-s', type=int, default=0,
                        help='Skip lines before this (0-indexed)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the plot')
    args = parser.parse_args()

    print(f"Loading metrics from {args.file} (starting from line {args.start_line})...")
    learner_metrics, actor_metrics = load_metrics(args.file, args.start_line)

    print(f"\nLearner metrics (x-axis: learner_step):")
    for key, values in learner_metrics.items():
        if values:
            print(f"  {key}: {len(values)} data points")

    print(f"\nActor metrics (x-axis: env_steps):")
    for key, values in actor_metrics.items():
        if values:
            print(f"  {key}: {len(values)} data points")

    save_path = None if args.no_save else args.file.replace('.jsonl', '_plot.png')
    plot_metrics(learner_metrics, actor_metrics, save_path)
