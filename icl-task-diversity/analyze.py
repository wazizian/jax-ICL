#!/usr/bin/env python3
"""
Analysis script for training logs.

Usage:
    python analyze.py [run_id]
    python analyze.py 2025-08-06_12-24-25
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def get_most_recent_run() -> str:
    """Find the most recent run ID in the outputs directory."""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        raise FileNotFoundError("outputs directory not found")
    
    # Find all directories with log.json files
    run_dirs = []
    for run_dir in outputs_dir.iterdir():
        if run_dir.is_dir() and (run_dir / "log.json").exists():
            run_dirs.append(run_dir.name)
    
    if not run_dirs:
        raise FileNotFoundError("No completed runs found (no log.json files)")
    
    # Sort by folder name (date format: 2025-08-06_12-24-25)
    run_dirs.sort()
    return run_dirs[-1]  # Return the most recent (last when sorted)


def load_log(run_id: str) -> dict:
    """Load the log.json file for a given run ID."""
    log_path = Path("outputs") / run_id / "log.json"
    
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    with open(log_path, "r") as f:
        return json.load(f)


def plot_training_loss(log: dict, run_id: str):
    """Plot training loss over steps."""
    steps = log["train/step"]
    lr_values = log["train/lr"]
    train_losses = log.get("train/loss", [])
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 12))
    
    # Plot training loss
    if train_losses:
        axes[0].plot(steps, train_losses, 'r-', linewidth=2)
        axes[0].set_xlabel("Training Step")
        axes[0].set_ylabel("Training Loss")
        axes[0].set_title(f"Training Loss - {run_id}")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
    
    # Plot learning rate
    axes[1].plot(steps, lr_values, 'b-', linewidth=2)
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title(f"Learning Rate Schedule - {run_id}")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    # Plot evaluation metrics
    eval_metrics = {}
    for key, value in log.items():
        if key.startswith("eval/"):
            task_name = key.split("/")[1]
            if task_name not in eval_metrics:
                eval_metrics[task_name] = {}
            for metric_name, metric_values in value.items():
                eval_metrics[task_name][metric_name] = metric_values
    
    # Plot MSE for each task and baseline
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    color_idx = 0
    
    for task_name, metrics in eval_metrics.items():
        for metric_name, values in metrics.items():
            if "Transformer |" in metric_name:
                # Convert list of lists to mean values
                mean_values = [np.mean(v) for v in values]
                axes[2].plot(steps, mean_values, 
                        color=colors[color_idx % len(colors)], 
                        linewidth=2,
                        label=f"{task_name}: {metric_name}")
                color_idx += 1
    
    axes[2].set_xlabel("Training Step")
    axes[2].set_ylabel("Mean Squared Error")
    axes[2].set_title(f"Evaluation Metrics - {run_id}")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("outputs") / run_id / "training_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()


def print_summary(log: dict, run_id: str):
    """Print a summary of the training run."""
    steps = log["train/step"]
    lr_values = log["train/lr"]
    train_losses = log.get("train/loss", [])
    
    print(f"\n=== Training Summary: {run_id} ===")
    print(f"Total steps: {steps[-1] if steps else 0}")
    print(f"Final learning rate: {lr_values[-1]:.2e}" if lr_values else "N/A")
    if train_losses:
        print(f"Final training loss: {train_losses[-1]:.6f}")
        print(f"Initial training loss: {train_losses[0]:.6f}")
        if len(train_losses) > 1:
            improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
            print(f"Loss improvement: {improvement:.1f}%")
    
    # Print final evaluation metrics
    print("\nFinal Evaluation Metrics:")
    for key, value in log.items():
        if key.startswith("eval/"):
            task_name = key.split("/")[1]
            print(f"\n{task_name}:")
            for metric_name, metric_values in value.items():
                if "Transformer |" in metric_name and metric_values:
                    final_mse = np.mean(metric_values[-1])
                    print(f"  {metric_name}: {final_mse:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training logs and plot metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py                    # Analyze most recent run
  python analyze.py 2025-08-06_12-24-25   # Analyze specific run
        """
    )
    parser.add_argument(
        'run_id', 
        nargs='?', 
        help='Run ID to analyze (e.g., 2025-08-06_12-24-25). If not provided, uses most recent run.'
    )
    
    args = parser.parse_args()
    
    if args.run_id:
        run_id = args.run_id
    else:
        try:
            run_id = get_most_recent_run()
            print(f"Using most recent run: {run_id}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
    
    try:
        log = load_log(run_id)
        print_summary(log, run_id)
        plot_training_loss(log, run_id)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error loading or plotting log: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
