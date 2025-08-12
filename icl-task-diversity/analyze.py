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
from scipy.optimize import curve_fit


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


def get_most_recent_multirun() -> str:
    """Find the most recent multirun ID in the outputs/multirun directory."""
    multirun_dir = Path("outputs/multirun")
    if not multirun_dir.exists():
        raise FileNotFoundError("outputs/multirun directory not found")
    
    # Find all multirun directories with multirun.yaml files
    multirun_dirs = []
    for run_dir in multirun_dir.iterdir():
        if run_dir.is_dir() and (run_dir / "multirun.yaml").exists():
            multirun_dirs.append(run_dir.name)
    
    if not multirun_dirs:
        raise FileNotFoundError("No completed multiruns found (no multirun.yaml files)")
    
    # Sort by folder name (date format: 2025-08-11_11-45-46)
    multirun_dirs.sort()
    return multirun_dirs[-1]  # Return the most recent (last when sorted)


def load_log(run_id: str) -> dict:
    """Load the log.json file for a given run ID."""
    log_path = Path("outputs") / run_id / "log.json"
    
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    with open(log_path, "r") as f:
        return json.load(f)


def plot_training_loss(log: dict, run_id: str, output_dir: Path = None):
    """Plot training loss over steps."""
    steps = log["train/step"]
    eval_steps = log.get("eval/step", [])
    lr_values = log["train/lr"]
    train_losses = log.get("train/loss", [])
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    
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
        if key.startswith("eval/") and key != "eval/step":
            task_name = key.split("/")[1]
            if task_name not in eval_metrics:
                eval_metrics[task_name] = {}
            for metric_name, metric_values in value.items():
                eval_metrics[task_name][metric_name] = metric_values
    
    # Plot MSE for each task and baseline
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    color_idx_mse = 0
    color_idx_rel = 0
    
    for task_name, metrics in eval_metrics.items():
        for metric_name, values in metrics.items():
            if "Transformer |" in metric_name and "(RelErr)" not in metric_name:
                # MSE metrics (exclude relative error)
                mean_values = [np.mean(v) for v in values]
                axes[2].plot(eval_steps, mean_values, 
                        color=colors[color_idx_mse % len(colors)], 
                        linewidth=2,
                        label=f"{format_task_name_for_display(task_name)}: {metric_name}")
                color_idx_mse += 1
            elif "Transformer |" in metric_name and "(RelErr)" in metric_name:
                # Relative error metrics
                mean_values = [np.mean(v) for v in values]
                axes[3].plot(eval_steps, mean_values, 
                        color=colors[color_idx_rel % len(colors)], 
                        linewidth=2,
                        label=f"{format_task_name_for_display(task_name)}: {metric_name}")
                color_idx_rel += 1
    
    # Configure MSE plot (axes[2])
    axes[2].set_xlabel("Training Step")
    axes[2].set_ylabel("Mean Squared Error")
    axes[2].set_title(f"MSE Evaluation Metrics - {run_id}")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Configure Relative Error plot (axes[3])
    axes[3].set_xlabel("Training Step")
    axes[3].set_ylabel("Relative Error")
    axes[3].set_title(f"Relative Error Evaluation Metrics - {run_id}")
    axes[3].grid(True, alpha=0.3)
    axes[3].set_yscale('log')
    axes[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = Path("outputs") / run_id
    output_path = output_dir / "training_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()


def format_task_name_for_display(task_name):
    """Format task name for display in legends, replacing 'Fixed task' with 'Shifted task'."""
    if task_name.startswith("Fixed task"):
        return task_name.replace("Fixed task", "Shifted task")
    return task_name


def icl_power_law(k, D, alpha, C):
    """Power law function: D/(k+1)^alpha + C"""
    # return D / ((k + 1) ** alpha) + C
    return C / ((k + 1) ** alpha)


def extract_min_mse_params(log: dict) -> tuple[dict, dict]:
    """Extract minimum MSE over context length and mean MSE over context length for last iteration for all tasks.
    
    Returns:
        tuple: (min_mse_dict, mean_mse_dict) where:
            min_mse_dict: {task_name: min_mse}
            mean_mse_dict: {task_name: mean_mse_over_context}
    """
    eval_steps = log.get("eval/step", [])
    if not eval_steps:
        return {}, {}
    
    # Extract evaluation metrics for the final step
    final_step_idx = -1
    eval_metrics = {}
    for key, value in log.items():
        if key.startswith("eval/") and key != "eval/step":
            task_name = key.split("/")[1]
            if task_name not in eval_metrics:
                eval_metrics[task_name] = {}
            for metric_name, metric_values in value.items():
                eval_metrics[task_name][metric_name] = metric_values
    
    min_mse_results = {}
    mean_mse_results = {}
    
    for task_name, metrics in eval_metrics.items():
        if "Fixed task" not in task_name:
            continue
        for metric_name, values in metrics.items():
            if "Transformer | True" in metric_name and "(RelErr)" not in metric_name and values:
                # Find the minimum MSE across all context lengths and time steps
                min_global_mse = float('inf')
                
                for step_idx, step_mse_values in enumerate(values):
                    if step_mse_values:
                        step_min_mse = min(step_mse_values)
                        if step_min_mse < min_global_mse:
                            min_global_mse = step_min_mse
                
                # Get mean MSE over context length for the last iteration
                final_step_mse_values = values[final_step_idx]
                if final_step_mse_values:
                    mean_mse_over_context = np.mean(final_step_mse_values)
                else:
                    mean_mse_over_context = float('inf')
                
                mean_mse_results[task_name] = mean_mse_over_context
                min_mse_results[task_name] = min_global_mse
                break  # Only take the first valid metric per task
    
    return min_mse_results, mean_mse_results


def extract_power_law_params(log: dict) -> dict:
    """Extract power law parameters (alpha, C) for all tasks from log data.
    
    Returns:
        dict: {task_name: (alpha, C, r_squared)} for successfully fitted tasks
    """
    eval_steps = log.get("eval/step", [])
    if not eval_steps:
        return {}
    
    # Extract evaluation metrics for the final step
    final_step_idx = -1
    eval_metrics = {}
    for key, value in log.items():
        if key.startswith("eval/") and key != "eval/step":
            task_name = key.split("/")[1]
            if task_name not in eval_metrics:
                eval_metrics[task_name] = {}
            for metric_name, metric_values in value.items():
                eval_metrics[task_name][metric_name] = metric_values
    
    results = {}
    
    for task_name, metrics in eval_metrics.items():
        for metric_name, values in metrics.items():
            if "Transformer | True" in metric_name and "(RelErr)" not in metric_name and values:
                # Get MSE values for final step
                mse_values = values[final_step_idx]
                if not mse_values or len(mse_values) < 3:
                    continue
                    
                k = np.arange(len(mse_values))  # Context lengths: 0, 1, 2, ...
                
                try:
                    # Fit the power law curve
                    # initial_guess = [mse_values[0] - np.min(mse_values), 1.0, np.min(mse_values)]
                    initial_guess = [0., 1.0, mse_values[0]]
                    
                    popt, pcov = curve_fit(icl_power_law, k, mse_values, p0=initial_guess, 
                                         bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=5000)
                    
                    D_fit, alpha_fit, C_fit = popt
                    
                    # Compute R-squared
                    y_pred = icl_power_law(k, D_fit, alpha_fit, C_fit)
                    ss_res = np.sum((mse_values - y_pred) ** 2)
                    ss_tot = np.sum((mse_values - np.mean(mse_values)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    results[task_name] = (alpha_fit, C_fit, r_squared)
                    break  # Only take the first valid metric per task
                    
                except Exception as e:
                    # Skip failed fits
                    continue
    
    return results


def fit_mse_curves_and_compute_metrics(log: dict, run_id: str):
    """Fit MSE curves and compute ICL performance metrics."""
    eval_steps = log.get("eval/step", [])
    if not eval_steps:
        print("No evaluation steps found in log")
        return
    
    # Extract evaluation metrics for the final step
    final_step_idx = -1
    eval_metrics = {}
    for key, value in log.items():
        if key.startswith("eval/") and key != "eval/step":
            task_name = key.split("/")[1]
            if task_name not in eval_metrics:
                eval_metrics[task_name] = {}
            for metric_name, metric_values in value.items():
                eval_metrics[task_name][metric_name] = metric_values
    
    print(f"\n=== ICL Performance Metrics Analysis: {run_id} ===")
    print("Fitting MSE curves with formula: D/(k+1)^alpha + C")
    print("where k is context length (0-indexed), D = init error at k=0 - C")
    
    for task_name, metrics in eval_metrics.items():
        print(f"\n{task_name}:")
        
        for metric_name, values in metrics.items():
            if "Transformer | True" in metric_name and "(RelErr)" not in metric_name and values:
                # Get MSE values for final step
                mse_values = values[final_step_idx]
                if not mse_values or len(mse_values) < 3:
                    continue
                    
                k = np.arange(len(mse_values))  # Context lengths: 0, 1, 2, ...
                
                try:
                    # Fit the power law curve
                    # Initial guess: D = mse_values[0] - min(mse_values), alpha = 1, C = min(mse_values)
                    initial_guess = [mse_values[0] - np.min(mse_values), 1.0, np.min(mse_values)]
                    
                    popt, pcov = curve_fit(icl_power_law, k, mse_values, p0=initial_guess, 
                                         bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=5000)
                    
                    D_fit, alpha_fit, C_fit = popt
                    
                    # Compute metrics
                    avg_performance = np.mean(mse_values)
                    
                    # Compute R-squared
                    y_pred = icl_power_law(k, D_fit, alpha_fit, C_fit)
                    ss_res = np.sum((mse_values - y_pred) ** 2)
                    ss_tot = np.sum((mse_values - np.mean(mse_values)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    print(f"  {metric_name}:")
                    print(f"    Average MSE over k: {avg_performance:.6f}")
                    print(f"    Power law fit - alpha: {alpha_fit:.4f}, C: {C_fit:.6f}")
                    print(f"    D (init error - C): {D_fit:.6f}")
                    print(f"    R²: {r_squared:.4f}")
                    
                except Exception as e:
                    print(f"  {metric_name}: Failed to fit curve - {str(e)}")
                    avg_performance = np.mean(mse_values)
                    print(f"    Average MSE over k: {avg_performance:.6f}")


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
        if key.startswith("eval/") and key != "eval/step":
            task_name = key.split("/")[1]
            print(f"\n{task_name}:")
            for metric_name, metric_values in value.items():
                if "Transformer |" in metric_name and metric_values:
                    final_mse = np.mean(metric_values[-1])
                    print(f"  {metric_name}: {final_mse:.6f}")


def plot_icl_for_all_steps(log: dict, run_id: str, output_dir: Path = None):
    """Plot MSE and Relative Error vs context length for every evaluation step."""
    eval_steps = log.get("eval/step", [])
    if not eval_steps:
        print("No evaluation steps found in log")
        return
    
    # Extract evaluation metrics
    eval_metrics = {}
    for key, value in log.items():
        if key.startswith("eval/") and key != "eval/step":
            task_name = key.split("/")[1]
            if task_name not in eval_metrics:
                eval_metrics[task_name] = {}
            for metric_name, metric_values in value.items():
                eval_metrics[task_name][metric_name] = metric_values
    
    # Create output directories for ICL plots
    if output_dir is None:
        output_dir = Path("outputs") / run_id
    icl_mse_dir = output_dir / "icl_plots_mse"
    icl_rel_err_dir = output_dir / "icl_plots_rel_err"
    icl_mse_dir.mkdir(exist_ok=True)
    icl_rel_err_dir.mkdir(exist_ok=True)
    
    # Colors for different tasks
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    
    # Generate plots for each evaluation step
    for step_idx, eval_step in enumerate(eval_steps):
        
        # MSE Plot
        plt.figure(figsize=(12, 8))
        color_idx = 0
        
        for task_name, metrics in eval_metrics.items():
            for metric_name, values in metrics.items():
                if "Transformer | True" in metric_name and "(RelErr)" not in metric_name and values and step_idx < len(values):
                    # Get MSE by position for this step
                    mse_by_position = values[step_idx]  # List of MSE values by position
                    n_points = len(mse_by_position)
                    positions = list(range(1, n_points + 1))  # Context length positions
                    
                    plt.plot(positions, mse_by_position,
                            color=colors[color_idx % len(colors)],
                            linewidth=2,
                            marker='o',
                            markersize=6,
                            label=f"{format_task_name_for_display(task_name)}")
                    color_idx += 1
        
        plt.xlabel("Context Length (Position)")
        plt.ylabel("MSE (Transformer vs True)")
        plt.title(f"ICL MSE Performance at Step {eval_step} - {run_id}")
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save MSE plot for this step
        output_path = icl_mse_dir / f"icl_step_{eval_step:04d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close to save memory
        
        # Relative Error Plot
        plt.figure(figsize=(12, 8))
        color_idx = 0
        
        for task_name, metrics in eval_metrics.items():
            for metric_name, values in metrics.items():
                if "Transformer | True (RelErr)" in metric_name and values and step_idx < len(values):
                    # Get Relative Error by position for this step
                    rel_err_by_position = values[step_idx]  # List of RelErr values by position
                    n_points = len(rel_err_by_position)
                    positions = list(range(1, n_points + 1))  # Context length positions
                    
                    plt.plot(positions, rel_err_by_position,
                            color=colors[color_idx % len(colors)],
                            linewidth=2,
                            marker='o',
                            markersize=6,
                            label=f"{format_task_name_for_display(task_name)}")
                    color_idx += 1
        
        plt.xlabel("Context Length (Position)")
        plt.ylabel("Relative Error (Transformer vs True)")
        plt.title(f"ICL Relative Error Performance at Step {eval_step} - {run_id}")
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save Relative Error plot for this step
        output_path = icl_rel_err_dir / f"icl_step_{eval_step:04d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close to save memory
    
    print(f"ICL MSE plots for {len(eval_steps)} steps saved to: {icl_mse_dir}")
    print(f"ICL Relative Error plots for {len(eval_steps)} steps saved to: {icl_rel_err_dir}")


def plot_task_shift_analysis(run_paths: list, output_dir: Path = None, run_labels: list = None):
    """Plot alpha and C parameters vs task shift for multiple runs.
    
    Args:
        run_paths: List of Path objects pointing to runs or multirun subdirs
        output_dir: Directory to save plots (optional)
        run_labels: Custom labels for runs (optional)
    """
    if not run_paths:
        print("No run paths provided for task shift analysis")
        return
    
    # Collect data from all runs
    data = {}  # {run_label: [(task_center, alpha, C, r_squared, task_name), ...]}
    
    for i, run_path in enumerate(run_paths):
        run_path = Path(run_path)
        
        # Determine run label
        if run_labels and i < len(run_labels):
            run_label = run_labels[i]
        else:
            run_label = run_path.name
        
        # Check if this is a multirun directory or single run
        if (run_path / "multirun.yaml").exists():
            # This is a multirun directory - we want to analyze each subrun separately
            subdirs = []
            for subdir in run_path.iterdir():
                if subdir.is_dir() and subdir.name.isdigit() and (subdir / "log.json").exists():
                    subdirs.append(subdir)
            
            # Sort numerically
            subdirs.sort(key=lambda x: int(x.name))
            
            # Process each subrun as a separate run
            for subdir_idx, subdir in enumerate(subdirs):
                # Load config and log for this subrun
                config_path = subdir / "config.json"
                log_path = subdir / "log.json"
                
                if not config_path.exists() or not log_path.exists():
                    continue
                
                with open(config_path, 'r') as f:
                    config = json.load(f)
                with open(log_path, 'r') as f:
                    log = json.load(f)
                
                # Extract task centers from config
                task_centers = config.get('eval', {}).get('task_centers', [])
                
                # Extract power law parameters for all tasks
                power_law_params = extract_power_law_params(log)
                
                # Create run data for this subrun
                if run_labels and subdir_idx < len(run_labels):
                    # Use custom name for this subrun
                    subrun_label = run_labels[subdir_idx].strip()
                else:
                    subrun_label = f"{run_label}-{subdir.name}"
                run_data = []
                
                # Add Test tasks (task center = 0)
                if "Test tasks" in power_law_params:
                    alpha, C, r_squared = power_law_params["Test tasks"]
                    run_data.append((0.0, alpha, C, r_squared, "Test tasks"))
                
                # Add Fixed tasks
                for task_center in task_centers:
                    task_name = f"Fixed task {task_center}"
                    if task_name in power_law_params:
                        alpha, C, r_squared = power_law_params[task_name]
                        run_data.append((task_center, alpha, C, r_squared, task_name))
                
                if run_data:
                    data[subrun_label] = run_data
        
        elif (run_path / "log.json").exists():
            # This is a single run
            config_path = run_path / "config.json"
            log_path = run_path / "log.json"
            
            if not config_path.exists():
                continue
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            with open(log_path, 'r') as f:
                log = json.load(f)
            
            # Extract task centers from config
            task_centers = config.get('eval', {}).get('task_centers', [])
            
            # Extract power law parameters for all tasks
            power_law_params = extract_power_law_params(log)
            
            run_data = []
            
            # Add Test tasks (task center = 0)
            if "Test tasks" in power_law_params:
                alpha, C, r_squared = power_law_params["Test tasks"]
                run_data.append((0.0, alpha, C, r_squared, "Test tasks"))
            
            # Add Fixed tasks
            for task_center in task_centers:
                task_name = f"Fixed task {task_center}"
                if task_name in power_law_params:
                    alpha, C, r_squared = power_law_params[task_name]
                    run_data.append((task_center, alpha, C, r_squared, task_name))
            
            if run_data:
                data[run_label] = run_data
        
        else:
            print(f"Warning: {run_path} is neither a valid run nor multirun directory")
            continue
    
    if not data:
        print("No valid data found for task shift analysis")
        return
    
    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    max_shift = float('inf')
    
    for i, (run_label, run_data) in enumerate(data.items()):
        if not run_data:
            continue
        
        # Sort by task center
        run_data.sort(key=lambda x: x[0])
        
        task_centers = [x[0] for x in run_data if x[0] <= max_shift]
        alphas = [x[1] for x in run_data if x[0] <= max_shift]
        Cs = [x[2] for x in run_data if x[0] <= max_shift]
        
        color = colors[i % len(colors)]
        
        # Plot alpha vs task center
        ax1.plot(task_centers, alphas, 'o-', color=color, linewidth=2, 
                markersize=6, label=run_label)
        
        # Plot C vs task center
        ax2.plot(task_centers, Cs, 'o-', color=color, linewidth=2, 
                markersize=6, label=run_label)
    
    # Configure alpha plot
    ax1.set_xlabel("Task Center (Task Shift)")
    ax1.set_ylabel("Alpha (Power Law Exponent)")
    ax1.set_title("Alpha vs Task Shift")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Configure C plot
    ax2.set_xlabel("Task Center (Task Shift)")
    ax2.set_ylabel("C (Asymptotic Error)")
    ax2.set_title("C vs Task Shift")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        # Check if we're analyzing multiruns by looking at the first run path
        if run_paths and (run_paths[0] / "multirun.yaml").exists():
            output_dir = run_paths[0]  # Use the multirun directory
        else:
            output_dir = Path("outputs")
    output_path = output_dir / "task_shift_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Task shift analysis plot saved to: {output_path}")
    
    plt.show()
    
    # Print summary
    print(f"\n=== Task Shift Analysis Summary ===")
    for run_label, run_data in data.items():
        if not run_data:
            continue
        print(f"\n{run_label}:")
        for task_center, alpha, C, r_squared, task_name in sorted(run_data, key=lambda x: x[0]):
            print(f"  {task_name} (center={task_center}): alpha={alpha:.4f}, C={C:.6f}, R²={r_squared:.4f}")


def plot_min_mse_analysis(run_paths: list, output_dir: Path = None, run_labels: list = None):
    """Plot minimum MSE vs task shift and mean MSE over context length for last iteration for multiple runs.
    
    Args:
        run_paths: List of Path objects pointing to runs or multirun subdirs
        output_dir: Directory to save plots (optional)
        run_labels: Custom labels for runs (optional)
    """
    if not run_paths:
        print("No run paths provided for minimum MSE analysis")
        return
    
    # Collect minimum MSE and mean MSE over context length data from all runs
    min_mse_data = {}  # {run_label: [(task_center, min_mse, task_name), ...]}
    mean_mse_data = {}  # {run_label: [(task_center, mean_mse, task_name), ...]}
    
    for i, run_path in enumerate(run_paths):
        run_path = Path(run_path)
        
        # Determine run label
        if run_labels and i < len(run_labels):
            run_label = run_labels[i]
        else:
            run_label = run_path.name
        
        # Check if this is a multirun directory or single run
        if (run_path / "multirun.yaml").exists():
            # This is a multirun directory - we want to analyze each subrun separately
            subdirs = []
            for subdir in run_path.iterdir():
                if subdir.is_dir() and subdir.name.isdigit() and (subdir / "log.json").exists():
                    subdirs.append(subdir)
            
            # Sort numerically
            subdirs.sort(key=lambda x: int(x.name))
            
            # Process each subrun as a separate run
            for subdir_idx, subdir in enumerate(subdirs):
                # Load config and log for this subrun
                config_path = subdir / "config.json"
                log_path = subdir / "log.json"
                
                if not config_path.exists() or not log_path.exists():
                    continue
                
                with open(config_path, 'r') as f:
                    config = json.load(f)
                with open(log_path, 'r') as f:
                    log = json.load(f)
                
                # Extract task centers from config
                task_centers = config.get('eval', {}).get('task_centers', [])
                
                # Extract minimum MSE and mean MSE over context length for all tasks
                min_mse_params, mean_mse_params = extract_min_mse_params(log)
                
                # Create run data for this subrun
                if run_labels and subdir_idx < len(run_labels):
                    # Use custom name for this subrun
                    subrun_label = run_labels[subdir_idx].strip()
                else:
                    subrun_label = f"{run_label}-{subdir.name}"
                min_mse_run_data = []
                mean_mse_run_data = []
                
                # Add Test tasks (task center = 0)
                if "Test tasks" in min_mse_params:
                    min_mse = min_mse_params["Test tasks"]
                    mean_mse = mean_mse_params.get("Test tasks", 0)
                    min_mse_run_data.append((0.0, min_mse, "Test tasks"))
                    mean_mse_run_data.append((0.0, mean_mse, "Test tasks"))
                
                # Add Fixed tasks
                for task_center in task_centers:
                    task_name = f"Fixed task {task_center}"
                    if task_name in min_mse_params:
                        min_mse = min_mse_params[task_name]
                        mean_mse = mean_mse_params.get(task_name, 0)
                        min_mse_run_data.append((task_center, min_mse, task_name))
                        mean_mse_run_data.append((task_center, mean_mse, task_name))
                
                if min_mse_run_data:
                    min_mse_data[subrun_label] = min_mse_run_data
                    mean_mse_data[subrun_label] = mean_mse_run_data
        
        elif (run_path / "log.json").exists():
            # This is a single run
            config_path = run_path / "config.json"
            log_path = run_path / "log.json"
            
            if not config_path.exists():
                continue
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            with open(log_path, 'r') as f:
                log = json.load(f)
            
            # Extract task centers from config
            task_centers = config.get('eval', {}).get('task_centers', [])
            
            # Extract minimum MSE and mean MSE over context length for all tasks
            min_mse_params, mean_mse_params = extract_min_mse_params(log)
            
            min_mse_run_data = []
            mean_mse_run_data = []
            
            # Add Test tasks (task center = 0)
            if "Test tasks" in min_mse_params:
                min_mse = min_mse_params["Test tasks"]
                mean_mse = mean_mse_params.get("Test tasks", 0)
                min_mse_run_data.append((0.0, min_mse, "Test tasks"))
                mean_mse_run_data.append((0.0, mean_mse, "Test tasks"))
            
            # Add Fixed tasks
            for task_center in task_centers:
                task_name = f"Fixed task {task_center}"
                if task_name in min_mse_params:
                    min_mse = min_mse_params[task_name]
                    mean_mse = mean_mse_params.get(task_name, 0)
                    min_mse_run_data.append((task_center, min_mse, task_name))
                    mean_mse_run_data.append((task_center, mean_mse, task_name))
            
            if min_mse_run_data:
                min_mse_data[run_label] = min_mse_run_data
                mean_mse_data[run_label] = mean_mse_run_data
        
        else:
            print(f"Warning: {run_path} is neither a valid run nor multirun directory")
            continue
    
    if not min_mse_data:
        print("No valid data found for minimum MSE analysis")
        return
    
    # Create the plot with two subplots in a separate figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    max_shift = float('inf')
    
    # Plot minimum MSE data (left subplot)
    for i, (run_label, run_data) in enumerate(min_mse_data.items()):
        if not run_data:
            continue
        
        # Sort by task center
        run_data.sort(key=lambda x: x[0])
        
        task_centers = [x[0] for x in run_data if x[0] <= max_shift]
        min_mses = [x[1] for x in run_data if x[0] <= max_shift]
        
        color = colors[i % len(colors)]
        
        # Plot minimum MSE vs task center
        ax1.plot(task_centers, min_mses, 'o-', color=color, linewidth=2, 
                markersize=6, label=run_label)
    
    # Configure minimum MSE plot
    ax1.set_xlabel("Task Center (Task Shift)")
    ax1.set_ylabel("Minimum MSE")
    ax1.set_title("Minimum MSE vs Task Shift")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.legend()
    
    # Plot mean MSE over context length data (right subplot)
    for i, (run_label, run_data) in enumerate(mean_mse_data.items()):
        if not run_data:
            continue
        
        # Sort by task center
        run_data.sort(key=lambda x: x[0])
        
        task_centers = [x[0] for x in run_data if x[0] <= max_shift]
        mean_mses = [x[1] for x in run_data if x[0] <= max_shift]
        
        color = colors[i % len(colors)]
        
        # Plot mean MSE over context length vs task center
        ax2.plot(task_centers, mean_mses, 'o-', color=color, linewidth=2, 
                markersize=6, label=run_label)
    
    # Configure mean MSE over context length plot
    ax2.set_xlabel("Task Center (Task Shift)")
    ax2.set_ylabel("Mean MSE over Context Length (Last Iteration)")
    ax2.set_title("Mean MSE over Context Length vs Task Shift")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        # Check if we're analyzing multiruns by looking at the first run path
        if run_paths and (run_paths[0] / "multirun.yaml").exists():
            output_dir = run_paths[0]  # Use the multirun directory
        else:
            output_dir = Path("outputs")
    output_path = output_dir / "min_mse_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Minimum MSE analysis plot saved to: {output_path}")
    
    plt.show()
    
    # Print summary
    print(f"\n=== Minimum MSE Analysis Summary ===")
    for run_label, run_data in min_mse_data.items():
        if not run_data:
            continue
        print(f"\n{run_label}:")
        mean_data = mean_mse_data.get(run_label, [])
        mean_dict = {x[2]: x[1] for x in mean_data}  # task_name -> mean_mse
        
        for task_center, min_mse, task_name in sorted(run_data, key=lambda x: x[0]):
            mean_mse = mean_dict.get(task_name, "N/A")
            print(f"  {task_name} (center={task_center}): min_mse={min_mse:.6f}, mean_mse_last_iter={mean_mse:.6f}")


def analyze_multirun(multirun_id: str, custom_names: list = None):
    """Analyze all runs within a multirun experiment.
    
    Args:
        multirun_id: The multirun ID to analyze
        custom_names: Optional list of custom names for each run
    """
    multirun_dir = Path("outputs/multirun") / multirun_id
    
    if not multirun_dir.exists():
        raise FileNotFoundError(f"Multirun directory not found: {multirun_dir}")
    
    # Find all subdirectories with log.json files
    run_subdirs = []
    for subdir in multirun_dir.iterdir():
        if subdir.is_dir() and subdir.name.isdigit() and (subdir / "log.json").exists():
            run_subdirs.append(subdir.name)
    
    if not run_subdirs:
        raise FileNotFoundError(f"No completed runs found in multirun {multirun_id}")
    
    # Sort subdirectories numerically
    run_subdirs.sort(key=int)
    
    print(f"\n=== Analyzing Multirun: {multirun_id} ===")
    print(f"Found {len(run_subdirs)} completed runs: {', '.join(run_subdirs)}")
    
    for i, subdir_name in enumerate(run_subdirs):
        # Use custom name if provided, otherwise use subdir_name
        display_name = custom_names[i] if custom_names and i < len(custom_names) else subdir_name
        print(f"\n--- Analyzing run {subdir_name} ({display_name}) ---")
        
        # Load log for this run
        log_path = multirun_dir / subdir_name / "log.json"
        with open(log_path, "r") as f:
            log = json.load(f)
        
        # Create run identifier for display
        run_display_id = f"{multirun_id}/{display_name}"
        
        # Run all analysis functions for this individual run
        print_summary(log, run_display_id)
        fit_mse_curves_and_compute_metrics(log, run_display_id)
        
        # For plotting functions, pass the custom output directory
        run_output_dir = Path("outputs/multirun") / multirun_id / subdir_name
        plot_training_loss(log, run_display_id, run_output_dir)
        plot_icl_for_all_steps(log, run_display_id, run_output_dir)





def parse_multirun_args(multirun_arg, run_id_arg):
    """Parse multirun arguments to extract custom names and multirun ID.
    
    Args:
        multirun_arg: The --multirun argument value (True, string with names, or None)
        run_id_arg: The run_id positional argument
    
    Returns:
        tuple: (multirun_id, custom_names_list) where custom_names_list is None if no custom names
    """
    if multirun_arg is True:
        # Just --multirun flag, no custom names
        return run_id_arg, None
    elif isinstance(multirun_arg, str):
        # Custom names provided, run_id should be the multirun_id
        custom_names = [name.strip() for name in multirun_arg.split(',')]
        return run_id_arg, custom_names
    else:
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training logs and plot metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py                    # Analyze most recent run
  python analyze.py 2025-08-06_12-24-25   # Analyze specific run
  python analyze.py --multirun         # Analyze most recent multirun
  python analyze.py --multirun 2025-08-11_11-45-46   # Analyze specific multirun
  python analyze.py --multirun "GPT-2,Transformer,LSTM" 2025-08-11_11-45-46   # Custom names
  python analyze.py --multirun --shift-analysis 2025-08-11_11-45-46   # Task shift analysis
        """
    )
    parser.add_argument(
        'run_id', 
        nargs='?', 
        help='Run ID to analyze (e.g., 2025-08-06_12-24-25). If not provided, uses most recent run.'
    )
    parser.add_argument(
        '--multirun',
        nargs='?',
        const=True,
        help='Analyze a multirun experiment instead of a single run. Optionally provide comma-separated names for runs (e.g., "name0,name1,name2")'
    )
    parser.add_argument(
        '--shift-analysis',
        action='store_true',
        help='Perform task shift analysis (alpha and C vs task centers)'
    )
    
    args = parser.parse_args()
    
    if args.shift_analysis:
        # Handle distribution shift analysis
        if args.multirun:
            # Distribution analysis for multirun(s)
            multirun_id, custom_names = parse_multirun_args(args.multirun, args.run_id)
            
            if multirun_id:
                multirun_ids = [multirun_id]
            else:
                try:
                    multirun_ids = [get_most_recent_multirun()]
                    print(f"Using most recent multirun: {multirun_ids[0]}")
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    return 1
            
            # Convert to full paths
            run_paths = [Path("outputs/multirun") / multirun_id for multirun_id in multirun_ids]
        else:
            # Distribution analysis for single run(s) - not typical but supported
            if args.run_id:
                run_ids = [args.run_id]
            else:
                try:
                    run_ids = [get_most_recent_run()]
                    print(f"Using most recent run: {run_ids[0]}")
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    return 1
            
            # Convert to full paths
            run_paths = [Path("outputs") / run_id for run_id in run_ids]
        
        # Perform task shift analysis
        try:
            plot_task_shift_analysis(run_paths, run_labels=custom_names)
            plot_min_mse_analysis(run_paths, run_labels=custom_names)
        except Exception as e:
            print(f"Error in task shift analysis: {e}")
            return 1
            
    elif args.multirun:
        # Handle multirun analysis
        multirun_id, custom_names = parse_multirun_args(args.multirun, args.run_id)
        
        if multirun_id:
            pass  # Use provided multirun_id
        else:
            try:
                multirun_id = get_most_recent_multirun()
                print(f"Using most recent multirun: {multirun_id}")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return 1
        
        # Analyze the multirun
        try:
            analyze_multirun(multirun_id, custom_names)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
    else:
        # Handle single run analysis (existing behavior)
        if args.run_id:
            run_id = args.run_id
        else:
            try:
                run_id = get_most_recent_run()
                print(f"Using most recent run: {run_id}")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return 1
        
        log = load_log(run_id)
        print_summary(log, run_id)
        fit_mse_curves_and_compute_metrics(log, run_id)
        plot_training_loss(log, run_id)
        plot_icl_for_all_steps(log, run_id)
        
    return 0


if __name__ == "__main__":
    main()
