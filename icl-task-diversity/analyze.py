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
import yaml
from scipy.optimize import curve_fit
import itertools
from matplotlib.colors import LogNorm
from safetensors.numpy import load_file


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


def load_log_with_safetensors(run_path: Path) -> dict:
    """Load log with optional safetensor speedup for evaluation data.
    
    This function is backward compatible:
    - If safetensor files exist, loads eval data from them (much faster)
    - Falls back to JSON for everything else or if safetensors don't exist
    
    Args:
        run_path: Path to run directory (e.g., outputs/2025-08-15_10-30-45)
    
    Returns:
        dict: Complete log dictionary with evaluation data
    """
    log_path = run_path / "log.json"
    eval_results_dir = run_path / "eval_results"
    
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    # Always load the base log from JSON
    with open(log_path, "r") as f:
        log = json.load(f)
    
    # Check if safetensor eval results exist and try to use them
    if eval_results_dir.exists():
        safetensor_files = sorted(eval_results_dir.glob("eval_step_*.safetensors"))
        
        if safetensor_files:
            print(f"Found {len(safetensor_files)} safetensor eval files, loading for faster analysis...")
            
            # Get evaluation steps from log
            eval_steps = log.get("eval/step", [])
            
            # Initialize eval data structure if not present
            for key in log.keys():
                if key.startswith("eval/") and key != "eval/step":
                    for metric_name in log[key].keys():
                        if isinstance(log[key][metric_name], list):
                            log[key][metric_name] = []  # Clear existing data
            
            # Load data from safetensor files
            for i, safetensor_file in enumerate(safetensor_files):
                if i >= len(eval_steps):
                    break
                    
                try:
                    tensors = load_file(safetensor_file)
                    
                    # Convert safetensor data back to log format
                    for tensor_key, tensor_data in tensors.items():
                        # Parse tensor key: {task_name}_Transformer_vs_{baseline}_MSE/RelErr
                        parts = tensor_key.split('_')
                        if len(parts) < 4:
                            continue
                            
                        # Reconstruct task name and baseline name
                        transformer_idx = parts.index('Transformer') if 'Transformer' in parts else -1
                        vs_idx = parts.index('vs') if 'vs' in parts else -1
                        
                        if transformer_idx == -1 or vs_idx == -1:
                            continue
                            
                        task_name_parts = parts[:transformer_idx]
                        baseline_parts = parts[vs_idx+1:]
                        
                        # Remove MSE/RelErr suffix
                        if baseline_parts and (baseline_parts[-1] == 'MSE' or baseline_parts[-1] == 'RelErr'):
                            metric_type = baseline_parts[-1]
                            baseline_parts = baseline_parts[:-1]
                        else:
                            continue
                            
                        # Reconstruct names (convert back from safe names)
                        task_name = ' '.join(task_name_parts).replace('_', ' ')
                        baseline_name = ' '.join(baseline_parts).replace('_', ' ')
                        
                        # Handle common task name patterns
                        if 'Fixed_task' in tensor_key:
                            task_name = task_name.replace('Fixed task', 'Fixed task')
                        if 'Test_tasks' in tensor_key:
                            task_name = 'Test tasks'
                            
                        # Build log key
                        log_key = f"eval/{task_name}"
                        if metric_type == 'RelErr':
                            metric_key = f"Transformer | {baseline_name} (RelErr)"
                        else:
                            metric_key = f"Transformer | {baseline_name}"
                        
                        # Initialize structure if needed
                        if log_key not in log:
                            log[log_key] = {}
                        if metric_key not in log[log_key]:
                            log[log_key][metric_key] = []
                            
                        # Append tensor data (convert to list for compatibility)
                        log[log_key][metric_key].append(tensor_data.tolist())
                        
                except Exception as e:
                    print(f"Warning: Could not load safetensor file {safetensor_file}: {e}")
                    # Continue with next file or fall back to JSON
                    
            print("Successfully loaded evaluation data from safetensors")
            return log
    
    # Fall back to original JSON loading if no safetensors or loading failed
    print("Using JSON evaluation data (slower)")
    return log


def normalize_error_values(values):
    """Normalize error values to handle both old and new logging formats.
    
    Args:
        values: Either list of scalars (old format) or list of lists (new format)
    
    Returns:
        List of scalars (averaged over samples if needed)
    """
    if not values:
        return values
    
    # Check if this is the new format (list of lists)
    if isinstance(values[0], list):
        # New format: average over batch dimension for each position
        return [np.mean(pos_values) for pos_values in values]
    else:
        # Old format: already scalars
        return values


def plot_training_loss(log: dict, run_id: str, output_dir: Path = None):
    """Plot training loss over steps."""
    steps = log["train/step"]
    eval_steps = log.get("eval/step", [])
    lr_values = log["train/lr"]
    train_losses = log.get("train/loss", [])
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    
    # Plot training loss
    if train_losses:
        axes[0].plot(steps, train_losses, 'r-', linewidth=2)
        axes[0].set_xlabel("Training Step")
        axes[0].set_ylabel("Training Loss")
        axes[0].set_title(f"Training Loss - {run_id}")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
    
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
                mean_values = [np.mean(normalize_error_values(v)) for v in values]
                axes[1].plot(eval_steps, mean_values, 
                        color=colors[color_idx_mse % len(colors)], 
                        linewidth=2,
                        label=f"{format_task_name_for_display(task_name)}: {metric_name}")
                color_idx_mse += 1
    
    # Plot min MSE over context length as function of training step
    for task_name, metrics in eval_metrics.items():
        for metric_name, values in metrics.items():
            if "Transformer |" in metric_name and "(RelErr)" not in metric_name:
                # Calculate min MSE over context length for each training step
                min_values = [np.min(normalize_error_values(v)) for v in values]
                axes[2].plot(eval_steps, min_values, 
                        color=colors[color_idx_rel % len(colors)], 
                        linewidth=2,
                        label=f"{format_task_name_for_display(task_name)}: {metric_name}")
                color_idx_rel += 1
    
    # Configure MSE plot (axes[1])
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Mean Squared Error")
    axes[1].set_title(f"MSE Evaluation Metrics - {run_id}")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Configure Min MSE plot (axes[2])
    axes[2].set_xlabel("Training Step")
    axes[2].set_ylabel("Min MSE over Context Length")
    axes[2].set_title(f"Min MSE over Context Length - {run_id}")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
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


def extract_min_mse_params_for_baseline(log: dict, baseline_type: str) -> tuple[dict, dict]:
    """Extract minimum MSE over context length and mean MSE over context length for last iteration for all tasks for a specific baseline.
    
    Args:
        log: The log dictionary
        baseline_type: Either 'Ridge' or 'True' to specify which baseline to use
    
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
        
        # Look for the specific baseline type
        selected_metric = None
        for metric_name, values in metrics.items():
            if f"Transformer | {baseline_type}" in metric_name and "(RelErr)" not in metric_name and values:
                selected_metric = (metric_name, values)
                break
        
        if selected_metric:
            metric_name, values = selected_metric
            # Get mean MSE over context length for the last iteration
            final_step_mse_values = normalize_error_values(values[final_step_idx])
            if final_step_mse_values:
                mean_mse_over_context = np.mean(final_step_mse_values)
                min_global_mse = min(final_step_mse_values[1:])
            else:
                mean_mse_over_context = float('inf')
                min_global_mse = float('inf')

            mean_mse_results[task_name] = mean_mse_over_context
            min_mse_results[task_name] = min_global_mse
    
    return min_mse_results, mean_mse_results


def extract_min_mse_params(log: dict) -> tuple[dict, dict]:
    """Extract minimum MSE over context length and mean MSE over context length for last iteration for all tasks.
    
    Returns:
        tuple: (min_mse_dict, mean_mse_dict) where:
            min_mse_dict: {task_name: min_mse}
            mean_mse_dict: {task_name: mean_mse_over_context}
    """
    # Try Ridge first, fallback to True
    min_mse_ridge, mean_mse_ridge = extract_min_mse_params_for_baseline(log, 'Ridge')
    if min_mse_ridge:  # If Ridge data exists, use it
        return min_mse_ridge, mean_mse_ridge
    else:  # Fallback to True
        return extract_min_mse_params_for_baseline(log, 'True')


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
        # Look for preferred metric (Ridge) first, then fallback to True
        preferred_metric = None
        fallback_metric = None
        
        for metric_name, values in metrics.items():
            if "(RelErr)" not in metric_name and values:
                if "Transformer | Ridge" in metric_name:
                    preferred_metric = (metric_name, values)
                elif "Transformer | True" in metric_name:
                    fallback_metric = (metric_name, values)
        
        # Use preferred metric if available, otherwise fallback
        selected_metric = preferred_metric or fallback_metric
        
        if selected_metric:
            metric_name, values = selected_metric
            # Get MSE values for final step
            mse_values = normalize_error_values(values[final_step_idx])
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
        
        # Look for preferred metric (Ridge) first, then fallback to True
        preferred_metric = None
        fallback_metric = None
        
        for metric_name, values in metrics.items():
            if "(RelErr)" not in metric_name and values:
                if "Transformer | Ridge" in metric_name:
                    preferred_metric = (metric_name, values)
                elif "Transformer | True" in metric_name:
                    fallback_metric = (metric_name, values)
        
        # Use preferred metric if available, otherwise fallback
        selected_metric = preferred_metric or fallback_metric
        
        if selected_metric:
            metric_name, values = selected_metric
            # Get MSE values for final step
            mse_values = normalize_error_values(values[final_step_idx])
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
            # Look for preferred metric (Ridge) first, then fallback to any Transformer metric
            preferred_metrics = []
            fallback_metrics = []
            
            for metric_name, metric_values in value.items():
                if metric_values:
                    if "Transformer | Ridge" in metric_name:
                        preferred_metrics.append((metric_name, metric_values))
                    elif "Transformer |" in metric_name:
                        fallback_metrics.append((metric_name, metric_values))
            
            # Show preferred metrics first, then fallback metrics
            metrics_to_show = preferred_metrics if preferred_metrics else fallback_metrics
            
            for metric_name, metric_values in metrics_to_show:
                final_mse = np.mean(normalize_error_values(metric_values[-1]))
                print(f"  {metric_name}: {final_mse:.6f}")


def plot_icl_for_all_steps(log: dict, run_id: str, output_dir: Path = None):
    """Plot MSE and Relative Error vs context length for every evaluation step.
    Creates separate plots for True and Ridge baselines when both are available."""
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
    
    # Check which baselines are available
    ridge_available = False
    true_available = False
    
    for task_name, metrics in eval_metrics.items():
        for metric_name in metrics.keys():
            if "Transformer | Ridge" in metric_name:
                ridge_available = True
            if "Transformer | True" in metric_name:
                true_available = True
    
    if not ridge_available and not true_available:
        print("No Transformer baseline metrics found in log")
        return
    
    # Helper function to create plots for a specific baseline
    def create_icl_plots_for_baseline(baseline_type: str, baseline_suffix: str):
        # Create output directories for ICL plots
        if output_dir is None:
            base_output_dir = Path("outputs") / run_id
        else:
            base_output_dir = output_dir
            
        icl_mse_dir = base_output_dir / f"icl_plots_mse_{baseline_suffix}"
        icl_rel_err_dir = base_output_dir / f"icl_plots_rel_err_{baseline_suffix}"
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
                    if f"Transformer | {baseline_type}" in metric_name and "(RelErr)" not in metric_name and values and step_idx < len(values):
                        # Get MSE by position for this step
                        mse_by_position = normalize_error_values(values[step_idx])  # List of MSE values by position
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
            plt.ylabel(f"MSE (Transformer vs {baseline_type})")
            plt.title(f"ICL MSE vs {baseline_type} Baseline at Step {eval_step} - {run_id}")
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
                    if f"Transformer | {baseline_type} (RelErr)" in metric_name and values and step_idx < len(values):
                        # Get Relative Error by position for this step
                        rel_err_by_position = normalize_error_values(values[step_idx])  # List of RelErr values by position
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
            plt.ylabel(f"Relative Error (Transformer vs {baseline_type})")
            plt.title(f"ICL Relative Error vs {baseline_type} Baseline at Step {eval_step} - {run_id}")
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save Relative Error plot for this step
            output_path = icl_rel_err_dir / f"icl_step_{eval_step:04d}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()  # Close to save memory
        
        print(f"ICL MSE plots ({baseline_type} baseline) for {len(eval_steps)} steps saved to: {icl_mse_dir}")
        print(f"ICL Relative Error plots ({baseline_type} baseline) for {len(eval_steps)} steps saved to: {icl_rel_err_dir}")
    
    # Create plots for available baselines
    if ridge_available:
        create_icl_plots_for_baseline('Ridge', 'ridge')
    
    if true_available:
        create_icl_plots_for_baseline('True', 'true')


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
            
            # Extract parameter names from multirun.yaml if no custom labels provided
            if run_labels is None:
                param_names = create_run_display_names(run_path, [subdir.name for subdir in subdirs])
                if param_names:
                    run_labels = [param_names.get(int(subdir.name), subdir.name) for subdir in subdirs]
            
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
    
    # Choose colormap based on number of runs
    num_runs = len(data)
    if num_runs > 10:
        # Use colormap for many runs
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20' if num_runs <= 20 else 'hsv')
        colors = [cmap(i / num_runs) for i in range(num_runs)]
    else:
        # Use discrete colors for few runs
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
        
        color = colors[i] if i < len(colors) else colors[i % len(colors)]
        
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
    
    # Helper function to collect data for a specific baseline
    def collect_mse_data_for_baseline(baseline_type: str, run_labels: list = None) -> tuple[dict, dict]:
        """Collect MSE data for a specific baseline type (True or Ridge)"""
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
                
                # Extract parameter names from multirun.yaml if no custom labels provided
                if run_labels is None:
                    param_names = create_run_display_names(run_path, [subdir.name for subdir in subdirs])
                    if param_names:
                        run_labels = [param_names.get(int(subdir.name), subdir.name) for subdir in subdirs]
                
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
                    min_mse_params, mean_mse_params = extract_min_mse_params_for_baseline(log, baseline_type)
                    
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
                min_mse_params, mean_mse_params = extract_min_mse_params_for_baseline(log, baseline_type)
                
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
        
        return min_mse_data, mean_mse_data
    
    # Check which baselines are available
    ridge_min_data, ridge_mean_data = collect_mse_data_for_baseline('Ridge', run_labels = run_labels)
    true_min_data, true_mean_data = collect_mse_data_for_baseline('True', run_labels = run_labels)
    
    # Determine which plots to create
    create_ridge_plot = bool(ridge_min_data)
    create_true_plot = bool(true_min_data)
    
    if not create_ridge_plot and not create_true_plot:
        print("No valid data found for minimum MSE analysis")
        return
    
    # Helper function to create a plot for a specific baseline
    def create_mse_plot(min_mse_data, mean_mse_data, baseline_type: str, fig_suffix: str):
        # Create the plot with two subplots in a separate figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Choose colormap based on number of runs
        num_runs = len(min_mse_data)
        if num_runs > 10:
            # Use colormap for many runs
            import matplotlib.cm as cm
            cmap = cm.get_cmap('tab20' if num_runs <= 20 else 'hsv')
            colors = [cmap(i / num_runs) for i in range(num_runs)]
        else:
            # Use discrete colors for few runs
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
            
            color = colors[i] if i < len(colors) else colors[i % len(colors)]
            
            # Plot minimum MSE vs task center
            ax1.plot(task_centers, min_mses, 'o-', color=color, linewidth=2, 
                    markersize=6, label=run_label)
        
        # Configure minimum MSE plot
        ax1.set_xlabel("Task Center (Task Shift)")
        ax1.set_ylabel(f"Best MSE vs {baseline_type} Baseline (Last Iteration)")
        ax1.set_title(f"Best MSE vs {baseline_type} Baseline vs Task Shift")
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
            
            color = colors[i] if i < len(colors) else colors[i % len(colors)]
            
            # Plot mean MSE over context length vs task center
            ax2.plot(task_centers, mean_mses, 'o-', color=color, linewidth=2, 
                    markersize=6, label=run_label)
        
        # Configure mean MSE over context length plot
        ax2.set_xlabel("Task Center (Task Shift)")
        ax2.set_ylabel(f"Mean MSE vs {baseline_type} Baseline (Last Iteration)")
        ax2.set_title(f"Mean MSE vs {baseline_type} Baseline vs Task Shift")
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        if output_dir is None:
            # Check if we're analyzing multiruns by looking at the first run path
            if run_paths and (run_paths[0] / "multirun.yaml").exists():
                output_dir_to_use = run_paths[0]  # Use the multirun directory
            else:
                output_dir_to_use = Path("outputs")
        else:
            output_dir_to_use = output_dir
            
        output_path = output_dir_to_use / f"min_mse_analysis_{fig_suffix}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Minimum MSE analysis ({baseline_type}) plot saved to: {output_path}")
        
        plt.show()
        
        # Print summary for this baseline
        print(f"\n=== Minimum MSE Analysis Summary ({baseline_type} Baseline) ===")
        for run_label, run_data in min_mse_data.items():
            if not run_data:
                continue
            print(f"\n{run_label}:")
            mean_data = mean_mse_data.get(run_label, [])
            mean_dict = {x[2]: x[1] for x in mean_data}  # task_name -> mean_mse
            
            for task_center, min_mse, task_name in sorted(run_data, key=lambda x: x[0]):
                mean_mse = mean_dict.get(task_name, "N/A")
                print(f"  {task_name} (center={task_center}): min_mse={min_mse:.6f}, mean_mse_last_iter={mean_mse:.6f}")
    
    # Create plots for available baselines
    if create_ridge_plot:
        create_mse_plot(ridge_min_data, ridge_mean_data, 'Ridge', 'ridge')
    
    if create_true_plot:
        create_mse_plot(true_min_data, true_mean_data, 'True', 'true')


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
    
    # Extract parameter names from multirun.yaml if custom names not provided
    if custom_names is None:
        param_names = create_run_display_names(multirun_dir, run_subdirs)
        if param_names:
            custom_names = [param_names.get(int(subdir), subdir) for subdir in run_subdirs]
    
    print(f"\n=== Analyzing Multirun: {multirun_id} ===")
    print(f"Found {len(run_subdirs)} completed runs: {', '.join(run_subdirs)}")
    
    for i, subdir_name in enumerate(run_subdirs):
        # Use custom name if provided, otherwise use subdir_name
        display_name = custom_names[i] if custom_names and i < len(custom_names) else subdir_name
        print(f"\n--- Analyzing run {subdir_name} ({display_name}) ---")
        
        # Load log for this run (with safetensor speedup if available)
        run_path = multirun_dir / subdir_name
        log = load_log_with_safetensors(run_path)
        
        # Create run identifier for display
        run_display_id = f"{multirun_id}/{display_name}"
        
        # Run all analysis functions for this individual run
        print_summary(log, run_display_id)
        fit_mse_curves_and_compute_metrics(log, run_display_id)
        
        # For plotting functions, pass the custom output directory
        run_output_dir = Path("outputs/multirun") / multirun_id / subdir_name
        plot_training_loss(log, run_display_id, run_output_dir)
        plot_icl_for_all_steps(log, run_display_id, run_output_dir)





def extract_swept_params(multirun_path: Path) -> list:
    """Extract swept parameter names from multirun.yaml file.
    
    Args:
        multirun_path: Path to the multirun directory
    
    Returns:
        list: List of parameter paths that were swept (e.g., ['task.distrib_param'])
    """
    multirun_yaml_path = multirun_path / "multirun.yaml"
    
    if not multirun_yaml_path.exists():
        return []
    
    try:
        with open(multirun_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract swept parameters from hydra config
        params = config.get('hydra', {}).get('sweeper', {}).get('params', {})
        
        if not params:
            return []
        
        # Return list of parameter paths
        return list(params.keys())
        
    except Exception as e:
        print(f"Warning: Could not parse multirun.yaml: {e}")
        return []


def get_param_value_from_config(config: dict, param_path: str):
    """Get parameter value from config using dotted path.
    
    Args:
        config: Configuration dictionary
        param_path: Dotted parameter path (e.g., 'task.distrib_param')
    
    Returns:
        Parameter value or None if not found
    """
    keys = param_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return None


def create_run_display_names(multirun_path: Path, run_subdirs: list) -> dict:
    """Create display names for runs based on their actual parameter values.
    
    Args:
        multirun_path: Path to the multirun directory
        run_subdirs: List of run subdirectory names (e.g., ['0', '1'])
    
    Returns:
        dict: {run_index: display_name} mapping based on actual parameter values
    """
    # Get swept parameters from multirun.yaml
    swept_params = extract_swept_params(multirun_path)
    
    if not swept_params:
        return {}
    
    display_names = {}
    
    for subdir in run_subdirs:
        config_path = multirun_path / subdir / "config.json"
        
        if not config_path.exists():
            continue
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Build display name from parameter values
            name_parts = []
            for param_path in swept_params:
                value = get_param_value_from_config(config, param_path)
                if value is not None:
                    param_name = param_path.split('.')[-1]  # Get last part of dotted path
                    name_parts.append(f"{param_name}={value}")
            
            if name_parts:
                display_names[int(subdir)] = ", ".join(name_parts)
            
        except Exception as e:
            print(f"Warning: Could not read config for run {subdir}: {e}")
            continue
    
    return display_names


def hyperparam_analysis(multirun_path: Path, output_dir: Path = None):
    """Perform hyperparameter analysis: create heatmaps of Test Task MSE vs hyperparameter pairs.
    
    For each value of task.distrib_param, creates heatmaps showing average Test Task MSE
    as a function of pairs of other hyperparameters.
    
    Args:
        multirun_path: Path to the multirun directory
        output_dir: Directory to save plots (optional)
    """
    if not multirun_path.exists():
        raise FileNotFoundError(f"Multirun directory not found: {multirun_path}")
    
    # Get swept parameters
    swept_params = extract_swept_params(multirun_path)
    if not swept_params:
        print("No swept parameters found in multirun.yaml")
        return
    
    print(f"Found swept parameters: {swept_params}")
    
    # Find all run subdirectories
    run_subdirs = []
    for subdir in multirun_path.iterdir():
        if subdir.is_dir() and subdir.name.isdigit() and (subdir / "log.json").exists():
            run_subdirs.append(subdir.name)
    
    if not run_subdirs:
        raise FileNotFoundError(f"No completed runs found in multirun")
    
    # Sort numerically
    run_subdirs.sort(key=int)
    
    # Collect data from all runs
    run_data = []  # List of dicts with parameters and MSE
    
    for subdir in run_subdirs:
        config_path = multirun_path / subdir / "config.json"
        log_path = multirun_path / subdir / "log.json"
        
        if not config_path.exists() or not log_path.exists():
            continue
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Use safetensor loading for faster analysis
        log = load_log_with_safetensors(subdir)
        
        # Extract parameter values
        param_values = {}
        for param_path in swept_params:
            value = get_param_value_from_config(config, param_path)
            param_values[param_path] = value
        
        # Extract Test Task MSE (final iteration, average over context length)
        test_task_mse = None
        eval_steps = log.get("eval/step", [])
        if eval_steps:
            final_step_idx = -1
            
            # Look for Test tasks metrics
            for key, value in log.items():
                if key.startswith("eval/Test tasks"):
                    for metric_name, metric_values in value.items():
                        if "Transformer |" in metric_name and "(RelErr)" not in metric_name and metric_values:
                            # Get MSE values for final step
                            final_mse_values = normalize_error_values(metric_values[final_step_idx])
                            if final_mse_values:
                                test_task_mse = np.mean(final_mse_values)
                                break
                    if test_task_mse is not None:
                        break
        
        if test_task_mse is not None:
            param_values['test_task_mse'] = test_task_mse
            run_data.append(param_values)
    
    if not run_data:
        print("No valid Test Task MSE data found")
        return
    
    print(f"Collected data from {len(run_data)} runs")
    
    # Group by task.distrib_param values
    distrib_param_groups = {}
    for data in run_data:
        distrib_param_val = data.get('task.distrib_param')
        if distrib_param_val is not None:
            if distrib_param_val not in distrib_param_groups:
                distrib_param_groups[distrib_param_val] = []
            distrib_param_groups[distrib_param_val].append(data)
    
    if not distrib_param_groups:
        print("No task.distrib_param found in swept parameters")
        return
    
    # Get other parameters (excluding task.distrib_param)
    other_params = [p for p in swept_params if p != 'task.distrib_param']
    
    if len(other_params) < 2:
        print(f"Need at least 2 other parameters for heatmap analysis, found: {other_params}")
        return
    
    # Create output directory
    if output_dir is None:
        output_dir = multirun_path
    heatmap_dir = output_dir / "hyperparam_heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    
    # Generate heatmaps for each distrib_param value and each pair of other parameters
    for distrib_param_val, group_data in distrib_param_groups.items():
        print(f"\nProcessing distrib_param = {distrib_param_val} ({len(group_data)} runs)")
        
        # Generate all pairs of other parameters
        param_pairs = list(itertools.combinations(other_params, 2))
        
        for param1, param2 in param_pairs:
            print(f"  Creating heatmap for {param1} vs {param2}")
            
            # Extract unique values for each parameter
            param1_values = sorted(set(data[param1] for data in group_data if param1 in data))
            param2_values = sorted(set(data[param2] for data in group_data if param2 in data))
            
            if len(param1_values) < 2 or len(param2_values) < 2:
                print(f"    Skipping: insufficient parameter variation ({len(param1_values)} x {len(param2_values)})")
                continue
            
            # Create MSE grid
            mse_grid = np.full((len(param2_values), len(param1_values)), np.nan)
            
            # Fill grid with MSE values
            for data in group_data:
                if param1 in data and param2 in data:
                    try:
                        i = param1_values.index(data[param1])
                        j = param2_values.index(data[param2])
                        mse_grid[j, i] = data['test_task_mse']
                    except ValueError:
                        continue
            
            # Check if we have enough data points
            valid_points = np.sum(~np.isnan(mse_grid))
            if valid_points < 4:
                print(f"    Skipping: insufficient data points ({valid_points})")
                continue
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            
            # Use log scale for MSE values
            valid_mse = mse_grid[~np.isnan(mse_grid)]
            vmin, vmax = np.min(valid_mse), np.max(valid_mse)
            
            im = plt.imshow(mse_grid, aspect='auto', origin='lower', 
                          norm=LogNorm(vmin=vmin, vmax=vmax), cmap='viridis')
            
            # Set ticks and labels
            plt.xticks(range(len(param1_values)), [str(v) for v in param1_values])
            plt.yticks(range(len(param2_values)), [str(v) for v in param2_values])
            
            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label('Test Task MSE (log scale)')
            
            # Add text annotations with MSE values
            for i in range(len(param1_values)):
                for j in range(len(param2_values)):
                    if not np.isnan(mse_grid[j, i]):
                        text_color = 'white' if mse_grid[j, i] < np.exp(np.log(vmin) + 0.7 * (np.log(vmax) - np.log(vmin))) else 'black'
                        plt.text(i, j, f'{mse_grid[j, i]:.2e}', 
                               ha='center', va='center', color=text_color, fontsize=8)
            
            # Labels and title
            param1_name = param1.split('.')[-1]
            param2_name = param2.split('.')[-1]
            plt.xlabel(f'{param1_name} ({param1})')
            plt.ylabel(f'{param2_name} ({param2})')
            plt.title(f'Test Task MSE Heatmap\ndistrib_param={distrib_param_val}\n{param1_name} vs {param2_name}')
            
            plt.tight_layout()
            
            # Save plot
            safe_param1 = param1.replace('.', '_')
            safe_param2 = param2.replace('.', '_')
            filename = f"heatmap_distrib_{distrib_param_val}_{safe_param1}_vs_{safe_param2}.png"
            output_path = heatmap_dir / filename
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"    Saved: {output_path}")
            
            plt.close()
    
    print(f"\nHyperparameter analysis completed. Heatmaps saved to: {heatmap_dir}")


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
  python analyze.py --multirun --hyperparam-analysis 2025-08-11_11-45-46   # Hyperparameter analysis
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
    parser.add_argument(
        '--hyperparam-analysis',
        action='store_true',
        help='Perform hyperparameter analysis with heatmaps of MSE vs hyperparameter pairs for each distrib_param value'
    )
    
    args = parser.parse_args()
    
    if args.hyperparam_analysis:
        # Handle hyperparameter analysis
        if not args.multirun:
            print("Error: --hyperparam-analysis requires --multirun")
            return 1
        
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
        
        # Perform hyperparameter analysis
        try:
            multirun_path = Path("outputs/multirun") / multirun_id
            hyperparam_analysis(multirun_path)
        except Exception as e:
            print(f"Error in hyperparameter analysis: {e}")
            return 1
            
    elif args.shift_analysis:
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
            raise e
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
        
        # Use safetensor loading for single runs too
        run_path = Path("outputs") / run_id
        log = load_log_with_safetensors(run_path)
        print_summary(log, run_id)
        fit_mse_curves_and_compute_metrics(log, run_id)
        plot_training_loss(log, run_id)
        plot_icl_for_all_steps(log, run_id)
        
    return 0


if __name__ == "__main__":
    main()
