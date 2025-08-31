#!/usr/bin/env python3
"""
Loading utilities for training logs and safetensor files.
"""
import json
from pathlib import Path
import time
from functools import lru_cache
import concurrent.futures
import multiprocessing
from safetensors.numpy import load_file

# Global configuration for parallel processing
MAX_NUM_CPUS = min(8, multiprocessing.cpu_count())


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


@lru_cache(maxsize=1024)
def parse_tensor_key(tensor_key: str) -> tuple:
    """Parse tensor key with caching for performance.
    
    Args:
        tensor_key: String like "Fixed_task_2_0_Transformer_vs_Ridge_MSE"
    
    Returns:
        tuple: (task_name, baseline_name, metric_key, log_key) or None if invalid
    """
    # Find metric type (MSE or RelErr)
    if tensor_key.endswith('_MSE'):
        metric_type = 'MSE'
        base_key = tensor_key[:-4]  # Remove '_MSE'
    elif tensor_key.endswith('_RelErr'):
        metric_type = 'RelErr'  
        base_key = tensor_key[:-7]  # Remove '_RelErr'
    else:
        return None
    
    # Split by '_Transformer_vs_' to separate task from baseline
    if '_Transformer_vs_' not in base_key:
        return None
    
    task_part, baseline_part = base_key.split('_Transformer_vs_', 1)
    
    # Optimize string replacements - avoid regex
    task_name = task_part.replace('_', ' ')
    # Handle decimal numbers in task names (e.g., "2_0" -> "2.0")
    if ' ' in task_name:
        parts = task_name.split(' ')
        for i in range(len(parts) - 1):
            if parts[i].isdigit() and parts[i + 1].isdigit():
                parts[i] = parts[i] + '.' + parts[i + 1]
                parts.pop(i + 1)
                break
        task_name = ' '.join(parts)
    
    baseline_name = baseline_part.replace('_', ' ')
    # Handle decimal numbers in baseline names
    if ' ' in baseline_name:
        parts = baseline_name.split(' ')
        for i in range(len(parts) - 1):
            if parts[i].isdigit() and parts[i + 1].isdigit():
                parts[i] = parts[i] + '.' + parts[i + 1]
                parts.pop(i + 1)
                break
        baseline_name = ' '.join(parts)
    
    # Handle special cases
    if task_name.startswith('Test tasks'):
        task_name = 'Test tasks'
    
    # Build log key and metric key
    log_key = f"eval/{task_name}"
    if metric_type == 'RelErr':
        metric_key = f"Transformer | {baseline_name} (RelErr)"
    else:
        metric_key = f"Transformer | {baseline_name}"
    
    return task_name, baseline_name, metric_key, log_key


def filter_load_file(file_path):
    """Load a safetensor file but filter out keys that end in 'Std'.
    
    Args:
        file_path: Path to the safetensor file
    
    Returns:
        dict: Filtered tensor dictionary without Std keys
    """
    tensors = load_file(file_path)
    # Filter out keys ending with 'Std'
    filtered_tensors = {k: v for k, v in tensors.items() if not k.endswith('_Std')}
    return filtered_tensors


def load_safetensor_file(file_info: tuple) -> tuple:
    """Load a single safetensor file and return parsed data.
    
    Args:
        file_info: (file_path, file_index)
    
    Returns:
        tuple: (file_index, tensors_dict, success_flag, error_msg)
    """
    file_path, file_index = file_info
    
    try:
        tensors = filter_load_file(file_path)
        return file_index, tensors, True, None
    except Exception as e:
        return file_index, None, False, str(e)


def load_log_with_safetensors(run_path: Path) -> dict:
    """Load log with optimized parallel safetensor speedup for evaluation data.
    
    This function is backward compatible:
    - If safetensor files exist, loads eval data from them using parallel processing (much faster)
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
    start_time = time.time()
    with open(log_path, "r") as f:
        log = json.load(f)
    
    # Check if safetensor eval results exist and try to use them
    if eval_results_dir.exists():
        safetensor_files = sorted(eval_results_dir.glob("eval_step_*.safetensors"))
        
        if safetensor_files:
            print(f"Found {len(safetensor_files)} safetensor eval files, loading with parallel optimization...")
            
            # Get evaluation steps from log
            eval_steps = log.get("eval/step", [])
            
            # Pre-allocate eval data structure based on existing log structure
            eval_data = {}
            all_tensor_keys = set()
            
            # First pass: collect all possible tensor keys from first file to pre-allocate
            try:
                first_tensors = filter_load_file(safetensor_files[0])
                all_tensor_keys = set(first_tensors.keys())
            except Exception:
                # Fall back to JSON if we can't even load first file
                print("Warning: Could not load first safetensor file, falling back to JSON")
                return log
            
            # Pre-build structure for all expected metrics
            num_files = min(len(safetensor_files), len(eval_steps))
            for tensor_key in all_tensor_keys:
                parsed = parse_tensor_key(tensor_key)
                if parsed:
                    task_name, baseline_name, metric_key, log_key = parsed
                    
                    if log_key not in eval_data:
                        eval_data[log_key] = {}
                    if metric_key not in eval_data[log_key]:
                        # Pre-allocate list of correct size to avoid dynamic resizing
                        eval_data[log_key][metric_key] = [None] * num_files
            
            # Parallel loading of safetensor files
            file_infos = [(safetensor_files[i], i) for i in range(num_files)]
            
            # Use parallel processing with ThreadPoolExecutor
            num_workers = min(MAX_NUM_CPUS, len(file_infos))
            
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    load_start = time.time()
                    
                    # Submit all file loading tasks
                    future_to_index = {
                        executor.submit(load_safetensor_file, file_info): file_info[1] 
                        for file_info in file_infos
                    }
                    
                    # Process results as they complete
                    failed_files = 0
                    for future in concurrent.futures.as_completed(future_to_index):
                        file_index, tensors, success, error_msg = future.result()
                        
                        if not success:
                            print(f"Warning: Could not load safetensor file index {file_index}: {error_msg}")
                            failed_files += 1
                            continue
                        
                        # Process each tensor in the loaded file
                        for tensor_key, tensor_data in tensors.items():
                            parsed = parse_tensor_key(tensor_key)
                            if parsed:
                                task_name, baseline_name, metric_key, log_key = parsed
                                
                                # Direct assignment instead of list operations
                                if (log_key in eval_data and 
                                    metric_key in eval_data[log_key] and 
                                    file_index < len(eval_data[log_key][metric_key])):
                                    eval_data[log_key][metric_key][file_index] = tensor_data.tolist()
                    
                    load_time = time.time() - load_start
                    print(f"Parallel loading completed in {load_time:.2f}s using {num_workers} workers")
                    if failed_files > 0:
                        print(f"Warning: {failed_files} files failed to load")
                
            except Exception as e:
                print(f"Warning: Parallel loading failed ({e}), falling back to sequential loading")
                
                # Fall back to sequential loading
                for i, safetensor_file in enumerate(safetensor_files):
                    if i >= len(eval_steps):
                        break
                        
                    try:
                        tensors = filter_load_file(safetensor_file)
                        
                        for tensor_key, tensor_data in tensors.items():
                            parsed = parse_tensor_key(tensor_key)
                            if parsed:
                                task_name, baseline_name, metric_key, log_key = parsed
                                
                                if (log_key in eval_data and 
                                    metric_key in eval_data[log_key] and 
                                    i < len(eval_data[log_key][metric_key])):
                                    eval_data[log_key][metric_key][i] = tensor_data.tolist()
                                    
                    except Exception as e:
                        print(f"Warning: Could not load safetensor file {safetensor_file}: {e}")
                        continue
            
            # Update log with loaded eval data - direct assignment, no filtering needed
            for log_key, metrics in eval_data.items():
                if log_key not in log:
                    log[log_key] = {}
                for metric_key, values in metrics.items():
                    # Filter out None values (files that failed to load)
                    filtered_values = [v for v in values if v is not None]
                    if filtered_values:
                        log[log_key][metric_key] = filtered_values
                    
            total_time = time.time() - start_time
            print(f"Successfully loaded evaluation data from safetensors in {total_time:.2f}s total")
            return log
    
    # Fall back to original JSON loading if no safetensors or loading failed
    print("Using JSON evaluation data (slower)")
    return log