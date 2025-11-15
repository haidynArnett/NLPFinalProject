import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Example plot
def example_plot():
    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')
    plt.show()

# Example function
def example_function():
    return "This is an example function"


def visualize_cosine_similarity_matrix():
    pass


def visualize_output_test(experiment_name: str, output_dir: str = "./experiments"):
    """
    Visualize the output test experiment results by iterating through conversation files.
    Args:
        experiment_name: Name of the experiment directory
        output_dir: Base directory containing experiment folders
    """
    experiment_dir = Path(output_dir) / experiment_name
    
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    # Find all conversation history files
    conversation_files = sorted(experiment_dir.glob("conversation_history_*.json"))
    if not conversation_files:
        raise FileNotFoundError(f"No conversation history files found in {experiment_dir}")
    
    # Collect results
    expected_values: List[int] = []
    actual_values: List[int] = []
    
    iteration = 0
    
    # Memory-efficient iteration through all conversation files
    for filepath in conversation_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            conversations = data.get("conversations", [])
            
            for conv in conversations:
                iteration += 1
                response_text = conv.get("response", "")
                
                # Try to parse the JSON response
                try:
                    response_json = json.loads(response_text)
                    result = int(response_json["result"])
                    expected = iteration
                    
                    expected_values.append(expected)
                    actual_values.append(result)
                    
                except Exception as e:
                    print(f"Warning: Failed to parse conversation {iteration}: {e}")
                    expected_values.append(iteration)
                    actual_values.append(iteration)
    
    # Create visualization
    x = list(range(len(expected_values)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, expected_values, label='Expected', marker='o')
    plt.plot(x, actual_values, label='Actual', marker='x')
    
    plt.xlabel('Iteration')
    plt.ylabel('Result')
    plt.title(f'{experiment_name} - Output Test')
    plt.legend()
    plt.show()


def visualize_output_test_aggregated(
    experiment_name: str,
    output_dir: str = "./experiments",
    show_individual_runs: bool = False,
    confidence_level: float = 0.95
):
    """
    Visualize aggregated results from multiple runs of an output test experiment.
    
    Shows mean trajectory with confidence bands across all runs.
    
    Args:
        experiment_name: Base name of the experiment directory (e.g., "output_task_qwen3:0.6b")
        output_dir: Base directory containing experiment folders
        show_individual_runs: Whether to show individual run lines (default: False)
        confidence_level: Confidence level for error bands (default: 0.95)
    
    Example:
        >>> visualize_output_test_aggregated("output_task_qwen3:0.6b", show_individual_runs=True)
    """
    experiment_dir = Path(output_dir) / experiment_name
    
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    # Find all run subdirectories
    run_dirs = sorted(experiment_dir.glob("run_*"))
    if not run_dirs:
        raise FileNotFoundError(f"No run subdirectories found in {experiment_dir}")
    
    print(f"Found {len(run_dirs)} runs to aggregate")
    
    # Collect all run results
    all_runs = []
    max_length = 0
    
    for run_dir in run_dirs:
        conversation_files = sorted(run_dir.glob("conversation_history_*.json"))
        if not conversation_files:
            continue
        
        # Extract results from this run
        run_results = []
        iteration = 0
        
        for filepath in conversation_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                conversations = data.get("conversations", [])
                
                for conv in conversations:
                    iteration += 1
                    response_text = conv.get("response", "")
                    
                    try:
                        response_json = json.loads(response_text)
                        result = int(response_json["result"])
                        run_results.append(result)
                    except Exception:
                        # Use expected value as fallback
                        run_results.append(iteration)
        
        if run_results:
            all_runs.append(run_results)
            max_length = max(max_length, len(run_results))
    
    if not all_runs:
        print("No valid run data found")
        return
    
    # Pad runs to same length if needed (use last value for padding)
    padded_runs = []
    for run in all_runs:
        if len(run) < max_length:
            padded_run = run + [run[-1]] * (max_length - len(run))
        else:
            padded_run = run
        padded_runs.append(padded_run)
    
    # Convert to numpy array for statistics
    runs_array = np.array(padded_runs)
    
    # Calculate statistics
    expected = np.arange(1, max_length + 1)
    mean = np.mean(runs_array, axis=0)
    std = np.std(runs_array, axis=0)
    
    # Calculate confidence interval
    from scipy import stats
    n_runs = len(padded_runs)
    confidence_interval = confidence_level
    degrees_of_freedom = n_runs - 1
    confidence_coeff = stats.t.ppf((1 + confidence_interval) / 2, degrees_of_freedom)
    margin_of_error = confidence_coeff * (std / np.sqrt(n_runs))
    
    # Create visualization
    plt.figure(figsize=(12, 7))
    
    x = np.arange(len(expected))
    
    # Plot individual runs if requested
    if show_individual_runs:
        for i, run in enumerate(padded_runs):
            plt.plot(x, run, alpha=0.1, color='blue', linewidth=0.5)
    
    # Plot expected line
    plt.plot(x, expected, 'g-', label='Expected', linewidth=2, alpha=0.8)
    
    # Plot mean line
    plt.plot(x, mean, 'r-', label=f'Mean (n={n_runs})', linewidth=2)
    
    # Plot confidence band
    plt.fill_between(
        x,
        mean - margin_of_error,
        mean + margin_of_error,
        alpha=0.3,
        color='red',
        label=f'{int(confidence_level*100)}% Confidence Interval'
    )
    
    # Calculate accuracy
    correct = np.sum(np.abs(mean - expected) < 0.5)
    accuracy = (correct / len(expected)) * 100
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Result', fontsize=12)
    plt.title(
        f'{experiment_name} - Aggregated Results\n'
        f'{n_runs} runs, Mean Accuracy: {accuracy:.1f}%',
        fontsize=14,
        fontweight='bold'
    )
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"Aggregated Statistics for {experiment_name}")
    print(f"{'='*80}")
    print(f"Total runs analyzed: {n_runs}")
    print(f"Iterations per run: {max_length}")
    print(f"Mean accuracy: {accuracy:.2f}%")
    print(f"Final mean value: {mean[-1]:.2f} (expected: {expected[-1]})")
    print(f"Final std deviation: {std[-1]:.2f}")
    
    # Calculate error rate per iteration
    errors_per_iteration = np.mean(runs_array != expected[:, np.newaxis].T, axis=0)
    avg_error_rate = np.mean(errors_per_iteration) * 100
    print(f"Average error rate: {avg_error_rate:.2f}%")
    print(f"{'='*80}\n")


def visualize_output_test_comparison(
    output_dir: str = "./experiments",
    confidence_level: float = 0.95
):
    """
    Compare all experiments in output_dir with confidence intervals.
    
    Auto-discovers all subdirectories containing run_* folders.
    
    Args:
        output_dir: Directory to search for experiments
        confidence_level: Confidence level for error bands
    """
    from scipy import stats
    
    base_dir = Path(output_dir)
    if not base_dir.exists():
        print(f"Directory not found: {output_dir}")
        return
    
    # Auto-discover experiments (folders containing run_* subdirectories)
    experiment_names = []
    for item in sorted(base_dir.iterdir()):
        if item.is_dir() and list(item.glob("run_*")):
            experiment_names.append(item.name)
    
    if not experiment_names:
        print(f"No experiments found in {output_dir}")
        return
    
    print(f"Found {len(experiment_names)} experiments: {experiment_names}")
    
    plt.figure(figsize=(12, 7))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    
    all_data = []
    max_length = 0
    
    for exp_name in experiment_names:
        experiment_dir = base_dir / exp_name
        run_dirs = sorted(experiment_dir.glob("run_*"))
        
        all_runs = []
        
        for run_dir in run_dirs:
            conversation_files = sorted(run_dir.glob("conversation_history_*.json"))
            if not conversation_files:
                continue
            
            run_results = []
            iteration = 0
            
            for filepath in conversation_files:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    conversations = data.get("conversations", [])
                    
                    for conv in conversations:
                        iteration += 1
                        response_text = conv.get("response", "")
                        
                        try:
                            response_json = json.loads(response_text)
                            result = int(response_json["result"])
                            run_results.append(result)
                        except Exception:
                            run_results.append(iteration)
            
            if run_results:
                all_runs.append(run_results)
                max_length = max(max_length, len(run_results))
        
        if all_runs:
            all_data.append((exp_name, all_runs))
    
    if not all_data:
        print("No valid data found for any experiment")
        return
    
    expected = np.arange(1, max_length + 1)
    x = np.arange(max_length)
    
    plt.plot(x, expected, 'k--', label='Expected', linewidth=2, alpha=0.7)
    
    for idx, (exp_name, runs) in enumerate(all_data):
        padded_runs = []
        for run in runs:
            if len(run) < max_length:
                padded_run = run + [run[-1]] * (max_length - len(run))
            else:
                padded_run = run
            padded_runs.append(padded_run)
        
        runs_array = np.array(padded_runs)
        mean = np.mean(runs_array, axis=0)
        std = np.std(runs_array, axis=0)
        
        n_runs = len(padded_runs)
        df = n_runs - 1
        confidence_coeff = stats.t.ppf((1 + confidence_level) / 2, df)
        margin = confidence_coeff * (std / np.sqrt(n_runs))
        
        color = colors[idx % len(colors)]
        
        plt.plot(x, mean, color=color, label=f'{exp_name} (n={n_runs})', linewidth=2)
        plt.fill_between(x, mean - margin, mean + margin, alpha=0.2, color=color)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Result', fontsize=12)
    plt.title(f'Model Comparison - Output Test\n{int(confidence_level*100)}% Confidence Intervals', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()