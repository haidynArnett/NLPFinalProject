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


def visualize_telephone_test_aggregated(
    experiment_name: str,
    output_dir: str = "./experiments",
    show_individual_runs: bool = False,
    confidence_level: float = 0.95
):
    """
    Visualize aggregated similarity decay from multiple telephone test runs.
    
    Shows cosine similarity to initial text over iterations with confidence bands.
    
    Args:
        experiment_name: Base name of the experiment directory
        output_dir: Base directory containing experiment folders
        show_individual_runs: Whether to show individual run lines
        confidence_level: Confidence level for error bands
    """
    from scipy import stats
    from sklearn.metrics.pairwise import cosine_similarity as cosine_sim
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    experiment_dir = Path(output_dir) / experiment_name
    
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    run_dirs = sorted(experiment_dir.glob("run_*"))
    if not run_dirs:
        raise FileNotFoundError(f"No run subdirectories found in {experiment_dir}")
    
    print(f"Found {len(run_dirs)} runs to aggregate")
    
    all_runs_similarities = []
    max_length = 0
    
    for run_dir in run_dirs:
        conversation_files = sorted(run_dir.glob("conversation_history_*.json"))
        if not conversation_files:
            continue
        
        embeddings = []
        
        for filepath in conversation_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                conversations = data.get("conversations", [])
                
                for conv in conversations:
                    embedding = conv.get("embedding")
                    if embedding:
                        embeddings.append(embedding)
        
        if embeddings and len(embeddings) > 0:
            embeddings_array = np.array(embeddings)
            initial_embedding = embeddings_array[0:1]
            
            # Compute similarity to initial text for each iteration
            similarities = []
            for i in range(len(embeddings_array)):
                sim = cosine_sim(initial_embedding, embeddings_array[i:i+1])[0][0]
                similarities.append(sim)
            
            all_runs_similarities.append(similarities)
            max_length = max(max_length, len(similarities))
    
    if not all_runs_similarities:
        print("No valid run data found")
        return
    
    # Pad runs to same length
    padded_runs = []
    for run in all_runs_similarities:
        if len(run) < max_length:
            padded_run = run + [run[-1]] * (max_length - len(run))
        else:
            padded_run = run
        padded_runs.append(padded_run)
    
    runs_array = np.array(padded_runs)
    
    # Calculate statistics
    mean = np.mean(runs_array, axis=0)
    std = np.std(runs_array, axis=0)
    
    n_runs = len(padded_runs)
    df = n_runs - 1
    confidence_coeff = stats.t.ppf((1 + confidence_level) / 2, df)
    margin = confidence_coeff * (std / np.sqrt(n_runs))
    
    # Create visualization
    plt.figure(figsize=(12, 7))
    
    x = np.arange(len(mean))

    # Fit logarithmic function
    def log_func(x, a, b):
        return -a * np.log(x + 1e-9) + b
    popt, pcov = curve_fit(log_func, x, mean, p0=(1, 1))
    # Plot fitted curve
    # x_fit = np.linspace(min(x), max(x), 300)
    # y_fit = log_func(x_fit, *popt)
    plt.plot(x, log_func(x, *popt), color='black', linewidth=2, label='Fitted Log Curve')

    # Plot individual runs if requested
    if show_individual_runs:
        for i, run in enumerate(padded_runs):
            plt.plot(x, run, alpha=0.1, color='blue', linewidth=0.5)
    
    # Plot mean line
    plt.plot(x, mean, 'r-', label=f'Mean Similarity (n={n_runs})', linewidth=2)
    
    # Plot confidence band
    plt.fill_between(
        x,
        mean - margin,
        mean + margin,
        alpha=0.3,
        color='red',
        label=f'{int(confidence_level*100)}% Confidence Interval'
    )
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cosine Similarity to Initial Text', fontsize=12)
    plt.title(
        f'{experiment_name} - Telephone Test\n'
        f'{n_runs} runs, Final Similarity: {mean[-1]:.3f}',
        fontsize=14,
        fontweight='bold'
    )
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"Aggregated Statistics for {experiment_name}")
    print(f"{'='*80}")
    print(f"Total runs analyzed: {n_runs}")
    print(f"Iterations per run: {max_length}")
    print(f"Initial similarity: {mean[0]:.4f}")
    print(f"Final mean similarity: {mean[-1]:.4f}")
    print(f"Similarity drop: {mean[0] - mean[-1]:.4f}")
    print(f"Final std deviation: {std[-1]:.4f}")
    print(f"{'='*80}\n")


def visualize_telephone_test_comparison(
    output_dir: str = "./experiments",
    confidence_level: float = 0.95
):
    """
    Compare all telephone test experiments with confidence intervals.
    
    Auto-discovers all subdirectories containing run_* folders.
    
    Args:
        output_dir: Directory to search for experiments
        confidence_level: Confidence level for error bands
    """
    from scipy import stats
    from sklearn.metrics.pairwise import cosine_similarity as cosine_sim
    
    base_dir = Path(output_dir)
    if not base_dir.exists():
        print(f"Directory not found: {output_dir}")
        return
    
    # Auto-discover telephone experiments
    experiment_names = []
    for item in sorted(base_dir.iterdir()):
        if item.is_dir() and item.name.startswith("telephone_") and list(item.glob("run_*")):
            experiment_names.append(item.name)
    
    if not experiment_names:
        print(f"No telephone experiments found in {output_dir}")
        return
    
    print(f"Found {len(experiment_names)} experiments: {experiment_names}")
    
    plt.figure(figsize=(12, 7))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    
    all_data = []
    max_length = 0
    
    for exp_name in experiment_names:
        experiment_dir = base_dir / exp_name
        run_dirs = sorted(experiment_dir.glob("run_*"))
        
        all_runs_similarities = []
        
        for run_dir in run_dirs:
            conversation_files = sorted(run_dir.glob("conversation_history_*.json"))
            if not conversation_files:
                continue
            
            embeddings = []
            
            for filepath in conversation_files:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    conversations = data.get("conversations", [])
                    
                    for conv in conversations:
                        embedding = conv.get("embedding")
                        if embedding:
                            embeddings.append(embedding)
            
            if embeddings and len(embeddings) > 0:
                embeddings_array = np.array(embeddings)
                initial_embedding = embeddings_array[0:1]
                
                similarities = []
                for i in range(len(embeddings_array)):
                    sim = cosine_sim(initial_embedding, embeddings_array[i:i+1])[0][0]
                    similarities.append(sim)
                
                all_runs_similarities.append(similarities)
                max_length = max(max_length, len(similarities))
        
        if all_runs_similarities:
            all_data.append((exp_name, all_runs_similarities))
    
    if not all_data:
        print("No valid data found for any experiment")
        return
    
    x = np.arange(max_length)
    
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
    plt.ylabel('Cosine Similarity to Initial Text', fontsize=12)
    plt.title(f'Model Comparison - Telephone Test\n{int(confidence_level*100)}% Confidence Intervals', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.show()

def visualize_telephone_log_functions(
    output_dir: str = "./experiments",
    confidence_level: float = 0.95
):
    """
    Fit and plot logarithmic decay curves for each telephone-test experiment.
    
    Auto-discovers experiment directories by looking for subdirectories that
    contain run_* folders. For each experiment, loads all runs, computes the mean
    similarity decay curve, fits a logarithmic function, and visualizes the fit.
    
    Args:
        output_dir: Directory containing experiment folders.
        confidence_level: Confidence level for confidence bands.
    """
    from scipy import stats
    from sklearn.metrics.pairwise import cosine_similarity as cosine_sim
    from scipy.optimize import curve_fit
    
    base_dir = Path(output_dir)
    if not base_dir.exists():
        print(f"Directory not found: {output_dir}")
        return

    # ------------------------------------
    # Auto-discover experiments
    # ------------------------------------
    experiment_names = []
    for item in sorted(base_dir.iterdir()):
        if item.is_dir() and list(item.glob("run_*")):
            experiment_names.append(item.name)

    if not experiment_names:
        print(f"No experiments found in {output_dir}")
        return

    print(f"Found {len(experiment_names)} experiments: {experiment_names}")

    # ------------------------------------
    # Plot setup
    # ------------------------------------
    plt.figure(figsize=(12, 7))
    # colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
        "#aec7e8",  # Light Blue
    ]

    # ------------------------------------
    # Process each experiment
    # ------------------------------------
    for idx, exp_name in enumerate(experiment_names):

        experiment_dir = base_dir / exp_name
        run_dirs = sorted(experiment_dir.glob("run_*"))

        all_runs_similarities = []
        max_length = 0

        # ---- Load all run cosine similarity curves ----
        for run_dir in run_dirs:
            conversation_files = sorted(run_dir.glob("conversation_history_*.json"))
            if not conversation_files:
                continue

            embeddings = []

            for filepath in conversation_files:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    conversations = data.get("conversations", [])

                    for conv in conversations:
                        emb = conv.get("embedding")
                        if emb:
                            embeddings.append(emb)

            if not embeddings:
                continue

            embeddings_array = np.array(embeddings)
            initial_embedding = embeddings_array[0:1]

            similarities = []
            for i in range(len(embeddings_array)):
                sim = cosine_sim(initial_embedding, embeddings_array[i:i+1])[0][0]
                similarities.append(sim)

            all_runs_similarities.append(similarities)
            max_length = max(max_length, len(similarities))

        if not all_runs_similarities:
            print(f"No valid runs for experiment {exp_name}, skipping.")
            continue

        # ---- Pad runs ----
        padded_runs = []
        for run in all_runs_similarities:
            if len(run) < max_length:
                padded_run = run + [run[-1]] * (max_length - len(run))
            else:
                padded_run = run
            padded_runs.append(padded_run)

        runs_array = np.array(padded_runs)

        # ---- Mean + CI ----
        mean = np.mean(runs_array, axis=0)
        std = np.std(runs_array, axis=0)

        n_runs = len(padded_runs)
        df = n_runs - 1
        confidence_coeff = stats.t.ppf((1 + confidence_level) / 2, df)
        margin = confidence_coeff * (std / np.sqrt(n_runs))

        x = np.arange(len(mean))
        color = colors[idx % len(colors)]

        # ------------------------------------
        # Fit logarithmic function to mean curve
        # ------------------------------------
        def log_func(x, a, b):
            # Same structure you used earlier
            return -a * np.log(x + 1e-9) + b

        try:
            popt, pcov = curve_fit(log_func, x, mean, p0=(1, 1))
            fitted_y = log_func(x, *popt)
            log_label = f"{exp_name} Log Fit (a={popt[0]:.3f}, b={popt[1]:.3f})"
        except Exception as e:
            print(f"Curve fit failed for {exp_name}: {e}")
            fitted_y = None

        # ------------------------------------
        # Plot
        # ------------------------------------
        plt.plot(x, mean, color=color, linewidth=2, label=f"{exp_name} Mean (n={n_runs})")
        plt.fill_between(x, mean - margin, mean + margin, alpha=0.15, color=color)

        if fitted_y is not None:
            plt.plot(x, fitted_y, '--', color=color, linewidth=2, label=log_label)

    # ------------------------------------
    # Final plot setup
    # ------------------------------------
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.title("Logarithmic Fits for Telephone Test Experiments", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    # plt.legend(fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
    plt.show()

def visualize_telephone_log_functions_separate(
    output_dir: str = "./experiments",
    confidence_level: float = 0.95
):
    """
    For each telephone test experiment, generate a **separate visualization** with:
    - mean similarity curve
    - confidence interval
    - logarithmic decay curve fit
    
    Args:
        output_dir: Directory of experiments
        confidence_level: Confidence interval level
    """
    from scipy import stats
    from sklearn.metrics.pairwise import cosine_similarity as cosine_sim
    from scipy.optimize import curve_fit

    base_dir = Path(output_dir)
    if not base_dir.exists():
        print(f"Directory not found: {output_dir}")
        return

    # Discover experiments
    experiment_names = []
    for item in sorted(base_dir.iterdir()):
        if item.is_dir() and list(item.glob("run_*")):
            experiment_names.append(item.name)

    if not experiment_names:
        print(f"No experiments found in {output_dir}")
        return

    print(f"Found {len(experiment_names)} experiments: {experiment_names}")

    # Process each experiment individually
    for exp_name in experiment_names:

        experiment_dir = base_dir / exp_name
        run_dirs = sorted(experiment_dir.glob("run_*"))

        all_runs_similarities = []
        max_length = 0

        # Load all runs
        for run_dir in run_dirs:
            conversation_files = sorted(run_dir.glob("conversation_history_*.json"))
            if not conversation_files:
                continue

            embeddings = []
            for filepath in conversation_files:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    conversations = data.get("conversations", [])

                    for conv in conversations:
                        emb = conv.get("embedding")
                        if emb:
                            embeddings.append(emb)

            if not embeddings:
                continue

            embeddings_array = np.array(embeddings)
            initial_embedding = embeddings_array[0:1]

            similarities = []
            for i in range(len(embeddings_array)):
                sim = cosine_sim(initial_embedding, embeddings_array[i:i+1])[0][0]
                similarities.append(sim)

            all_runs_similarities.append(similarities)
            max_length = max(max_length, len(similarities))

        if not all_runs_similarities:
            print(f"No valid runs for experiment {exp_name}")
            continue

        # Pad runs
        padded_runs = []
        for run in all_runs_similarities:
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

        x = np.arange(len(mean))

        # Fit log curve
        def log_func(x, a, b):
            return -a * np.log(x + 1e-9) + b

        try:
            popt, pcov = curve_fit(log_func, x, mean, p0=(1, 1))
            fitted_y = log_func(x, *popt)
        except Exception as e:
            print(f"Curve fit failed for {exp_name}: {e}")
            fitted_y = None

        # ---------------------------------------------------------
        # Separate visualization per model
        # ---------------------------------------------------------
        plt.figure(figsize=(12, 7))

        plt.plot(x, mean, label=f"Mean (n={n_runs})", color="blue", linewidth=2)
        plt.fill_between(x, mean - margin, mean + margin,
                         alpha=0.3, color="blue",
                         label=f"{int(confidence_level*100)}% Confidence Interval")

        if fitted_y is not None:
            plt.plot(x, fitted_y, '--', color="black",
                     linewidth=2,
                     label=f"Log Fit (a={popt[0]:.3f}, b={popt[1]:.3f})")

        plt.ylim([0, 1.05])
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Cosine Similarity", fontsize=12)

        plt.title(
            f"{exp_name} — Telephone Test\n"
            f"{n_runs} runs, Final similarity: {mean[-1]:.3f}",
            fontsize=14, fontweight='bold'
        )

        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def visualize_compare_model_parameters():
    model_parameters_map = {
        'telephone_gemma3_1b-it-fp16': 1e9,
        'telephone_gemma3_1b-it-q4_K_M': 1e9,
        'telephone_gemma3_1b-it-q8_0': 1e9,
        'telephone_gemma3_1b-it-qat': 1e9,
        'telephone_gemma3_270m-it-bf16': 270e6,
        'telephone_gemma3_270m-it-q8_0': 270e6,
        'telephone_gemma3_270m-it-qat': 270e6,
        'telephone_gemma3_4b-it-fp16': 4e9,
        'telephone_gemma3_4b-it-q4_K_M': 4e9,
        'telephone_gemma3_4b-it-q8_0': 4e9,
        'telephone_gemma3_4b-it-qat': 4e9
    }

def visualize_param_vs_performance(
    model_params: dict,
    output_dir: str = "./experiments",
    iteration: int = 20,
    confidence_level: float = 0.95,
):
    """
    Compute log-fit parameters directly from experiment data and visualize
    predicted performance at a given iteration (default = 20) as a function
    of model parameter count.

    Args:
        model_params: dict mapping model_name → parameter_count
        output_dir: directory with the experiment folders
        iteration: iteration number at which to evaluate the log fit
        confidence_level: unused but kept for consistency
    """
    from scipy.optimize import curve_fit
    from sklearn.metrics.pairwise import cosine_similarity as cosine_sim
    from scipy import stats

    base_dir = Path(output_dir)
    if not base_dir.exists():
        print(f"Directory not found: {output_dir}")
        return

    # Auto-discover experiments
    experiment_names = []
    for item in sorted(base_dir.iterdir()):
        if item.is_dir() and list(item.glob("run_*")):
            experiment_names.append(item.name)

    if not experiment_names:
        print(f"No experiments found in {output_dir}")
        return

    # ---- storage ----
    predicted_at_iteration = {}
    fitted_params = {}

    # =========================================================
    # Process each experiment / model
    # =========================================================
    for exp_name in experiment_names:
        experiment_dir = base_dir / exp_name
        run_dirs = sorted(experiment_dir.glob("run_*"))

        all_runs_sim = []
        max_length = 0

        # ---- Load all similarity sequences for this experiment ----
        for run_dir in run_dirs:
            conversation_files = sorted(run_dir.glob("conversation_history_*.json"))
            embeddings = []

            for filepath in conversation_files:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    conversations = data.get("conversations", [])

                    for conv in conversations:
                        emb = conv.get("embedding")
                        if emb:
                            embeddings.append(emb)

            if len(embeddings) == 0:
                continue

            emb_array = np.array(embeddings)
            initial = emb_array[0:1]

            similarities = []
            for i in range(len(emb_array)):
                sim = cosine_sim(initial, emb_array[i:i+1])[0][0]
                similarities.append(sim)

            all_runs_sim.append(similarities)
            max_length = max(max_length, len(similarities))

        if len(all_runs_sim) == 0:
            print(f"No usable runs for {exp_name}; skipping.")
            continue

        # ---- Pad runs for consistent length ----
        padded_runs = []
        for run in all_runs_sim:
            if len(run) < max_length:
                run = run + [run[-1]] * (max_length - len(run))
            padded_runs.append(run)

        runs_array = np.array(padded_runs)
        mean = np.mean(runs_array, axis=0)
        x = np.arange(len(mean))

        # =========================================================
        # Fit the logarithmic function
        # =========================================================
        def log_func(x, a, b):
            return -a * np.log(x + 1e-9) + b

        try:
            popt, pcov = curve_fit(log_func, x, mean, p0=(0.1, 1.0))
        except Exception as e:
            print(f"Log fit failed for {exp_name}: {e}")
            continue

        fitted_params[exp_name] = popt

        # ---- Evaluate fitted curve at desired iteration ----
        pred = log_func(iteration, *popt)
        predicted_at_iteration[exp_name] = pred

    # =========================================================
    # Scatter plot: param_count vs predicted performance
    # =========================================================
    xs = []
    ys = []
    labels = []

    for model_name, pred_y in predicted_at_iteration.items():
        if model_name not in model_params:
            print(f"Warning: {model_name} missing from model_params; skipping.")
            continue

        xs.append(model_params[model_name])
        ys.append(pred_y)
        labels.append(model_name)

    xs = np.array(xs)
    ys = np.array(ys)

    # ---- Plot ----
    plt.figure(figsize=(10, 6))
    plt.scatter(xs, ys, s=120, alpha=0.9)

    # Label each point
    for i, name in enumerate(labels):
        plt.text(xs[i] * 1.02, ys[i], name, fontsize=10)

    plt.title(
        f"Predicted Telephone-Test Performance at Iteration {iteration}\n"
        "Based on Log Fit Parameters",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Model Parameter Count", fontsize=12)
    plt.ylabel("Predicted Similarity", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()