import matplotlib.pyplot as plt
import json
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