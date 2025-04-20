"""
Single Layer Analysis Example

This script demonstrates how to analyze a specific layer's forward and backward 
computations in detail, including mathematical formulations.
"""
import os
import sys
import torch
import torch.nn as nn
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modular components
from torch_formula.models import TinyCNN
from torch_formula.core.workflow import AnalysisWorkflow

def main():
    """Main function for single layer analysis"""
    parser = argparse.ArgumentParser(description="PyTorch Layer Analysis Example")
    parser.add_argument("--layer", type=str, default="conv1",
                       help="Layer name to analyze (conv1, pool1, conv2, etc.)")
    parser.add_argument("--position", type=str, default="0,0,4,4",
                       help="Position for detailed analysis (comma-separated)")
    parser.add_argument("--output-dir", type=str, default="./output/layer_analysis",
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse position
    base_position = tuple(map(int, args.position.split(',')))
    
    # Setup model with specific weights
    model = setup_model()
    
    # Create input tensor
    input_tensor = create_patterned_input(size=8)
    
    # Analyze the specific layer
    analyze_layer(model, input_tensor, args.layer, base_position, args.output_dir)
    
def analyze_layer(model, input_tensor, layer_name, base_position, output_dir):
    """
    Analyze a specific layer of the model
    
    Args:
        model: The PyTorch model
        input_tensor: Input tensor for the model
        layer_name: Name of the layer to analyze
        base_position: Base position tuple for detailed analysis
        output_dir: Directory to save the analysis
    """
    # Adjust position based on the layer
    position = adjust_position_for_layer(layer_name, base_position)
    
    print(f"Analyzing layer '{layer_name}' at position {position}")
    
    # Create workflow instance
    workflow = AnalysisWorkflow(model, input_tensor.shape, output_dir)
    
    # Run the detailed analysis for the specified layer
    try:
        result = workflow.analyze_specific_layer(layer_name, position)
        print(f"Analysis complete! Results saved to: {result['html_path']}")
        print(f"Open {result['html_path']} to view the detailed analysis")
    except Exception as e:
        print(f"Error analyzing layer: {str(e)}")
        print("Please check if the position is appropriate for this layer's output dimensions.")

def adjust_position_for_layer(layer_name, base_position):
    """
    Adjust position based on the layer type
    
    Args:
        layer_name: Name of the layer
        base_position: Base position (typically for conv1)
        
    Returns:
        Adjusted position for the specific layer
    """
    if layer_name == "conv1":
        # Original position
        return base_position
    elif layer_name in ["pool1", "conv2"]:
        # Half size (8x8 -> 4x4 after pool1)
        return (base_position[0], base_position[1], base_position[2]//2, base_position[3]//2)
    elif layer_name in ["pool2", "conv3"]:
        # Quarter size (4x4 -> 2x2 after pool2)
        return (base_position[0], base_position[1], base_position[2]//4, base_position[3]//4)
    elif layer_name == "conv_out":
        # Position for final layer (output is 1x1 after conv_out)
        return (base_position[0], base_position[1], 0, 0)
    else:
        # Default to original position
        print(f"Warning: No position adjustment defined for layer '{layer_name}'. Using original position.")
        return base_position

def setup_model():
    """Setup and initialize model with specific weights"""
    model = TinyCNN()
    
    # Set specific weights for better visualization
    torch.manual_seed(42)
    with torch.no_grad():
        # First filter
        model.conv1.weight[0, 0, :, :] = torch.tensor([
            [0.3, -0.5, 0.2],
            [0.7, 0.4, -0.1],
            [-0.3, 0.8, 0.5]
        ])
        
        # Second filter
        model.conv1.weight[1, 0, :, :] = torch.tensor([
            [0.2, 0.5, -0.3],
            [-0.4, 0.6, 0.1],
            [0.7, -0.2, 0.4]
        ])
        
        # Bias values
        model.conv1.bias[0] = 0.1
        model.conv1.bias[1] = -0.2
    
    return model

def create_patterned_input(size=8):
    """Create an input tensor with a pattern for better visualization"""
    # Create input tensor
    input_tensor = torch.zeros(1, 1, size, size)
    
    # Fill with a pattern
    for i in range(size):
        for j in range(size):
            # Create a pattern: gradual increase with variations
            val = (i + j) / (2*size - 2)  # normalized between 0 and 1
            
            # Add variations
            if (i + j) % 2 == 0:
                val += 0.2
            else:
                val -= 0.1
            
            input_tensor[0, 0, i, j] = val
    
    # Add a cross pattern in the center
    center_start = size//2 - 1
    center_end = size//2 + 1
    input_tensor[0, 0, center_start:center_end, :] += 0.3  # Horizontal line
    input_tensor[0, 0, :, center_start:center_end] += 0.3  # Vertical line
    
    # Ensure values are within a reasonable range
    input_tensor = torch.clamp(input_tensor, -1.0, 1.0)
    
    return input_tensor

if __name__ == "__main__":
    main()
