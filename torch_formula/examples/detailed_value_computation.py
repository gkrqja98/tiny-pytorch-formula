"""
Example script demonstrating detailed computation with actual value substitution
Show detailed computation with actual value substitution in PyTorch layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
import pandas as pd
import argparse

# Add the parent directory to the path to import our package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_formula.models import TinyCNN
from torch_formula.core.model_analyzer import ModelAnalyzer
from torch_formula.utils.detailed_computation import (
    get_detailed_computation,
    get_computation_markdown,
    format_tensor_as_markdown_table,
    format_detailed_computation_as_markdown
)


def create_simple_model():
    """
    Create a simple model with custom weights for demonstration
    """
    model = TinyCNN()
    
    # Set specific weights for clearer demonstration
    torch.manual_seed(42)
    
    # Conv1 layer: set first filter to specific values
    with torch.no_grad():
        # Set specific values for the first filter (in_channels=1, out_channels=2)
        model.conv1.weight[0, 0, :, :] = torch.tensor([
            [0.3, -0.5, 0.2],
            [0.7, 0.4, -0.1],
            [-0.3, 0.8, 0.5]
        ])
        
        # Set specific values for the second filter
        model.conv1.weight[1, 0, :, :] = torch.tensor([
            [0.2, 0.5, -0.3],
            [-0.4, 0.6, 0.1],
            [0.7, -0.2, 0.4]
        ])
        
        model.conv1.bias[0] = 0.1
        model.conv1.bias[1] = -0.2
    
    return model


def generate_input_tensor():
    """
    Generate an input tensor with specific values for clearer demonstration
    """
    # Create an 8x8 input tensor with specific pattern
    input_tensor = torch.zeros(1, 1, 8, 8)
    
    # Fill with specific values for better visualization
    for i in range(8):
        for j in range(8):
            # Create a pattern: gradual increase with some variations
            val = (i + j) / 14.0  # normalized between 0 and 1
            # Add some variations
            if (i + j) % 2 == 0:
                val += 0.2
            else:
                val -= 0.1
            
            input_tensor[0, 0, i, j] = val
    
    # Add a cross pattern
    input_tensor[0, 0, 3:5, :] += 0.3  # Horizontal line
    input_tensor[0, 0, :, 3:5] += 0.3  # Vertical line
    
    # Ensure values are within reasonable range (-1 to 1)
    input_tensor = torch.clamp(input_tensor, -1.0, 1.0)
    
    return input_tensor


def analyze_specific_position(model, input_tensor, layer_name, position, output_dir):
    """
    Analyze a specific position in a specific layer
    
    Args:
        model: The PyTorch model
        input_tensor: Input tensor to the model
        layer_name: Name of the layer to analyze
        position: Position tuple for the analysis
        output_dir: Directory to save the output
    """
    # Run the model analyzer
    analyzer = ModelAnalyzer(model, input_tensor.shape)
    output, loss = analyzer.run_model(input_tensor)
    
    print(f"\n=== Analyzing: {layer_name} layer, position {position} ===")
    
    # Get detailed computation for the specific position
    detailed_md = analyzer.get_detailed_computation(layer_name, position=position, format="markdown")
    
    # Save to file
    layer_type = "unknown"
    for name, module in model.named_modules():
        if name == layer_name:
            layer_type = type(module).__name__
    
    position_str = "_".join(map(str, position))
    filename = f"{layer_name}_{layer_type}_pos{position_str}_computation.md"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        f.write(f"# Detailed Computation for {layer_type} Layer '{layer_name}'\n\n")
        f.write(f"## Position: {position}\n\n")
        f.write(detailed_md)
    
    print(f"Detailed computation results saved to '{filepath}'")
    
    return detailed_md, analyzer


def demo_conv_layers():
    """
    Demonstrate detailed computation for Conv2d layers
    """
    print("\n" + "="*50)
    print("Conv2d Layer Detailed Computation Demonstration")
    print("="*50)
    
    # Create model with custom weights
    model = create_simple_model()
    
    # Generate input tensor
    input_tensor = generate_input_tensor()
    
    print(f"Input tensor shape: {input_tensor.shape}")
    print("Input tensor sample values (central 4x4 region):")
    print(input_tensor[0, 0, 2:6, 2:6])
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "value_computation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze conv1 layer at a specific position
    # Position: (batch, output_channel, height, width)
    conv1_positions = [
        (0, 0, 4, 4),  # First filter, central position
        (0, 1, 4, 4)   # Second filter, central position
    ]
    
    for position in conv1_positions:
        detailed_md, analyzer = analyze_specific_position(
            model, input_tensor, "conv1", position, output_dir
        )
    
    # Analyze conv2 layer
    # First compute intermediate feature map after conv1 + pool1
    with torch.no_grad():
        x = model.conv1(input_tensor)
        x = F.relu(x)
        x = model.pool1(x)
    
    conv2_positions = [
        (0, 0, 2, 2),  # First filter, central position
        (0, 3, 2, 2)   # Fourth filter, central position
    ]
    
    print("\n" + "="*50)
    print("Conv2 Layer Detailed Computation Demonstration (after Pool1)")
    print("="*50)
    
    # Get actual module and feature map
    conv2_module = model.conv2
    output = conv2_module(x)
    
    for position in conv2_positions:
        # Direct computation to show exact values
        detailed_comp = get_detailed_computation(conv2_module, x, output, position)
        markdown = format_detailed_computation_as_markdown(detailed_comp)
        
        # Save to file
        position_str = "_".join(map(str, position))
        filename = f"conv2_Conv2d_pos{position_str}_direct_computation.md"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w") as f:
            f.write(f"# Detailed Computation for Conv2d Layer 'conv2' (Direct Calculation)\n\n")
            f.write(f"## Position: {position}\n\n")
            f.write(markdown)
        
        print(f"Detailed computation results saved to '{filepath}'")
    
    return analyzer


def analyze_all_layers(analyzer=None):
    """
    Analyze all layers of the model
    """
    print("\n" + "="*50)
    print("Complete Model Layer Analysis")
    print("="*50)
    
    if analyzer is None:
        # Create model with custom weights
        model = create_simple_model()
        
        # Generate input tensor
        input_tensor = generate_input_tensor()
        
        # Create analyzer
        analyzer = ModelAnalyzer(model, input_tensor.shape)
        output, loss = analyzer.run_model(input_tensor)
    
    # Get all layer names
    layer_names = []
    for name, module in analyzer.model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)) and name:
            layer_names.append(name)
    
    print(f"Model layers: {layer_names}")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "value_computation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all layer summaries with detailed computation
    all_layers_md = analyzer.get_all_layers_summary(format="markdown", include_detailed_computation=True)
    
    # Save to file
    with open(os.path.join(output_dir, "all_layers_computation.md"), "w") as f:
        f.write("# Detailed Analysis of All Layers in TinyCNN Model\n\n")
        f.write(all_layers_md)
    
    print(f"\nDetailed analysis of all layers saved to '{os.path.join(output_dir, 'all_layers_computation.md')}'")
    
    # Also create an HTML version with more detailed visualization
    all_layers_html = analyzer.get_all_layers_summary(format="html", include_detailed_computation=True)
    
    with open(os.path.join(output_dir, "all_layers_computation.html"), "w") as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write("<title>TinyCNN Model Analysis</title>\n")
        f.write("<meta charset=\"UTF-8\">\n")
        f.write("<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML'></script>\n")
        f.write("<style>\n")
        f.write("body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }\n")
        f.write(".section { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }\n")
        f.write(".tensor-table { border-collapse: collapse; margin: 10px 0; width: auto; }\n")
        f.write(".tensor-table th, .tensor-table td { border: 1px solid #ddd; padding: 8px; text-align: right; }\n")
        f.write(".tensor-table th { background-color: #f2f2f2; }\n")
        f.write("</style>\n")
        f.write("</head>\n<body>\n")
        f.write("<h1>Detailed Analysis of All Layers in TinyCNN Model</h1>\n")
        f.write(all_layers_html)
        f.write("</body>\n</html>")
    
    print(f"HTML format analysis results saved to '{os.path.join(output_dir, 'all_layers_computation.html')}'")


def main():
    """
    Main function to demonstrate detailed computation with value substitution
    """
    parser = argparse.ArgumentParser(description="PyTorch Layer Computation Detailed Example")
    parser.add_argument("--mode", type=str, default="all", 
                      choices=["conv", "all"],
                      help="Demo mode to run (conv, all)")
    
    args = parser.parse_args()
    mode = args.mode
    
    print("Running PyTorch Layer Detailed Computation Example")
    print(f"Selected mode: {mode}")
    
    if mode == "conv" or mode == "all":
        analyzer = demo_conv_layers()
    
    if mode == "all":
        analyze_all_layers(analyzer)
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
