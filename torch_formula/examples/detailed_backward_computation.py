"""
Example script demonstrating detailed backward computation in PyTorch layers
Shows detailed gradient calculation with value substitution
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
    get_backward_markdown,
    format_backward_as_markdown,
    get_detailed_backward
)


def create_simple_model():
    """
    Create a simple model with custom weights for better demonstration
    """
    model = TinyCNN()
    
    # Set specific weights for clearer demonstration
    torch.manual_seed(42)
    
    # Set specific values for the first filter (in_channels=1, out_channels=2)
    with torch.no_grad():
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


def analyze_backward_at_position(model, input_tensor, layer_name, position, output_dir):
    """
    Analyze backward computation at a specific position in a specific layer
    
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
    
    print(f"\n=== Analyzing Backward Pass: {layer_name} layer, position {position} ===")
    
    # Get detailed backward computation for the specific position
    backward_md = analyzer.get_detailed_backward(layer_name, position=position, format="markdown")
    
    # Save to file
    layer_type = "unknown"
    for name, module in model.named_modules():
        if name == layer_name:
            layer_type = type(module).__name__
    
    position_str = "_".join(map(str, position))
    filename = f"{layer_name}_{layer_type}_pos{position_str}_backward.md"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        f.write(f"# Detailed Backward Computation for {layer_type} Layer '{layer_name}'\n\n")
        f.write(f"## Position: {position}\n\n")
        f.write(backward_md)
    
    print(f"Detailed backward computation results saved to '{filepath}'")
    
    return backward_md, analyzer


def demo_conv2d_backward():
    """
    Demonstrate detailed backward computation for Conv2d layers
    """
    print("\n" + "="*50)
    print("Conv2d Layer Detailed Backward Computation Demonstration")
    print("="*50)
    
    # Create model with custom weights
    model = create_simple_model()
    
    # Generate input tensor
    input_tensor = generate_input_tensor()
    
    print(f"Input tensor shape: {input_tensor.shape}")
    print("Input tensor sample values (central 4x4 region):")
    print(input_tensor[0, 0, 2:6, 2:6])
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "backward_computation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze conv1 layer at a specific position
    # Position: (batch, output_channel, height, width)
    conv1_positions = [
        (0, 0, 4, 4),  # First filter, central position
        (0, 1, 4, 4)   # Second filter, central position
    ]
    
    for position in conv1_positions:
        backward_md, analyzer = analyze_backward_at_position(
            model, input_tensor, "conv1", position, output_dir
        )
    
    return analyzer


def demo_maxpool_backward():
    """
    Demonstrate detailed backward computation for MaxPool2d layers
    """
    print("\n" + "="*50)
    print("MaxPool2d Layer Detailed Backward Computation Demonstration")
    print("="*50)
    
    # Create model with custom weights
    model = create_simple_model()
    
    # Generate input tensor
    input_tensor = generate_input_tensor()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "backward_computation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the model up to pool1 to get the activations
    with torch.no_grad():
        x = model.conv1(input_tensor)
        x = F.relu(x)
    
    # Position: (batch, channel, height, width)
    positions = [
        (0, 0, 2, 2),  # Position in the center of the output
        (0, 1, 1, 1)   # Another position
    ]
    
    # Create a standalone maxpool layer to analyze
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    output = pool(x)
    
    # Compute gradients
    output.requires_grad_(True)
    
    # Create a target to compute gradients
    target = torch.ones_like(output)
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    loss.backward()
    
    # Now analyze the gradients directly
    for position in positions:
        # Get backward computation for the specific position
        backward_comp = get_detailed_backward(pool, x, output, output.grad, position)
        backward_md = format_backward_as_markdown(backward_comp)
        
        # Save to file
        position_str = "_".join(map(str, position))
        filename = f"maxpool_MaxPool2d_pos{position_str}_backward.md"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w") as f:
            f.write(f"# Detailed Backward Computation for MaxPool2d Layer\n\n")
            f.write(f"## Position: {position}\n\n")
            f.write(backward_md)
        
        print(f"MaxPool2d detailed backward results saved to '{filepath}'")


def demo_relu_backward():
    """
    Demonstrate detailed backward computation for ReLU layers
    """
    print("\n" + "="*50)
    print("ReLU Layer Detailed Backward Computation Demonstration")
    print("="*50)
    
    # Create a simple input tensor with mixed positive and negative values
    input_tensor = torch.tensor([[[-1.0, 2.0], [3.0, -4.0]]], requires_grad=True)
    
    # Create a ReLU layer
    relu = nn.ReLU()
    
    # Forward pass
    output = relu(input_tensor)
    
    # Backward pass (create fake gradient, all ones)
    grad_output = torch.ones_like(output)
    output.backward(grad_output)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "backward_computation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze different positions (both positive and negative inputs)
    positions = [
        (0, 0, 0, 0),  # Negative input, should have zero gradient
        (0, 0, 0, 1),  # Positive input, should have gradient passed through
        (0, 0, 1, 0),  # Positive input, should have gradient passed through
        (0, 0, 1, 1)   # Negative input, should have zero gradient
    ]
    
    for position in positions:
        # Get backward computation
        backward_comp = get_detailed_backward(relu, input_tensor, output, grad_output, position)
        backward_md = format_backward_as_markdown(backward_comp)
        
        # Save to file
        position_str = "_".join(map(str, position))
        filename = f"relu_ReLU_pos{position_str}_backward.md"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w") as f:
            f.write(f"# Detailed Backward Computation for ReLU Layer\n\n")
            f.write(f"## Position: {position}\n\n")
            f.write(backward_md)
        
        print(f"ReLU detailed backward results saved to '{filepath}'")


def analyze_full_model_backward():
    """
    Analyze the backward computation of the full model
    """
    print("\n" + "="*50)
    print("Full Model Backward Computation Analysis")
    print("="*50)
    
    # Create model with custom weights
    model = create_simple_model()
    
    # Generate input tensor
    input_tensor = generate_input_tensor()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "backward_computation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run model analyzer
    analyzer = ModelAnalyzer(model, input_tensor.shape)
    output, loss = analyzer.run_model(input_tensor)
    
    print(f"Model output shape: {output.shape}")
    print(f"Loss value: {loss.item():.6f}")
    
    # Get summaries with both forward and backward computation
    all_layers_md = analyzer.get_all_layers_summary(format="markdown", include_detailed_computation=True)
    
    # Save to markdown file
    with open(os.path.join(output_dir, "all_layers_backward.md"), "w") as f:
        f.write("# Detailed Analysis of All Layers in TinyCNN Model (Forward and Backward)\n\n")
        f.write(all_layers_md)
    
    print(f"Full model analysis saved to '{os.path.join(output_dir, 'all_layers_backward.md')}'")
    
    # Also create HTML version
    all_layers_html = analyzer.get_all_layers_summary(format="html", include_detailed_computation=True)
    
    with open(os.path.join(output_dir, "all_layers_backward.html"), "w") as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write("<title>TinyCNN Model Forward and Backward Analysis</title>\n")
        f.write("<meta charset=\"UTF-8\">\n")
        f.write("<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML'></script>\n")
        f.write("<style>\n")
        f.write("body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }\n")
        f.write(".section { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }\n")
        f.write(".tensor-table { border-collapse: collapse; margin: 10px 0; width: auto; }\n")
        f.write(".tensor-table th, .tensor-table td { border: 1px solid #ddd; padding: 8px; text-align: right; }\n")
        f.write(".tensor-table th { background-color: #f2f2f2; }\n")
        f.write(".forward-computation { border-left: 3px solid #4CAF50; padding-left: 15px; margin: 20px 0; }\n")
        f.write(".backward-computation { border-left: 3px solid #f44336; padding-left: 15px; margin: 20px 0; }\n")
        f.write("</style>\n")
        f.write("</head>\n<body>\n")
        f.write("<h1>Detailed Analysis of All Layers in TinyCNN Model (Forward and Backward)</h1>\n")
        f.write(all_layers_html)
        f.write("</body>\n</html>")
    
    print(f"HTML format full analysis saved to '{os.path.join(output_dir, 'all_layers_backward.html')}'")
    
    return analyzer


def main():
    """
    Main function to demonstrate detailed backward computation
    """
    parser = argparse.ArgumentParser(description="PyTorch Layer Backward Computation Detailed Example")
    parser.add_argument("--mode", type=str, default="all", 
                      choices=["conv", "maxpool", "relu", "all"],
                      help="Demo mode to run (conv, maxpool, relu, all)")
    
    args = parser.parse_args()
    mode = args.mode
    
    print("Running PyTorch Layer Detailed Backward Computation Example")
    print(f"Selected mode: {mode}")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "backward_computation")
    os.makedirs(output_dir, exist_ok=True)
    
    if mode == "conv" or mode == "all":
        demo_conv2d_backward()
    
    if mode == "maxpool" or mode == "all":
        demo_maxpool_backward()
    
    if mode == "relu" or mode == "all":
        demo_relu_backward()
    
    if mode == "all":
        analyze_full_model_backward()
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
