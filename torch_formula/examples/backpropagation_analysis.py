"""
Example script demonstrating backpropagation analysis in PyTorch models
Shows detailed computation of gradients for different layers with full forward and backward passes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Add the parent directory to the path to import our package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_formula.models import TinyCNN
from torch_formula.core.model_analyzer import ModelAnalyzer
from torch_formula.utils.detailed_computation import (
    get_computation_markdown, 
    get_backward_markdown,
    format_tensor_as_markdown_table
)
from torch_formula.viz.computation_graph import create_layerwise_graph, visualize_layerwise_graph


def setup_custom_model():
    """Create and initialize a model with custom weights for better visualization"""
    # Create TinyCNN model
    model = TinyCNN()
    
    # Set specific weights for clearer demonstration
    torch.manual_seed(42)
    
    # Set specific values for the conv1 layer filters
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
        
        # Set some specific weights for conv2 layer
        model.conv2.weight[0, 0, 0, 0] = 0.5
        model.conv2.weight[0, 1, 0, 0] = 0.3
        model.conv2.bias[0] = 0.05
    
    return model


def create_patterned_input(size=8):
    """Create an input tensor with an easily recognizable pattern"""
    # Create input tensor with specific pattern
    input_tensor = torch.zeros(1, 1, size, size)
    
    # Fill with specific values for better visualization
    for i in range(size):
        for j in range(size):
            # Create a pattern: gradual increase with some variations
            val = (i + j) / (2*size - 2)  # normalized between 0 and 1
            
            # Add some variations
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


def analyze_layer_forward_backward(model, input_tensor, layer_name, position, output_dir):
    """
    Analyze both forward and backward computation for a specific layer position
    
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
    
    print(f"\n=== Analyzing {layer_name} at position {position} (Forward and Backward) ===")
    
    # Get detailed forward computation
    forward_md = analyzer.get_detailed_computation(layer_name, position=position, format="markdown")
    
    # Get detailed backward computation
    backward_md = analyzer.get_detailed_backward(layer_name, position=position, format="markdown")
    
    # Get layer type
    layer_type = "unknown"
    for name, module in model.named_modules():
        if name == layer_name:
            layer_type = type(module).__name__
    
    # Create output file
    position_str = "_".join(map(str, position))
    filename = f"{layer_name}_{layer_type}_pos{position_str}_forward_backward.md"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        f.write(f"# Complete Forward and Backward Analysis for {layer_type} Layer '{layer_name}'\n\n")
        f.write(f"## Position: {position}\n\n")
        f.write("## Forward Pass\n\n")
        f.write(forward_md)
        f.write("\n\n## Backward Pass\n\n")
        f.write(backward_md)
    
    print(f"Forward and backward analysis saved to '{filepath}'")
    
    return analyzer


def visualize_gradient_flow(model, input_tensor, output_dir):
    """
    Visualize gradient flow through the model
    
    Args:
        model: The PyTorch model
        input_tensor: Input tensor to the model
        output_dir: Directory to save the output
    """
    print("\n=== Visualizing Gradient Flow ===")
    
    # Run forward and backward passes manually
    input_tensor.requires_grad_(True)
    
    # Forward pass
    x = model.conv1(input_tensor)
    x1 = F.relu(x)
    x2 = model.pool1(x1)
    x3 = model.conv2(x2)
    x4 = F.relu(x3)
    x5 = model.pool2(x4)
    x6 = model.conv3(x5)
    x7 = F.relu(x6)
    output = model.conv_out(x7)
    output = output.squeeze(3).squeeze(2)
    
    # Create a target and loss
    target = torch.zeros_like(output)
    target[0, 0] = 1.0  # Set a specific target
    
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    
    # Backward pass
    loss.backward()
    
    # Collect gradients
    gradients = {
        'input': input_tensor.grad.detach(),
        'conv1': model.conv1.weight.grad.detach(),
        'conv2': model.conv2.weight.grad.detach(),
        'conv3': model.conv3.weight.grad.detach(),
        'conv_out': model.conv_out.weight.grad.detach()
    }
    
    # Visualize gradient norms across layers
    plt.figure(figsize=(10, 6))
    
    # Compute gradient norms for each layer
    layer_names = []
    grad_norms = []
    
    for name, grad in gradients.items():
        layer_names.append(name)
        norm = torch.norm(grad).item()
        grad_norms.append(norm)
    
    # Create bar plot
    bars = plt.bar(layer_names, grad_norms, color='skyblue')
    plt.title('Gradient Norms Across Layers', fontsize=14)
    plt.ylabel('Gradient Norm', fontsize=12)
    plt.xlabel('Layer', fontsize=12)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'gradient_norms.png'), dpi=300, bbox_inches='tight')
    print(f"Gradient norms visualization saved to '{os.path.join(output_dir, 'gradient_norms.png')}'")
    
    # Create a detailed markdown table with gradient statistics
    md_path = os.path.join(output_dir, 'gradient_statistics.md')
    with open(md_path, 'w') as f:
        f.write("# Gradient Statistics Across Model Layers\n\n")
        
        f.write("| Layer | Mean | Std | Min | Max | Norm |\n")
        f.write("| ----- | ---- | --- | --- | --- | ---- |\n")
        
        for name, grad in gradients.items():
            mean = torch.mean(grad).item()
            std = torch.std(grad).item()
            min_val = torch.min(grad).item()
            max_val = torch.max(grad).item()
            norm = torch.norm(grad).item()
            
            f.write(f"| {name} | {mean:.6f} | {std:.6f} | {min_val:.6f} | {max_val:.6f} | {norm:.6f} |\n")
    
    print(f"Gradient statistics saved to '{md_path}'")
    
    return gradients


def analyze_weight_updates(model, input_tensor, learning_rate, output_dir):
    """
    Analyze how weights would be updated after backpropagation
    
    Args:
        model: The PyTorch model
        input_tensor: Input tensor to the model
        learning_rate: Learning rate for weight updates
        output_dir: Directory to save the output
    """
    print("\n=== Analyzing Weight Updates ===")
    
    # Make a copy of the original weights
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.detach().clone()
    
    # Forward and backward pass
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    
    # Create a target and loss
    target = torch.zeros_like(output)
    target[0, 0] = 1.0  # Set a specific target
    
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    
    # Compute gradients
    loss.backward()
    
    # Compute and record weight updates
    updates_md = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Compute the weight update
            update = -learning_rate * param.grad.detach()
            
            # Calculate new weights
            new_weight = param.detach() + update
            
            # Prepare markdown content for this parameter
            if param.dim() <= 2:  # For weights that can be easily displayed
                md = f"### Weight Updates for {name}\n\n"
                
                md += "#### Original Weights\n\n"
                md += format_tensor_as_markdown_table(param.detach(), f"Original {name}")
                
                md += "#### Gradients\n\n"
                md += format_tensor_as_markdown_table(param.grad.detach(), f"Gradients for {name}")
                
                md += "#### Updates (learning_rate = {:.4f})\n\n".format(learning_rate)
                md += format_tensor_as_markdown_table(update, f"Updates for {name}")
                
                md += "#### New Weights\n\n"
                md += format_tensor_as_markdown_table(new_weight, f"New {name}")
                
                updates_md[name] = md
    
    # Combine all updates into one markdown file
    with open(os.path.join(output_dir, 'weight_updates.md'), 'w') as f:
        f.write(f"# Weight Update Analysis (Learning Rate: {learning_rate})\n\n")
        
        for name, md in updates_md.items():
            f.write(md + "\n\n")
    
    print(f"Weight update analysis saved to '{os.path.join(output_dir, 'weight_updates.md')}'")
    
    # Restore original weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_weights:
                param.copy_(original_weights[name])


def analyze_full_model_backpropagation(model, input_tensor, output_dir):
    """
    Analyze backpropagation through the entire model
    
    Args:
        model: The PyTorch model
        input_tensor: Input tensor to the model
        output_dir: Directory to save the output
    """
    print("\n=== Analyzing Full Model Backpropagation ===")
    
    # Create analyzer
    analyzer = ModelAnalyzer(model, input_tensor.shape)
    output, loss = analyzer.run_model(input_tensor)
    
    # Get full model summary with both forward and backward details
    all_layers_md = analyzer.get_all_layers_summary(format="markdown", include_detailed_computation=True)
    
    # Save the complete analysis
    with open(os.path.join(output_dir, 'full_model_backpropagation.md'), 'w') as f:
        f.write("# Complete Forward and Backward Analysis of TinyCNN Model\n\n")
        f.write(f"## Input Shape: {input_tensor.shape}\n")
        f.write(f"## Output Shape: {output.shape}\n")
        f.write(f"## Loss Value: {loss.item():.6f}\n\n")
        f.write(all_layers_md)
    
    print(f"Full model backpropagation analysis saved to '{os.path.join(output_dir, 'full_model_backpropagation.md')}'")
    
    # Create HTML version with pretty formatting
    all_layers_html = analyzer.get_all_layers_summary(format="html", include_detailed_computation=True)
    
    with open(os.path.join(output_dir, 'full_model_backpropagation.html'), 'w') as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write("<title>TinyCNN Model Forward and Backward Analysis</title>\n")
        f.write("<meta charset=\"UTF-8\">\n")
        f.write("<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML'></script>\n")
        f.write("<style>\n")
        f.write("body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }\n")
        f.write("h1 { color: #2c3e50; }\n")
        f.write("h2 { color: #3498db; border-bottom: 1px solid #3498db; padding-bottom: 5px; }\n")
        f.write("h3 { color: #2980b9; }\n")
        f.write("h4 { color: #27ae60; }\n")
        f.write(".layer-summary { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }\n")
        f.write(".forward-pass { border-left: 5px solid #2ecc71; padding-left: 15px; }\n")
        f.write(".backward-pass { border-left: 5px solid #e74c3c; padding-left: 15px; }\n")
        f.write(".tensor-table { border-collapse: collapse; margin: 10px 0; width: auto; }\n")
        f.write(".tensor-table th, .tensor-table td { border: 1px solid #ddd; padding: 8px; text-align: right; }\n")
        f.write(".tensor-table th { background-color: #f2f2f2; }\n")
        f.write(".formula { background-color: #f9f9f9; padding: 10px; border-radius: 5px; overflow-x: auto; }\n")
        f.write("</style>\n")
        f.write("</head>\n<body>\n")
        f.write("<h1>Complete Forward and Backward Analysis of TinyCNN Model</h1>\n")
        f.write(f"<p><strong>Input Shape:</strong> {input_tensor.shape}</p>\n")
        f.write(f"<p><strong>Output Shape:</strong> {output.shape}</p>\n")
        f.write(f"<p><strong>Loss Value:</strong> {loss.item():.6f}</p>\n")
        f.write(all_layers_html)
        f.write("</body>\n</html>")
    
    print(f"HTML format full model analysis saved to '{os.path.join(output_dir, 'full_model_backpropagation.html')}'")
    
    # Create computation graph visualization
    G, forward_layers, backward_layers = create_layerwise_graph(model, input_tensor.shape)
    fig = visualize_layerwise_graph(G, forward_layers, backward_layers, 
                                   os.path.join(output_dir, 'backpropagation_flow'))
    
    print(f"Backpropagation flow visualization saved to '{os.path.join(output_dir, 'backpropagation_flow.png')}'")
    
    return analyzer


def main():
    """Main function to demonstrate backpropagation analysis"""
    parser = argparse.ArgumentParser(description="PyTorch Backpropagation Analysis Example")
    parser.add_argument("--mode", type=str, default="all", 
                       choices=["layer", "gradient", "update", "full", "all"],
                       help="Analysis mode to run")
    parser.add_argument("--lr", type=float, default=0.01, 
                       help="Learning rate for weight update analysis")
    
    args = parser.parse_args()
    mode = args.mode
    learning_rate = args.lr
    
    print("Running PyTorch Backpropagation Analysis Example")
    print(f"Selected mode: {mode}")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "backpropagation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model and input tensor
    model = setup_custom_model()
    input_tensor = create_patterned_input(size=8)
    
    if mode == "layer" or mode == "all":
        # Analyze specific layers
        # Conv2d layer
        analyze_layer_forward_backward(model, input_tensor, "conv1", (0, 0, 4, 4), output_dir)
        
        # MaxPool2d layer - run model forward to get intermediate tensors
        with torch.no_grad():
            x = model.conv1(input_tensor)
            x = F.relu(x)
        # Analyze MaxPool2d
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        pool_output = pool(x)
        
        # Make sure tensors require grad for backward pass
        x.requires_grad_(True)
        pool_output.requires_grad_(True)
        
        # Create a target and loss
        target = torch.ones_like(pool_output)
        loss_fn = nn.MSELoss()
        loss = loss_fn(pool_output, target)
        loss.backward()
        
        # Use lower-level API directly for MaxPool2d
        from torch_formula.utils.detailed_computation import (
            get_detailed_backward, format_backward_as_markdown
        )
        backward_comp = get_detailed_backward(pool, x, pool_output, pool_output.grad, (0, 0, 2, 2))
        backward_md = format_backward_as_markdown(backward_comp)
        
        with open(os.path.join(output_dir, 'maxpool_backward.md'), 'w') as f:
            f.write("# Detailed Backward Computation for MaxPool2d Layer\n\n")
            f.write("## Position: (0, 0, 2, 2)\n\n")
            f.write(backward_md)
        
        print(f"MaxPool2d backward analysis saved to '{os.path.join(output_dir, 'maxpool_backward.md')}'")
    
    if mode == "gradient" or mode == "all":
        # Visualize gradient flow
        visualize_gradient_flow(model, input_tensor.clone(), output_dir)
    
    if mode == "update" or mode == "all":
        # Analyze weight updates
        analyze_weight_updates(model, input_tensor.clone(), learning_rate, output_dir)
    
    if mode == "full" or mode == "all":
        # Analyze full model backpropagation
        analyze_full_model_backpropagation(model, input_tensor, output_dir)
    
    print("\nBackpropagation analysis completed!")


if __name__ == "__main__":
    main()
