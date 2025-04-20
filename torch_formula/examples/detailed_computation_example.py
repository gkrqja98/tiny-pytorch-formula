"""
Simple example demonstrating the detailed computation functionality
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Add the parent directory to the path to import our package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_formula.utils.detailed_computation import (
    get_detailed_computation,
    get_computation_markdown,
    format_tensor_as_markdown_table
)


def create_and_compute_conv2d():
    """Create a Conv2d layer and compute its forward pass with detailed explanation"""
    # Create a simple Conv2d layer
    conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
    
    # Initialize weights for predictable results
    torch.manual_seed(42)
    nn.init.ones_(conv.weight)  # Set all weights to 1
    nn.init.zeros_(conv.bias)   # Set all biases to 0
    
    # Create a simple input tensor (1x1x4x4) filled with incremental values
    input_tensor = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    print("Input tensor:")
    print(input_tensor.squeeze())
    
    # Compute the output
    output_tensor = conv(input_tensor)
    print(f"\nOutput tensor shape: {output_tensor.shape}")
    print("Output tensor (channel 0):")
    print(output_tensor[0, 0].detach())
    print("Output tensor (channel 1):")
    print(output_tensor[0, 1].detach())
    
    # Get detailed computation
    detailed_comp = get_detailed_computation(conv, input_tensor, output_tensor)
    
    # Print the detailed computation as markdown
    markdown = get_computation_markdown(conv, input_tensor, output_tensor)
    print("\nDetailed computation:")
    print(markdown)
    
    # Save the markdown to a file
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "conv2d_computation.md"), "w") as f:
        f.write("# Detailed Conv2d Computation\n\n")
        f.write(markdown)
    
    print(f"Detailed computation saved to {os.path.join(output_dir, 'conv2d_computation.md')}")


def create_and_compute_maxpool():
    """Create a MaxPool2d layer and compute its forward pass with detailed explanation"""
    # Create a simple MaxPool2d layer
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Create a simple input tensor (1x1x4x4) filled with incremental values
    input_tensor = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    print("\nInput tensor for MaxPool2d:")
    print(input_tensor.squeeze())
    
    # Compute the output
    output_tensor = pool(input_tensor)
    print(f"\nOutput tensor shape: {output_tensor.shape}")
    print("Output tensor:")
    print(output_tensor.squeeze().detach())
    
    # Get detailed computation
    markdown = get_computation_markdown(pool, input_tensor, output_tensor)
    print("\nDetailed computation:")
    print(markdown)
    
    # Save the markdown to a file
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    
    with open(os.path.join(output_dir, "maxpool_computation.md"), "w") as f:
        f.write("# Detailed MaxPool2d Computation\n\n")
        f.write(markdown)
    
    print(f"Detailed computation saved to {os.path.join(output_dir, 'maxpool_computation.md')}")


def create_and_compute_linear():
    """Create a Linear layer and compute its forward pass with detailed explanation"""
    # Create a simple Linear layer
    linear = nn.Linear(in_features=4, out_features=2)
    
    # Initialize weights for predictable results
    torch.manual_seed(42)
    nn.init.ones_(linear.weight)  # Set all weights to 1
    nn.init.zeros_(linear.bias)   # Set all biases to 0
    
    # Create a simple input tensor (1x4) filled with incremental values
    input_tensor = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    print("\nInput tensor for Linear:")
    print(input_tensor.squeeze())
    
    # Compute the output
    output_tensor = linear(input_tensor)
    print(f"\nOutput tensor shape: {output_tensor.shape}")
    print("Output tensor:")
    print(output_tensor.detach())
    
    # Get detailed computation
    markdown = get_computation_markdown(linear, input_tensor, output_tensor)
    print("\nDetailed computation:")
    print(markdown)
    
    # Save the markdown to a file
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    
    with open(os.path.join(output_dir, "linear_computation.md"), "w") as f:
        f.write("# Detailed Linear Computation\n\n")
        f.write(markdown)
    
    print(f"Detailed computation saved to {os.path.join(output_dir, 'linear_computation.md')}")


def main():
    """Main function to demonstrate detailed computation"""
    print("Demonstrating detailed computation for PyTorch layers...\n")
    
    create_and_compute_conv2d()
    create_and_compute_maxpool()
    create_and_compute_linear()
    
    print("\nAll computations completed!")


if __name__ == "__main__":
    main()
