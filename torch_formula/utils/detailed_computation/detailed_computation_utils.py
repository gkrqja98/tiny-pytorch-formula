"""
Utilities for detailed computation of neural network operations
Provides functions to:
1. Sample regions from tensors
2. Calculate actual values for different layer operations
3. Format the computation steps in markdown/LaTeX
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import math


def sample_region(tensor: torch.Tensor, 
                 position: Optional[Tuple[int, ...]] = None, 
                 region_size: Optional[Tuple[int, ...]] = None,
                 channel_first: bool = True) -> torch.Tensor:
    """
    Sample a region from a tensor
    
    Args:
        tensor: The input tensor
        position: Position (indices) from where to start sampling. If None, use the center
        region_size: Size of the region to sample. If None, choose based on tensor dim
        channel_first: Whether the tensor has channels as the first dimension after batch
        
    Returns:
        Sampled tensor region
    """
    tensor = tensor.detach()
    dim = tensor.dim()
    shape = tensor.shape
    
    if dim == 2:  # Linear layer input/output (batch, features)
        if region_size is None:
            # For 2D tensors, limit to a reasonable number of features
            max_features = min(5, shape[1])
            region_size = (1, max_features)
        
        if position is None:
            # Default to first sample, first features
            position = (0, 0)
        
        # Ensure we don't go out of bounds
        end_positions = [min(p + s, shape[i]) for i, (p, s) in enumerate(zip(position, region_size))]
        return tensor[position[0]:end_positions[0], position[1]:end_positions[1]]
        
    elif dim == 4:  # Conv/Pool layer input/output (batch, channels, height, width)
        if region_size is None:
            # For 4D tensors, sample a small spatial region
            if channel_first:
                max_channels = min(3, shape[1])
                max_height = min(3, shape[2])
                max_width = min(3, shape[3])
                region_size = (1, max_channels, max_height, max_width)
            else:
                max_channels = min(3, shape[3])
                max_height = min(3, shape[1])
                max_width = min(3, shape[2])
                region_size = (1, max_height, max_width, max_channels)
        
        if position is None:
            # Default to center region
            if channel_first:
                c_start = 0
                h_start = max(0, (shape[2] - region_size[2]) // 2)
                w_start = max(0, (shape[3] - region_size[3]) // 2)
                position = (0, c_start, h_start, w_start)
            else:
                c_start = 0
                h_start = max(0, (shape[1] - region_size[1]) // 2)
                w_start = max(0, (shape[2] - region_size[2]) // 2)
                position = (0, h_start, w_start, c_start)
        
        # Ensure we don't go out of bounds
        end_positions = [min(p + s, shape[i]) for i, (p, s) in enumerate(zip(position, region_size))]
        
        if channel_first:
            return tensor[
                position[0]:end_positions[0],  # batch
                position[1]:end_positions[1],  # channels
                position[2]:end_positions[2],  # height
                position[3]:end_positions[3]   # width
            ]
        else:
            return tensor[
                position[0]:end_positions[0],  # batch
                position[1]:end_positions[1],  # height
                position[2]:end_positions[2],  # width
                position[3]:end_positions[3]   # channels
            ]
    
    # For other dimensions, return the first element
    return tensor[[0] * dim]


def conv2d_detailed_computation(
    module: nn.Conv2d,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    position: Optional[Tuple[int, ...]] = None
) -> Dict[str, Any]:
    """
    Generate detailed computation for Conv2d operation
    
    Args:
        module: The Conv2d module
        input_tensor: Input tensor to the module
        output_tensor: Output tensor from the module
        position: Position for which to generate the computation (batch, out_channel, out_y, out_x)
    
    Returns:
        Dictionary with computation details
    """
    # Get module parameters
    weight = module.weight.detach()
    bias = module.bias.detach() if module.bias is not None else None
    padding = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
    stride = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
    kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
    
    # Default position (batch, output_channel, output_y, output_x)
    if position is None:
        batch_idx = 0
        out_c = 0
        out_y = min(1, output_tensor.shape[2] - 1)
        out_x = min(1, output_tensor.shape[3] - 1)
        position = (batch_idx, out_c, out_y, out_x)
    else:
        batch_idx, out_c, out_y, out_x = position
    
    # Calculate input region indices
    in_y_start = out_y * stride[0] - padding[0]
    in_x_start = out_x * stride[1] - padding[1]
    
    in_y_end = in_y_start + kernel_size[0]
    in_x_end = in_x_start + kernel_size[1]
    
    # Get the receptive field from the input tensor
    # Handle padding by checking bounds
    receptive_field = torch.zeros(kernel_size[0], kernel_size[1], module.in_channels)
    
    for ky in range(kernel_size[0]):
        y_idx = in_y_start + ky
        for kx in range(kernel_size[1]):
            x_idx = in_x_start + kx
            for c in range(module.in_channels):
                if 0 <= y_idx < input_tensor.shape[2] and 0 <= x_idx < input_tensor.shape[3]:
                    receptive_field[ky, kx, c] = input_tensor[batch_idx, c, y_idx, x_idx]
    
    # Get the corresponding weight filter
    filter_weights = weight[out_c]  # Shape: [in_channels, kernel_height, kernel_width]
    
    # Calculate the detailed computation steps
    computation_steps = []
    value_terms = []
    sum_value = 0.0
    
    # For each input channel and each position in the kernel
    for c in range(module.in_channels):
        for ky in range(kernel_size[0]):
            for kx in range(kernel_size[1]):
                # Get input and weight values
                in_val = receptive_field[ky, kx, c].item()
                weight_val = filter_weights[c, ky, kx].item()
                
                # Calculate product
                product = in_val * weight_val
                sum_value += product
                
                # Add to computation steps
                computation_steps.append({
                    'input_channel': c,
                    'kernel_y': ky,
                    'kernel_x': kx,
                    'input_value': in_val,
                    'weight_value': weight_val,
                    'product': product
                })
                
                # Add to value terms for the formula
                value_terms.append(f"({in_val:.4f} \\times {weight_val:.4f})")
    
    # Add bias if present
    if bias is not None:
        bias_val = bias[out_c].item()
        sum_value += bias_val
        computation_steps.append({
            'bias_value': bias_val
        })
        value_terms.append(f"{bias_val:.4f}")
    
    # Verify result against actual output
    actual_output = output_tensor[batch_idx, out_c, out_y, out_x].item()
    
    # Prepare return information
    return {
        'module_type': 'Conv2d',
        'position': position,
        'receptive_field': receptive_field.numpy(),
        'filter_weights': filter_weights.numpy(),
        'computation_steps': computation_steps,
        'general_formula': "y_{n,c_{out},h_{out},w_{out}} = \\sum_{c_{in}} \\sum_{k_h=0}^{K_h-1} \\sum_{k_w=0}^{K_w-1} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \\cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}",
        'value_substitution': " + ".join(value_terms),
        'computed_result': sum_value,
        'actual_output': actual_output
    }


def maxpool2d_detailed_computation(
    module: nn.MaxPool2d,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    position: Optional[Tuple[int, ...]] = None
) -> Dict[str, Any]:
    """
    Generate detailed computation for MaxPool2d operation
    
    Args:
        module: The MaxPool2d module
        input_tensor: Input tensor to the module
        output_tensor: Output tensor from the module
        position: Position for which to generate the computation (batch, channel, out_y, out_x)
    
    Returns:
        Dictionary with computation details
    """
    # Get module parameters
    kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
    stride = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
    padding = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
    
    # Default position (batch, channel, output_y, output_x)
    if position is None:
        batch_idx = 0
        channel = 0
        out_y = min(1, output_tensor.shape[2] - 1)
        out_x = min(1, output_tensor.shape[3] - 1)
        position = (batch_idx, channel, out_y, out_x)
    else:
        batch_idx, channel, out_y, out_x = position
    
    # Calculate input region indices
    in_y_start = out_y * stride[0] - padding[0]
    in_x_start = out_x * stride[1] - padding[1]
    
    # Get the receptive field from the input tensor
    receptive_field = torch.zeros(kernel_size[0], kernel_size[1])
    max_value = float('-inf')
    max_pos = (0, 0)
    
    for ky in range(kernel_size[0]):
        y_idx = in_y_start + ky
        for kx in range(kernel_size[1]):
            x_idx = in_x_start + kx
            if 0 <= y_idx < input_tensor.shape[2] and 0 <= x_idx < input_tensor.shape[3]:
                val = input_tensor[batch_idx, channel, y_idx, x_idx].item()
                receptive_field[ky, kx] = val
                if val > max_value:
                    max_value = val
                    max_pos = (ky, kx)
    
    # Verify result against actual output
    actual_output = output_tensor[batch_idx, channel, out_y, out_x].item()
    
    # Get values from receptive field for formula
    value_terms = [f"{receptive_field[ky, kx].item():.4f}" for ky in range(kernel_size[0]) for kx in range(kernel_size[1])]
    
    # Prepare return information
    return {
        'module_type': 'MaxPool2d',
        'position': position,
        'receptive_field': receptive_field.numpy(),
        'max_position': max_pos,
        'general_formula': "y_{n,c,h_{out},w_{out}} = \\max_{0 \\leq i < k_h, 0 \\leq j < k_w} x_{n,c,h_{in}+i,w_{in}+j}",
        'value_substitution': "\\max(" + ", ".join(value_terms) + ")",
        'computed_result': max_value,
        'actual_output': actual_output
    }


def avgpool2d_detailed_computation(
    module: nn.AvgPool2d,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    position: Optional[Tuple[int, ...]] = None
) -> Dict[str, Any]:
    """
    Generate detailed computation for AvgPool2d operation
    
    Args:
        module: The AvgPool2d module
        input_tensor: Input tensor to the module
        output_tensor: Output tensor from the module
        position: Position for which to generate the computation (batch, channel, out_y, out_x)
    
    Returns:
        Dictionary with computation details
    """
    # Get module parameters
    kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
    stride = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
    padding = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
    
    # Default position (batch, channel, output_y, output_x)
    if position is None:
        batch_idx = 0
        channel = 0
        out_y = min(1, output_tensor.shape[2] - 1)
        out_x = min(1, output_tensor.shape[3] - 1)
        position = (batch_idx, channel, out_y, out_x)
    else:
        batch_idx, channel, out_y, out_x = position
    
    # Calculate input region indices
    in_y_start = out_y * stride[0] - padding[0]
    in_x_start = out_x * stride[1] - padding[1]
    
    # Get the receptive field from the input tensor
    receptive_field = torch.zeros(kernel_size[0], kernel_size[1])
    sum_value = 0.0
    count = 0
    
    for ky in range(kernel_size[0]):
        y_idx = in_y_start + ky
        for kx in range(kernel_size[1]):
            x_idx = in_x_start + kx
            if 0 <= y_idx < input_tensor.shape[2] and 0 <= x_idx < input_tensor.shape[3]:
                val = input_tensor[batch_idx, channel, y_idx, x_idx].item()
                receptive_field[ky, kx] = val
                sum_value += val
                count += 1
    
    # Calculate average
    avg_value = sum_value / (kernel_size[0] * kernel_size[1])  # Use kernel size, not count
    
    # Verify result against actual output
    actual_output = output_tensor[batch_idx, channel, out_y, out_x].item()
    
    # Get values from receptive field for formula
    value_terms = [f"{receptive_field[ky, kx].item():.4f}" for ky in range(kernel_size[0]) for kx in range(kernel_size[1])]
    
    # Prepare return information
    return {
        'module_type': 'AvgPool2d',
        'position': position,
        'receptive_field': receptive_field.numpy(),
        'kernel_size': kernel_size,
        'general_formula': "y_{n,c,h_{out},w_{out}} = \\frac{1}{k_h \\cdot k_w} \\sum_{i=0}^{k_h-1} \\sum_{j=0}^{k_w-1} x_{n,c,h_{in}+i,w_{in}+j}",
        'value_substitution': f"\\frac{1}{{{kernel_size[0]} \\times {kernel_size[1]}}} \\times (" + " + ".join(value_terms) + ")",
        'computed_result': avg_value,
        'actual_output': actual_output
    }


def linear_detailed_computation(
    module: nn.Linear,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    position: Optional[Tuple[int, ...]] = None
) -> Dict[str, Any]:
    """
    Generate detailed computation for Linear operation
    
    Args:
        module: The Linear module
        input_tensor: Input tensor to the module
        output_tensor: Output tensor from the module
        position: Position for which to generate the computation (batch, output_feature)
    
    Returns:
        Dictionary with computation details
    """
    # Get module parameters
    weight = module.weight.detach()  # Shape: [out_features, in_features]
    bias = module.bias.detach() if module.bias is not None else None
    
    # Default position (batch, output_feature)
    if position is None:
        batch_idx = 0
        out_feature = 0
        position = (batch_idx, out_feature)
    else:
        batch_idx, out_feature = position
    
    # Flatten input if needed
    if input_tensor.dim() > 2:
        input_flattened = input_tensor.view(input_tensor.size(0), -1)
    else:
        input_flattened = input_tensor
    
    # Get input features and corresponding weights
    input_features = input_flattened[batch_idx]  # Shape: [in_features]
    output_weights = weight[out_feature]  # Shape: [in_features]
    
    # Calculate the detailed computation steps
    computation_steps = []
    value_terms = []
    sum_value = 0.0
    
    # For each input feature
    for in_feature in range(module.in_features):
        # Get input and weight values
        in_val = input_features[in_feature].item()
        weight_val = output_weights[in_feature].item()
        
        # Calculate product
        product = in_val * weight_val
        sum_value += product
        
        # Add to computation steps
        computation_steps.append({
            'in_feature': in_feature,
            'input_value': in_val,
            'weight_value': weight_val,
            'product': product
        })
        
        # Add to value terms for the formula
        value_terms.append(f"({in_val:.4f} \\times {weight_val:.4f})")
    
    # Add bias if present
    if bias is not None:
        bias_val = bias[out_feature].item()
        sum_value += bias_val
        computation_steps.append({
            'bias_value': bias_val
        })
        value_terms.append(f"{bias_val:.4f}")
    
    # Verify result against actual output
    actual_output = output_tensor[batch_idx, out_feature].item()
    
    # Prepare return information
    return {
        'module_type': 'Linear',
        'position': position,
        'input_features': input_features.numpy(),
        'output_weights': output_weights.numpy(),
        'computation_steps': computation_steps,
        'general_formula': "y_{n,j} = \\sum_{i=0}^{I-1} x_{n,i} \\cdot w_{j,i} + b_j",
        'value_substitution': " + ".join(value_terms),
        'computed_result': sum_value,
        'actual_output': actual_output
    }


def get_detailed_computation(
    module: nn.Module,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    position: Optional[Tuple[int, ...]] = None
) -> Dict[str, Any]:
    """
    Generate detailed computation for a PyTorch module
    
    Args:
        module: The PyTorch module
        input_tensor: Input tensor to the module
        output_tensor: Output tensor from the module
        position: Position for which to generate the computation
    
    Returns:
        Dictionary with computation details
    """
    if isinstance(module, nn.Conv2d):
        return conv2d_detailed_computation(module, input_tensor, output_tensor, position)
    elif isinstance(module, nn.Linear):
        return linear_detailed_computation(module, input_tensor, output_tensor, position)
    elif isinstance(module, nn.MaxPool2d):
        return maxpool2d_detailed_computation(module, input_tensor, output_tensor, position)
    elif isinstance(module, nn.AvgPool2d):
        return avgpool2d_detailed_computation(module, input_tensor, output_tensor, position)
    else:
        return {
            'module_type': type(module).__name__,
            'error': f"Detailed computation not implemented for {type(module).__name__}"
        }
