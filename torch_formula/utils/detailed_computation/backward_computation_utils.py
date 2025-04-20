"""
Utilities for detailed computation of neural network backward operations
Provides functions to:
1. Calculate gradients for different layer operations
2. Visualize backward computation steps
3. Format the computation steps in markdown/LaTeX
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import math


def conv2d_detailed_backward(
    module: nn.Conv2d,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    grad_output: torch.Tensor,
    position: Optional[Tuple[int, ...]] = None
) -> Dict[str, Any]:
    """
    Generate detailed backward computation for Conv2d operation
    
    Args:
        module: The Conv2d module
        input_tensor: Input tensor to the module
        output_tensor: Output tensor from the module
        grad_output: Gradient of the loss with respect to the output
        position: Position for which to generate the computation (batch, out_channel, out_y, out_x)
    
    Returns:
        Dictionary with backward computation details
    """
    # Get module parameters
    weight = module.weight.detach()
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
    
    # Get grad_output at position
    grad_output_val = grad_output[batch_idx, out_c, out_y, out_x].item()
    
    # Calculate input region indices for this output position
    in_y_start = out_y * stride[0] - padding[0]
    in_x_start = out_x * stride[1] - padding[1]
    
    # Calculate gradients for input
    # For each position in the receptive field, compute gradient contribution
    input_grad_positions = []
    input_grad_formulas = []
    input_grad_values = []
    input_grad_total = {}  # (y, x, channel) -> gradient value
    
    for c_in in range(module.in_channels):
        for ky in range(kernel_size[0]):
            for kx in range(kernel_size[1]):
                in_y = in_y_start + ky
                in_x = in_x_start + kx
                
                # Skip if outside input bounds
                if not (0 <= in_y < input_tensor.shape[2] and 0 <= in_x < input_tensor.shape[3]):
                    continue
                
                # Get corresponding weight
                weight_val = weight[out_c, c_in, ky, kx].item()
                
                # Calculate gradient contribution to this input position
                grad_contribution = grad_output_val * weight_val
                
                # Track position, formula, and value
                input_grad_positions.append((in_y, in_x, c_in))
                input_grad_formulas.append(f"({grad_output_val:.4f} \\times {weight_val:.4f})")
                input_grad_values.append(grad_contribution)
                
                # Accumulate in total gradients
                pos_key = (in_y, in_x, c_in)
                if pos_key in input_grad_total:
                    input_grad_total[pos_key] += grad_contribution
                else:
                    input_grad_total[pos_key] = grad_contribution
    
    # Calculate gradients for weights
    # For each weight, its gradient is: grad_output * corresponding input value
    weight_grad_positions = []
    weight_grad_formulas = []
    weight_grad_values = []
    
    for c_in in range(module.in_channels):
        for ky in range(kernel_size[0]):
            for kx in range(kernel_size[1]):
                in_y = in_y_start + ky
                in_x = in_x_start + kx
                
                # Get input value (0 if outside bounds)
                if 0 <= in_y < input_tensor.shape[2] and 0 <= in_x < input_tensor.shape[3]:
                    input_val = input_tensor[batch_idx, c_in, in_y, in_x].item()
                else:
                    input_val = 0.0
                
                # Calculate gradient for this weight
                weight_grad = grad_output_val * input_val
                
                # Track position, formula, and value
                weight_grad_positions.append((out_c, c_in, ky, kx))
                weight_grad_formulas.append(f"({grad_output_val:.4f} \\times {input_val:.4f})")
                weight_grad_values.append(weight_grad)
    
    # Calculate gradient for bias
    bias_grad = grad_output_val
    
    # Prepare return information
    return {
        'module_type': 'Conv2d',
        'position': position,
        'grad_output_val': grad_output_val,
        'filter_weights': weight[out_c].detach().cpu().numpy(),
        'input_region': {
            'y_start': in_y_start,
            'x_start': in_x_start,
            'y_end': in_y_start + kernel_size[0],
            'x_end': in_x_start + kernel_size[1]
        },
        'input_grad': {
            'positions': input_grad_positions,
            'formulas': input_grad_formulas,
            'values': input_grad_values,
            'total': input_grad_total
        },
        'weight_grad': {
            'positions': weight_grad_positions,
            'formulas': weight_grad_formulas,
            'values': weight_grad_values
        },
        'bias_grad': bias_grad,
        'general_input_grad_formula': "\\frac{\\partial L}{\\partial x_{n,c_{in},h_{in},w_{in}}} = \\sum_{c_{out}} \\sum_{k_h=0}^{K_h-1} \\sum_{k_w=0}^{K_w-1} \\frac{\\partial L}{\\partial y_{n,c_{out},h_{out},w_{out}}} \\cdot w_{c_{out},c_{in},k_h,k_w}",
        'general_weight_grad_formula': "\\frac{\\partial L}{\\partial w_{c_{out},c_{in},k_h,k_w}} = \\sum_{n} \\sum_{h_{out},w_{out}} \\frac{\\partial L}{\\partial y_{n,c_{out},h_{out},w_{out}}} \\cdot x_{n,c_{in},h_{in}+k_h,w_{in}+k_w}",
        'general_bias_grad_formula': "\\frac{\\partial L}{\\partial b_{c_{out}}} = \\sum_{n} \\sum_{h_{out},w_{out}} \\frac{\\partial L}{\\partial y_{n,c_{out},h_{out},w_{out}}}"
    }


def maxpool2d_detailed_backward(
    module: nn.MaxPool2d,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    grad_output: torch.Tensor,
    position: Optional[Tuple[int, ...]] = None
) -> Dict[str, Any]:
    """
    Generate detailed backward computation for MaxPool2d operation
    
    Args:
        module: The MaxPool2d module
        input_tensor: Input tensor to the module
        output_tensor: Output tensor from the module
        grad_output: Gradient of the loss with respect to the output
        position: Position for which to generate the computation (batch, channel, out_y, out_x)
    
    Returns:
        Dictionary with backward computation details
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
    
    # Get grad_output at position
    grad_output_val = grad_output[batch_idx, channel, out_y, out_x].item()
    
    # Calculate input region indices
    in_y_start = out_y * stride[0] - padding[0]
    in_x_start = out_x * stride[1] - padding[1]
    
    # Get the receptive field from the input tensor
    receptive_field = torch.zeros(kernel_size[0], kernel_size[1])
    max_value = float('-inf')
    max_pos = (0, 0)
    
    # Find the max position in the receptive field
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
    
    # For MaxPool, only the max position gets the gradient, others get zero
    input_grad_positions = []
    input_grad_formulas = []
    input_grad_values = []
    input_grad_map = torch.zeros_like(receptive_field)
    
    # Set gradient only at max position
    max_y, max_x = max_pos
    in_y = in_y_start + max_y
    in_x = in_x_start + max_x
    
    # Check if max position is within bounds
    if 0 <= in_y < input_tensor.shape[2] and 0 <= in_x < input_tensor.shape[3]:
        input_grad_positions.append((in_y, in_x))
        input_grad_formulas.append(f"{grad_output_val:.4f}")  # Gradient passes through unchanged
        input_grad_values.append(grad_output_val)
        input_grad_map[max_y, max_x] = grad_output_val
    
    # Prepare return information
    return {
        'module_type': 'MaxPool2d',
        'position': position,
        'grad_output_val': grad_output_val,
        'receptive_field': receptive_field.numpy(),
        'max_position': max_pos,
        'input_region': {
            'y_start': in_y_start,
            'x_start': in_x_start,
            'y_end': in_y_start + kernel_size[0],
            'x_end': in_x_start + kernel_size[1]
        },
        'input_grad': {
            'positions': input_grad_positions,
            'formulas': input_grad_formulas,
            'values': input_grad_values,
            'map': input_grad_map.numpy()
        },
        'general_input_grad_formula': "\\frac{\\partial L}{\\partial x_{n,c,h_{in},w_{in}}} = \\begin{cases} \\frac{\\partial L}{\\partial y_{n,c,h_{out},w_{out}}} & \\text{if } x_{n,c,h_{in},w_{in}} \\text{ is max in pool} \\\\ 0 & \\text{otherwise} \\end{cases}"
    }


def avgpool2d_detailed_backward(
    module: nn.AvgPool2d,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    grad_output: torch.Tensor,
    position: Optional[Tuple[int, ...]] = None
) -> Dict[str, Any]:
    """
    Generate detailed backward computation for AvgPool2d operation
    
    Args:
        module: The AvgPool2d module
        input_tensor: Input tensor to the module
        output_tensor: Output tensor from the module
        grad_output: Gradient of the loss with respect to the output
        position: Position for which to generate the computation (batch, channel, out_y, out_x)
    
    Returns:
        Dictionary with backward computation details
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
    
    # Get grad_output at position
    grad_output_val = grad_output[batch_idx, channel, out_y, out_x].item()
    
    # Calculate input region indices
    in_y_start = out_y * stride[0] - padding[0]
    in_x_start = out_x * stride[1] - padding[1]
    
    # For AvgPool, gradient is distributed equally to all positions in the receptive field
    pool_size = kernel_size[0] * kernel_size[1]
    distributed_grad = grad_output_val / pool_size
    
    # Create map of gradients in receptive field (all equal)
    input_grad_map = torch.ones(kernel_size[0], kernel_size[1]) * distributed_grad
    
    # Record positions, formulas, and values for gradients
    input_grad_positions = []
    input_grad_formulas = []
    input_grad_values = []
    
    for ky in range(kernel_size[0]):
        y_idx = in_y_start + ky
        for kx in range(kernel_size[1]):
            x_idx = in_x_start + kx
            if 0 <= y_idx < input_tensor.shape[2] and 0 <= x_idx < input_tensor.shape[3]:
                input_grad_positions.append((y_idx, x_idx))
                input_grad_formulas.append(f"{grad_output_val:.4f} / {pool_size}")
                input_grad_values.append(distributed_grad)
    
    # Prepare return information
    return {
        'module_type': 'AvgPool2d',
        'position': position,
        'grad_output_val': grad_output_val,
        'kernel_size': kernel_size,
        'pool_size': pool_size,
        'distributed_grad': distributed_grad,
        'input_region': {
            'y_start': in_y_start,
            'x_start': in_x_start,
            'y_end': in_y_start + kernel_size[0],
            'x_end': in_x_start + kernel_size[1]
        },
        'input_grad': {
            'positions': input_grad_positions,
            'formulas': input_grad_formulas,
            'values': input_grad_values,
            'map': input_grad_map.numpy()
        },
        'general_input_grad_formula': "\\frac{\\partial L}{\\partial x_{n,c,h_{in},w_{in}}} = \\frac{1}{k_h \\cdot k_w} \\frac{\\partial L}{\\partial y_{n,c,h_{out},w_{out}}}"
    }


def linear_detailed_backward(
    module: nn.Linear,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    grad_output: torch.Tensor,
    position: Optional[Tuple[int, ...]] = None
) -> Dict[str, Any]:
    """
    Generate detailed backward computation for Linear operation
    
    Args:
        module: The Linear module
        input_tensor: Input tensor to the module
        output_tensor: Output tensor from the module
        grad_output: Gradient of the loss with respect to the output
        position: Position for which to generate the computation (batch, output_feature)
    
    Returns:
        Dictionary with backward computation details
    """
    # Get module parameters
    weight = module.weight.detach()  # Shape: [out_features, in_features]
    
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
    
    # Get grad_output at position
    grad_output_val = grad_output[batch_idx, out_feature].item()
    
    # Get input features and corresponding weights
    input_features = input_flattened[batch_idx]  # Shape: [in_features]
    
    # Calculate gradients for input
    # For each input feature, compute gradient contribution
    input_grad_positions = []
    input_grad_formulas = []
    input_grad_values = []
    
    for in_feature in range(module.in_features):
        weight_val = weight[out_feature, in_feature].item()
        
        # Gradient for input[in_feature] = grad_output * weight[out_feature, in_feature]
        grad_contribution = grad_output_val * weight_val
        
        input_grad_positions.append(in_feature)
        input_grad_formulas.append(f"({grad_output_val:.4f} \\times {weight_val:.4f})")
        input_grad_values.append(grad_contribution)
    
    # Calculate gradients for weights
    # For each weight, its gradient is: grad_output * corresponding input value
    weight_grad_positions = []
    weight_grad_formulas = []
    weight_grad_values = []
    
    for in_feature in range(module.in_features):
        input_val = input_features[in_feature].item()
        
        # Gradient for weight[out_feature, in_feature] = grad_output * input[in_feature]
        weight_grad = grad_output_val * input_val
        
        weight_grad_positions.append((out_feature, in_feature))
        weight_grad_formulas.append(f"({grad_output_val:.4f} \\times {input_val:.4f})")
        weight_grad_values.append(weight_grad)
    
    # Calculate gradient for bias
    bias_grad = grad_output_val
    
    # Prepare return information
    return {
        'module_type': 'Linear',
        'position': position,
        'grad_output_val': grad_output_val,
        'input_features': input_features.numpy(),
        'output_weights': weight[out_feature].numpy(),
        'input_grad': {
            'positions': input_grad_positions,
            'formulas': input_grad_formulas,
            'values': input_grad_values
        },
        'weight_grad': {
            'positions': weight_grad_positions,
            'formulas': weight_grad_formulas,
            'values': weight_grad_values
        },
        'bias_grad': bias_grad,
        'general_input_grad_formula': "\\frac{\\partial L}{\\partial x_{n,i}} = \\sum_{j} \\frac{\\partial L}{\\partial y_{n,j}} \\cdot w_{j,i}",
        'general_weight_grad_formula': "\\frac{\\partial L}{\\partial w_{j,i}} = \\sum_{n} \\frac{\\partial L}{\\partial y_{n,j}} \\cdot x_{n,i}",
        'general_bias_grad_formula': "\\frac{\\partial L}{\\partial b_{j}} = \\sum_{n} \\frac{\\partial L}{\\partial y_{n,j}}"
    }


def relu_detailed_backward(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    grad_output: torch.Tensor,
    position: Optional[Tuple[int, ...]] = None
) -> Dict[str, Any]:
    """
    Generate detailed backward computation for ReLU operation
    
    Args:
        input_tensor: Input tensor to the ReLU operation
        output_tensor: Output tensor from the ReLU operation
        grad_output: Gradient of the loss with respect to the output
        position: Position for which to generate the computation
    
    Returns:
        Dictionary with backward computation details
    """
    # For ReLU, position format depends on input/output shape
    if position is None:
        if input_tensor.dim() == 4:  # Conv input: [batch, channels, height, width]
            batch_idx = 0
            channel = 0
            y = min(1, input_tensor.shape[2] - 1)
            x = min(1, input_tensor.shape[3] - 1)
            position = (batch_idx, channel, y, x)
        elif input_tensor.dim() == 2:  # Linear input: [batch, features]
            batch_idx = 0
            feature = 0
            position = (batch_idx, feature)
        else:
            # Default to first element
            position = tuple([0] * input_tensor.dim())
    
    # Get input, output, and grad_output at position
    if input_tensor.dim() == 4:
        batch_idx, channel, y, x = position
        input_val = input_tensor[batch_idx, channel, y, x].item()
        output_val = output_tensor[batch_idx, channel, y, x].item()
        grad_output_val = grad_output[batch_idx, channel, y, x].item()
    elif input_tensor.dim() == 2:
        batch_idx, feature = position
        input_val = input_tensor[batch_idx, feature].item()
        output_val = output_tensor[batch_idx, feature].item()
        grad_output_val = grad_output[batch_idx, feature].item()
    else:
        # Get values using position tuple
        input_val = input_tensor[position].item()
        output_val = output_tensor[position].item()
        grad_output_val = grad_output[position].item()
    
    # For ReLU: gradient is equal to upstream gradient if input > 0, else 0
    if input_val > 0:
        grad_input = grad_output_val
        formula = f"{grad_output_val:.4f}"
        explanation = "Input > 0, so gradient passes through unchanged"
    else:
        grad_input = 0.0
        formula = "0"
        explanation = "Input <= 0, so gradient is zero"
    
    # Prepare return information
    return {
        'module_type': 'ReLU',
        'position': position,
        'input_val': input_val,
        'output_val': output_val,
        'grad_output_val': grad_output_val,
        'grad_input': grad_input,
        'formula': formula,
        'explanation': explanation,
        'general_formula': "\\frac{\\partial L}{\\partial x} = \\begin{cases} \\frac{\\partial L}{\\partial y} & \\text{if } x > 0 \\\\ 0 & \\text{if } x \\leq 0 \\end{cases}"
    }


def get_detailed_backward(
    module: nn.Module,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    grad_output: torch.Tensor,
    position: Optional[Tuple[int, ...]] = None
) -> Dict[str, Any]:
    """
    Generate detailed backward computation for any PyTorch module
    
    Args:
        module: The PyTorch module
        input_tensor: Input tensor to the module
        output_tensor: Output tensor from the module
        grad_output: Gradient of the loss with respect to the output
        position: Position for which to generate the computation
    
    Returns:
        Dictionary with backward computation details
    """
    if isinstance(module, nn.Conv2d):
        return conv2d_detailed_backward(module, input_tensor, output_tensor, grad_output, position)
    elif isinstance(module, nn.Linear):
        return linear_detailed_backward(module, input_tensor, output_tensor, grad_output, position)
    elif isinstance(module, nn.MaxPool2d):
        return maxpool2d_detailed_backward(module, input_tensor, output_tensor, grad_output, position)
    elif isinstance(module, nn.AvgPool2d):
        return avgpool2d_detailed_backward(module, input_tensor, output_tensor, grad_output, position)
    elif isinstance(module, nn.ReLU) or (isinstance(module, nn.Module) and hasattr(module, 'inplace') and module.__class__.__name__ == 'ReLU'):
        return relu_detailed_backward(input_tensor, output_tensor, grad_output, position)
    else:
        return {
            'module_type': type(module).__name__,
            'error': f"Detailed backward computation not implemented for {type(module).__name__}"
        }
