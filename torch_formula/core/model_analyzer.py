"""
Core functionality for analyzing PyTorch model operations
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

class ModelAnalyzer:
    """Analyzes PyTorch model operations to provide detailed formulas and computations"""
    
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...] = None):
        """
        Initialize the model analyzer
        
        Args:
            model: PyTorch model to analyze
            input_shape: Shape of the input tensor (batch_size, channels, height, width)
        """
        self.model = model
        self.input_shape = input_shape
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        self.computation_steps = {}
        
    def register_hooks(self):
        """Register hooks to capture activations and gradients"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AvgPool2d, nn.ReLU)):
                # Forward hook to capture activations
                def get_activation(name):
                    def hook(module, input, output):
                        self.activations[name] = {
                            'input': input[0].detach(),
                            'output': output.detach()
                        }
                    return hook
                
                # Backward hook to capture gradients
                def get_gradient(name):
                    def hook(module, grad_input, grad_output):
                        self.gradients[name] = {
                            'grad_input': grad_input[0].detach() if grad_input[0] is not None else None,
                            'grad_output': grad_output[0].detach()
                        }
                    return hook
                
                # Register hooks
                forward_hook = module.register_forward_hook(get_activation(name))
                backward_hook = module.register_backward_hook(get_gradient(name))
                self.hooks.extend([forward_hook, backward_hook])
                
    def run_model(self, input_tensor=None, target=None):
        """
        Run the model forward and backward passes
        
        Args:
            input_tensor: Input tensor for the model. If None, a random tensor is created
            target: Target tensor for loss calculation. If None, a random target is created
        
        Returns:
            Tuple of (output, loss)
        """
        # Clear previous data
        self.activations = {}
        self.gradients = {}
        
        # Register hooks
        self.register_hooks()
        
        # Create input tensor if not provided
        if input_tensor is None:
            if self.input_shape is None:
                raise ValueError("Either input_tensor or input_shape must be provided")
            input_tensor = torch.randn(*self.input_shape)
        
        # Set requires_grad for backward pass
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Create target tensor if not provided
        if target is None:
            target = torch.randn_like(output)
        
        # Compute loss and run backward pass
        loss_fn = nn.MSELoss()
        loss = loss_fn(output, target)
        loss.backward()
        
        # Collect computation steps
        self._collect_computation_steps()
        
        return output, loss
    
    def _collect_computation_steps(self):
        """Collect computation steps for each layer"""
        for name, module in self.model.named_modules():
            if name in self.activations:
                # Get input and output tensors
                input_tensor = self.activations[name]['input']
                output_tensor = self.activations[name]['output']
                
                # Get gradient tensors if available
                grad_input = self.gradients.get(name, {}).get('grad_input', None)
                grad_output = self.gradients.get(name, {}).get('grad_output', None)
                
                # Get detailed forward computation
                detailed_forward = self._get_detailed_computation(module, input_tensor, output_tensor)
                
                # Create computation step information
                self.computation_steps[name] = {
                    'module_type': type(module).__name__,
                    'input_shape': input_tensor.shape,
                    'output_shape': output_tensor.shape,
                    'parameters': {
                        param_name: param.detach().numpy() 
                        for param_name, param in module.named_parameters()
                    },
                    'forward': {
                        'general_formula': self._get_general_formula(module),
                        'value_substitution': detailed_forward.get('value_substitution', self._get_value_substitution(module, input_tensor, output_tensor)),
                        'result': output_tensor.detach().numpy(),
                        'detailed_computation': detailed_forward
                    }
                }
                
                # Add backward information if available
                if grad_output is not None:
                    # Get detailed backward computation
                    detailed_backward = self._get_detailed_backward(module, input_tensor, output_tensor, grad_output)
                    
                    self.computation_steps[name]['backward'] = {
                        'general_formula': self._get_gradient_formula(module),
                        'value_substitution': detailed_backward.get('value_substitution', 
                            self._get_gradient_value_substitution(module, input_tensor, output_tensor, grad_output)),
                        'result': grad_input.detach().numpy() if grad_input is not None else None,
                        'detailed_computation': detailed_backward
                    }
    
    def _get_detailed_computation(self, module, input_tensor, output_tensor, position=None):
        """Get detailed computation for a module"""
        try:
            from torch_formula.utils.detailed_computation import get_detailed_computation
            return get_detailed_computation(module, input_tensor, output_tensor, position)
        except (ImportError, AttributeError):
            return {}  # Fallback if the detailed computation module is not available
    
    def _get_detailed_backward(self, module, input_tensor, output_tensor, grad_output, position=None):
        """Get detailed backward computation for a module"""
        try:
            from torch_formula.utils.detailed_computation import get_detailed_backward
            return get_detailed_backward(module, input_tensor, output_tensor, grad_output, position)
        except (ImportError, AttributeError):
            return {}  # Fallback if the detailed backward computation module is not available
    
    def _get_general_formula(self, module):
        """Get the general mathematical formula for a module"""
        if isinstance(module, nn.Conv2d):
            return "y_{n,c_{out},h_{out},w_{out}} = \\sum_{c_{in},k_h,k_w} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \\cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}"
        elif isinstance(module, nn.Linear):
            return "y_{n,j} = \\sum_{i} x_{n,i} \\cdot w_{j,i} + b_j"
        elif isinstance(module, nn.MaxPool2d):
            return "y_{n,c,h_{out},w_{out}} = \\max_{k_h,k_w} x_{n,c,h_{in}+k_h,w_{in}+k_w}"
        elif isinstance(module, nn.AvgPool2d):
            return "y_{n,c,h_{out},w_{out}} = \\frac{1}{k_h \\cdot k_w} \\sum_{i=0}^{k_h-1} \\sum_{j=0}^{k_w-1} x_{n,c,h_{in}+i,w_{in}+j}"
        elif isinstance(module, nn.ReLU):
            return "y_{n,c,h,w} = \\max(0, x_{n,c,h,w})"
        return "Formula not available for this module type"
    
    def _get_value_substitution(self, module, input_tensor, output_tensor):
        """Generate a value substitution example for the module"""
        # This is a simplified version - a full implementation would show actual calculations
        return f"Example calculation for {type(module).__name__}"
    
    def _get_gradient_formula(self, module):
        """Get the general gradient formula for a module"""
        if isinstance(module, nn.Conv2d):
            return "\\frac{\\partial L}{\\partial x_{n,c_{in},h_{in},w_{in}}} = \\sum_{c_{out},k_h,k_w} \\frac{\\partial L}{\\partial y_{n,c_{out},h_{out},w_{out}}} \\cdot w_{c_{out},c_{in},k_h,k_w}"
        elif isinstance(module, nn.Linear):
            return "\\frac{\\partial L}{\\partial x_{n,i}} = \\sum_{j} \\frac{\\partial L}{\\partial y_{n,j}} \\cdot w_{j,i}"
        elif isinstance(module, nn.MaxPool2d):
            return "\\frac{\\partial L}{\\partial x_{n,c,h_{in},w_{in}}} = \\begin{cases} \\frac{\\partial L}{\\partial y_{n,c,h_{out},w_{out}}} & \\text{if } x_{n,c,h_{in},w_{in}} \\text{ is max in pool} \\\\ 0 & \\text{otherwise} \\end{cases}"
        elif isinstance(module, nn.AvgPool2d):
            return "\\frac{\\partial L}{\\partial x_{n,c,h_{in},w_{in}}} = \\frac{1}{k_h \\cdot k_w} \\sum_{h_{out},w_{out}} \\frac{\\partial L}{\\partial y_{n,c,h_{out},w_{out}}}"
        elif isinstance(module, nn.ReLU):
            return "\\frac{\\partial L}{\\partial x_{n,c,h,w}} = \\begin{cases} \\frac{\\partial L}{\\partial y_{n,c,h,w}} & \\text{if } x_{n,c,h,w} > 0 \\\\ 0 & \\text{otherwise} \\end{cases}"
        return "Gradient formula not available for this module type"
    
    def _get_gradient_value_substitution(self, module, input_tensor, output_tensor, grad_output):
        """Generate a gradient value substitution example"""
        # Simplified version
        return f"Example gradient calculation for {type(module).__name__}"
    
    def get_layer_summary(self, layer_name, format='dict', include_detailed_computation=False):
        """
        Get a summary of computations for a specific layer
        
        Args:
            layer_name: Name of the layer
            format: Output format ('dict', 'dataframe', 'markdown', or 'html')
            include_detailed_computation: Whether to include detailed computation in the output
            
        Returns:
            Summary of the layer in the requested format
        """
        if layer_name not in self.computation_steps:
            return f"Layer {layer_name} not found in computation steps"
        
        summary = self.computation_steps[layer_name]
        
        if format == 'dict':
            return summary
        elif format == 'dataframe':
            # Create a pandas DataFrame
            df = pd.DataFrame({
                'Layer': [layer_name],
                'Type': [summary['module_type']],
                'Input Shape': [str(summary['input_shape'])],
                'Output Shape': [str(summary['output_shape'])],
                'Forward Formula': [summary['forward']['general_formula']],
                'Backward Formula': [summary.get('backward', {}).get('general_formula', 'N/A')]
            })
            return df
        elif format == 'markdown':
            # Create a markdown string
            md = f"## Layer: {layer_name} ({summary['module_type']})\n\n"
            md += f"- **Input Shape**: {summary['input_shape']}\n"
            md += f"- **Output Shape**: {summary['output_shape']}\n\n"
            
            md += "### Forward Pass\n\n"
            md += f"- **General Formula**: ${summary['forward']['general_formula']}$\n\n"
            
            # Include detailed forward computation if requested
            if include_detailed_computation and 'detailed_computation' in summary['forward']:
                try:
                    from torch_formula.utils.detailed_computation import format_detailed_computation_as_markdown
                    md += format_detailed_computation_as_markdown(summary['forward']['detailed_computation'])
                except (ImportError, AttributeError):
                    pass
            
            if 'backward' in summary:
                md += "### Backward Pass\n\n"
                md += f"- **General Formula**: ${summary['backward']['general_formula']}$\n\n"
                
                # Include detailed backward computation if requested
                if include_detailed_computation and 'detailed_computation' in summary['backward']:
                    try:
                        from torch_formula.utils.detailed_computation import format_backward_as_markdown
                        md += format_backward_as_markdown(summary['backward']['detailed_computation'])
                    except (ImportError, AttributeError):
                        pass
            
            return md
        elif format == 'html':
            # Create HTML string
            try:
                from torch_formula.utils.detailed_computation import format_detailed_computation_as_html, format_backward_as_html
                html = f"<div class='layer-summary'>"
                html += f"<h2>Layer: {layer_name} ({summary['module_type']})</h2>\n\n"
                html += f"<p><strong>Input Shape</strong>: {summary['input_shape']}</p>\n"
                html += f"<p><strong>Output Shape</strong>: {summary['output_shape']}</p>\n\n"
                
                html += "<h3>Forward Pass</h3>\n\n"
                html += f"<p><strong>General Formula</strong>: \\({summary['forward']['general_formula']}\\)</p>\n\n"
                
                # Include detailed forward computation if requested
                if include_detailed_computation and 'detailed_computation' in summary['forward']:
                    html += format_detailed_computation_as_html(summary['forward']['detailed_computation'])
                
                if 'backward' in summary:
                    html += "<h3>Backward Pass</h3>\n\n"
                    html += f"<p><strong>General Formula</strong>: \\({summary['backward']['general_formula']}\\)</p>\n\n"
                    
                    # Include detailed backward computation if requested
                    if include_detailed_computation and 'detailed_computation' in summary['backward']:
                        html += format_backward_as_html(summary['backward']['detailed_computation'])
                
                html += "</div>"
                return html
            except (ImportError, AttributeError):
                # Fallback to markdown if HTML formatting not available
                return self.get_layer_summary(layer_name, format='markdown', include_detailed_computation=include_detailed_computation)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def get_all_layers_summary(self, format='markdown', include_detailed_computation=False):
        """
        Get a summary of all layers in the specified format
        
        Args:
            format: Output format ('dict', 'dataframe', 'markdown', or 'html')
            include_detailed_computation: Whether to include detailed computation in the output
            
        Returns:
            Summary of all layers in the specified format
        """
        summaries = []
        
        for layer_name in self.computation_steps.keys():
            summaries.append(self.get_layer_summary(layer_name, format, include_detailed_computation))
            
        if format == 'dataframe':
            return pd.concat(summaries, ignore_index=True)
        elif format in ['markdown', 'html']:
            return "\n\n".join(summaries)
        else:
            return summaries
    
    def get_detailed_computation(self, layer_name, position=None, format='markdown'):
        """
        Get detailed computation for a specific layer
        
        Args:
            layer_name: Name of the layer
            position: Position for which to generate the computation
            format: Output format ('markdown' or 'html')
            
        Returns:
            Detailed computation in the specified format
        """
        if layer_name not in self.activations:
            return f"Layer {layer_name} not found"
        
        # Get the module and tensors
        for name, module in self.model.named_modules():
            if name == layer_name:
                input_tensor = self.activations[layer_name]['input']
                output_tensor = self.activations[layer_name]['output']
                
                try:
                    if format == 'markdown':
                        from torch_formula.utils.detailed_computation import get_computation_markdown
                        return get_computation_markdown(module, input_tensor, output_tensor, position)
                    elif format == 'html':
                        from torch_formula.utils.detailed_computation import get_computation_html
                        return get_computation_html(module, input_tensor, output_tensor, position)
                    else:
                        raise ValueError(f"Unsupported format: {format}")
                except (ImportError, AttributeError) as e:
                    return f"Detailed computation not available: {e}"
                
        return f"Module {layer_name} not found"
    
    def get_detailed_backward(self, layer_name, position=None, format='markdown'):
        """
        Get detailed backward computation for a specific layer
        
        Args:
            layer_name: Name of the layer
            position: Position for which to generate the backward computation
            format: Output format ('markdown' or 'html')
            
        Returns:
            Detailed backward computation in the specified format
        """
        if layer_name not in self.activations or layer_name not in self.gradients:
            return f"Layer {layer_name} not found or no gradient information available"
        
        # Get the module and tensors
        for name, module in self.model.named_modules():
            if name == layer_name:
                input_tensor = self.activations[layer_name]['input']
                output_tensor = self.activations[layer_name]['output']
                grad_output = self.gradients[layer_name]['grad_output']
                
                try:
                    if format == 'markdown':
                        from torch_formula.utils.detailed_computation import get_backward_markdown
                        return get_backward_markdown(module, input_tensor, output_tensor, grad_output, position)
                    elif format == 'html':
                        from torch_formula.utils.detailed_computation import get_backward_html
                        return get_backward_html(module, input_tensor, output_tensor, grad_output, position)
                    else:
                        raise ValueError(f"Unsupported format: {format}")
                except (ImportError, AttributeError) as e:
                    return f"Detailed backward computation not available: {e}"
                
        return f"Module {layer_name} not found"
    
    def get_tensor_region(self, layer_name, tensor_type='input', position=None, region_size=None, format='markdown'):
        """
        Get a region of a tensor for a specific layer
        
        Args:
            layer_name: Name of the layer
            tensor_type: Type of tensor ('input', 'output', 'grad_input', or 'grad_output')
            position: Position from where to start sampling
            region_size: Size of the region to sample
            format: Output format ('markdown' or 'html')
            
        Returns:
            Formatted tensor region
        """
        if layer_name not in self.activations:
            return f"Layer {layer_name} not found"
        
        # Get the tensor
        if tensor_type == 'input':
            tensor = self.activations[layer_name]['input']
        elif tensor_type == 'output':
            tensor = self.activations[layer_name]['output']
        elif tensor_type == 'grad_input':
            if layer_name not in self.gradients or self.gradients[layer_name].get('grad_input') is None:
                return f"Gradient input not available for layer {layer_name}"
            tensor = self.gradients[layer_name]['grad_input']
        elif tensor_type == 'grad_output':
            if layer_name not in self.gradients:
                return f"Gradient output not available for layer {layer_name}"
            tensor = self.gradients[layer_name]['grad_output']
        else:
            return f"Unsupported tensor type: {tensor_type}"
        
        try:
            from torch_formula.utils.detailed_computation import sample_region, format_tensor_as_markdown_table
            
            # Sample the region
            region = sample_region(tensor, position, region_size)
            
            # Format the tensor
            if format == 'markdown':
                return format_tensor_as_markdown_table(region, f"{tensor_type.capitalize()} Region for {layer_name}")
            elif format == 'html':
                from torch_formula.utils.detailed_computation.formatters import tensor_to_html_table
                return tensor_to_html_table(region, f"{tensor_type.capitalize()} Region for {layer_name}")
            else:
                raise ValueError(f"Unsupported format: {format}")
        except (ImportError, AttributeError) as e:
            return f"Tensor region formatting not available: {e}"
    
    def cleanup(self):
        """Clean up resources (remove hooks)"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
