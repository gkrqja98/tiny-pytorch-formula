"""
Formatters for detailed computation results
Provides functions to format computation results in Markdown, LaTeX, or HTML
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import html


def format_tensor_as_markdown_table(tensor: Union[torch.Tensor, np.ndarray], name: str = "Tensor") -> str:
    """
    Format a tensor as a markdown table
    
    Args:
        tensor: The tensor to format
        name: Name of the tensor
        
    Returns:
        Markdown table string
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    # Handle different tensor dimensions
    if tensor.ndim == 1:
        # 1D tensor (vector)
        md = f"#### {name} (Shape: {tensor.shape})\n\n"
        md += "| "
        # Add column headers
        for i in range(tensor.shape[0]):
            md += f"[{i}] | "
        md += "\n| "
        # Add separator row
        for i in range(tensor.shape[0]):
            md += "--- | "
        md += "\n| "
        # Add the data row
        for i in range(tensor.shape[0]):
            md += f"{tensor[i]:.4f} | "
        md += "\n\n"
        return md
    
    elif tensor.ndim == 2:
        # 2D tensor (matrix)
        md = f"#### {name} (Shape: {tensor.shape})\n\n"
        md += "| Index | "
        # Add column headers
        for i in range(tensor.shape[1]):
            md += f"[{i}] | "
        md += "\n| --- | "
        # Add separator row
        for i in range(tensor.shape[1]):
            md += "--- | "
        md += "\n"
        # Add data rows
        for i in range(tensor.shape[0]):
            md += f"| [{i}] | "
            for j in range(tensor.shape[1]):
                md += f"{tensor[i, j]:.4f} | "
            md += "\n"
        md += "\n"
        return md
    
    elif tensor.ndim == 3:
        # For 3D tensors, show each channel separately
        md = f"#### {name} (Shape: {tensor.shape})\n\n"
        
        for c in range(tensor.shape[0]):
            md += f"**Channel {c}**\n\n"
            md += "| Index | "
            # Add column headers
            for i in range(tensor.shape[2]):
                md += f"[{i}] | "
            md += "\n| --- | "
            # Add separator row
            for i in range(tensor.shape[2]):
                md += "--- | "
            md += "\n"
            # Add data rows
            for i in range(tensor.shape[1]):
                md += f"| [{i}] | "
                for j in range(tensor.shape[2]):
                    md += f"{tensor[c, i, j]:.4f} | "
                md += "\n"
            md += "\n"
        
        return md
    
    elif tensor.ndim == 4:
        # For 4D tensors, show first few channels of first sample
        md = f"#### {name} (Shape: {tensor.shape})\n\n"
        md += "> Note: Showing first sample, first 3 channels (or fewer)\n\n"
        
        max_channels = min(3, tensor.shape[1])
        for c in range(max_channels):
            md += f"**Batch 0, Channel {c}**\n\n"
            md += "| Index | "
            # Add column headers
            for i in range(tensor.shape[3]):
                md += f"[{i}] | "
            md += "\n| --- | "
            # Add separator row
            for i in range(tensor.shape[3]):
                md += "--- | "
            md += "\n"
            # Add data rows
            for i in range(tensor.shape[2]):
                md += f"| [{i}] | "
                for j in range(tensor.shape[3]):
                    md += f"{tensor[0, c, i, j]:.4f} | "
                md += "\n"
            md += "\n"
        
        return md
    
    else:
        return f"#### {name} (Shape: {tensor.shape})\n\n(Tensor with more than 4 dimensions, showing simplified view)\n\n"


def format_conv2d_computation_as_markdown(comp_results: Dict[str, Any]) -> str:
    """
    Format Conv2d computation results as markdown
    
    Args:
        comp_results: The computation results
        
    Returns:
        Markdown string with formatted computation
    """
    md = f"### Computing output at position: (batch={comp_results['position'][0]}, out_channel={comp_results['position'][1]}, y={comp_results['position'][2]}, x={comp_results['position'][3]})\n\n"
    
    # Filter weights
    filter_weights = comp_results['filter_weights']
    md += format_tensor_as_markdown_table(filter_weights, "Filter Weights")
    
    # Receptive field
    receptive_field = comp_results['receptive_field']
    md += format_tensor_as_markdown_table(receptive_field, "Input Receptive Field")
    
    # General formula
    md += "### General Formula\n\n"
    md += f"${comp_results['general_formula']}$\n\n"
    
    # Value substitution
    md += "### Value Substitution\n\n"
    md += f"${comp_results['value_substitution']}$\n\n"
    
    # Computation result
    md += "### Computation Result\n\n"
    md += f"Calculated value: {comp_results['computed_result']:.6f}\n\n"
    md += f"Actual output value: {comp_results['actual_output']:.6f}\n\n"
    
    return md


def format_maxpool2d_computation_as_markdown(comp_results: Dict[str, Any]) -> str:
    """
    Format MaxPool2d computation results as markdown
    
    Args:
        comp_results: The computation results
        
    Returns:
        Markdown string with formatted computation
    """
    md = f"### Computing output at position: (batch={comp_results['position'][0]}, channel={comp_results['position'][1]}, y={comp_results['position'][2]}, x={comp_results['position'][3]})\n\n"
    
    # Receptive field
    receptive_field = comp_results['receptive_field']
    max_y, max_x = comp_results['max_position']
    
    md += format_tensor_as_markdown_table(receptive_field, "Input Receptive Field")
    md += f"Maximum value position in receptive field: ({max_y}, {max_x}) with value {receptive_field[max_y, max_x]:.4f}\n\n"
    
    # General formula
    md += "### General Formula\n\n"
    md += f"${comp_results['general_formula']}$\n\n"
    
    # Value substitution
    md += "### Value Substitution\n\n"
    md += f"${comp_results['value_substitution']}$\n\n"
    
    # Computation result
    md += "### Computation Result\n\n"
    md += f"Calculated max value: {comp_results['computed_result']:.6f}\n\n"
    md += f"Actual output value: {comp_results['actual_output']:.6f}\n\n"
    
    return md


def format_avgpool2d_computation_as_markdown(comp_results: Dict[str, Any]) -> str:
    """
    Format AvgPool2d computation results as markdown
    
    Args:
        comp_results: The computation results
        
    Returns:
        Markdown string with formatted computation
    """
    md = f"### Computing output at position: (batch={comp_results['position'][0]}, channel={comp_results['position'][1]}, y={comp_results['position'][2]}, x={comp_results['position'][3]})\n\n"
    
    # Receptive field
    receptive_field = comp_results['receptive_field']
    kernel_size = comp_results['kernel_size']
    
    md += format_tensor_as_markdown_table(receptive_field, "Input Receptive Field")
    md += f"Kernel size: {kernel_size}\n\n"
    
    # General formula
    md += "### General Formula\n\n"
    md += f"${comp_results['general_formula']}$\n\n"
    
    # Value substitution
    md += "### Value Substitution\n\n"
    md += f"${comp_results['value_substitution']}$\n\n"
    
    # Computation result
    md += "### Computation Result\n\n"
    md += f"Calculated average value: {comp_results['computed_result']:.6f}\n\n"
    md += f"Actual output value: {comp_results['actual_output']:.6f}\n\n"
    
    return md


def format_linear_computation_as_markdown(comp_results: Dict[str, Any]) -> str:
    """
    Format Linear computation results as markdown
    
    Args:
        comp_results: The computation results
        
    Returns:
        Markdown string with formatted computation
    """
    md = f"### Computing output at position: (batch={comp_results['position'][0]}, output_feature={comp_results['position'][1]})\n\n"
    
    # Input features
    input_features = comp_results['input_features']
    md += format_tensor_as_markdown_table(input_features, "Input Features")
    
    # Weights
    output_weights = comp_results['output_weights']
    md += format_tensor_as_markdown_table(output_weights, "Weights for Output Feature")
    
    # General formula
    md += "### General Formula\n\n"
    md += f"${comp_results['general_formula']}$\n\n"
    
    # Value substitution
    md += "### Value Substitution\n\n"
    md += f"${comp_results['value_substitution']}$\n\n"
    
    # Computation result
    md += "### Computation Result\n\n"
    md += f"Calculated value: {comp_results['computed_result']:.6f}\n\n"
    md += f"Actual output value: {comp_results['actual_output']:.6f}\n\n"
    
    return md


def format_detailed_computation_as_markdown(comp_results: Dict[str, Any]) -> str:
    """
    Format detailed computation results as markdown
    
    Args:
        comp_results: The computation results
        
    Returns:
        Markdown string with formatted computation
    """
    module_type = comp_results['module_type']
    
    if module_type == 'Conv2d':
        return format_conv2d_computation_as_markdown(comp_results)
    elif module_type == 'Linear':
        return format_linear_computation_as_markdown(comp_results)
    elif module_type == 'MaxPool2d':
        return format_maxpool2d_computation_as_markdown(comp_results)
    elif module_type == 'AvgPool2d':
        return format_avgpool2d_computation_as_markdown(comp_results)
    else:
        return f"Detailed formatting not implemented for module type: {module_type}\n\n"


def format_detailed_computation_as_html(comp_results: Dict[str, Any]) -> str:
    """
    Format detailed computation results as HTML
    
    Args:
        comp_results: The computation results
        
    Returns:
        HTML string with formatted computation
    """
    module_type = comp_results['module_type']
    
    html_content = f"""
    <div class="computation-details">
    """
    
    # Convert the markdown to HTML
    if module_type == 'Conv2d':
        html_content += html_for_conv2d(comp_results)
    elif module_type == 'Linear':
        html_content += html_for_linear(comp_results)
    elif module_type == 'MaxPool2d':
        html_content += html_for_maxpool2d(comp_results)
    elif module_type == 'AvgPool2d':
        html_content += html_for_avgpool2d(comp_results)
    else:
        html_content += f"<p>Detailed formatting not implemented for module type: {module_type}</p>"
    
    html_content += "\n</div>"
    
    return html_content


def html_for_conv2d(comp_results):
    """Generate HTML for Conv2d computation"""
    pos = comp_results['position']
    
    html = f"""
    <h3>Computing output at position: (batch={pos[0]}, out_channel={pos[1]}, y={pos[2]}, x={pos[3]})</h3>
    
    <h4>Filter Weights (Shape: {comp_results['filter_weights'].shape})</h4>
    {tensor_to_html_table(comp_results['filter_weights'])}
    
    <h4>Input Receptive Field (Shape: {comp_results['receptive_field'].shape})</h4>
    {tensor_to_html_table(comp_results['receptive_field'])}
    
    <h3>General Formula</h3>
    <p>\\({comp_results['general_formula']}\\)</p>
    
    <h3>Value Substitution</h3>
    <p>\\({comp_results['value_substitution']}\\)</p>
    
    <h3>Computation Result</h3>
    <p>Calculated value: {comp_results['computed_result']:.6f}</p>
    <p>Actual output value: {comp_results['actual_output']:.6f}</p>
    """
    
    return html


def html_for_maxpool2d(comp_results):
    """Generate HTML for MaxPool2d computation"""
    pos = comp_results['position']
    max_y, max_x = comp_results['max_position']
    
    html = f"""
    <h3>Computing output at position: (batch={pos[0]}, channel={pos[1]}, y={pos[2]}, x={pos[3]})</h3>
    
    <h4>Input Receptive Field (Shape: {comp_results['receptive_field'].shape})</h4>
    {tensor_to_html_table(comp_results['receptive_field'])}
    <p>Maximum value position in receptive field: ({max_y}, {max_x}) with value {comp_results['receptive_field'][max_y, max_x]:.4f}</p>
    
    <h3>General Formula</h3>
    <p>\\({comp_results['general_formula']}\\)</p>
    
    <h3>Value Substitution</h3>
    <p>\\({comp_results['value_substitution']}\\)</p>
    
    <h3>Computation Result</h3>
    <p>Calculated max value: {comp_results['computed_result']:.6f}</p>
    <p>Actual output value: {comp_results['actual_output']:.6f}</p>
    """
    
    return html


def html_for_avgpool2d(comp_results):
    """Generate HTML for AvgPool2d computation"""
    pos = comp_results['position']
    
    html = f"""
    <h3>Computing output at position: (batch={pos[0]}, channel={pos[1]}, y={pos[2]}, x={pos[3]})</h3>
    
    <h4>Input Receptive Field (Shape: {comp_results['receptive_field'].shape})</h4>
    {tensor_to_html_table(comp_results['receptive_field'])}
    <p>Kernel size: {comp_results['kernel_size']}</p>
    
    <h3>General Formula</h3>
    <p>\\({comp_results['general_formula']}\\)</p>
    
    <h3>Value Substitution</h3>
    <p>\\({comp_results['value_substitution']}\\)</p>
    
    <h3>Computation Result</h3>
    <p>Calculated average value: {comp_results['computed_result']:.6f}</p>
    <p>Actual output value: {comp_results['actual_output']:.6f}</p>
    """
    
    return html


def html_for_linear(comp_results):
    """Generate HTML for Linear computation"""
    pos = comp_results['position']
    
    html = f"""
    <h3>Computing output at position: (batch={pos[0]}, output_feature={pos[1]})</h3>
    
    <h4>Input Features (Shape: {comp_results['input_features'].shape})</h4>
    {tensor_to_html_table(comp_results['input_features'])}
    
    <h4>Weights for Output Feature (Shape: {comp_results['output_weights'].shape})</h4>
    {tensor_to_html_table(comp_results['output_weights'])}
    
    <h3>General Formula</h3>
    <p>\\({comp_results['general_formula']}\\)</p>
    
    <h3>Value Substitution</h3>
    <p>\\({comp_results['value_substitution']}\\)</p>
    
    <h3>Computation Result</h3>
    <p>Calculated value: {comp_results['computed_result']:.6f}</p>
    <p>Actual output value: {comp_results['actual_output']:.6f}</p>
    """
    
    return html


def tensor_to_html_table(tensor):
    """Convert tensor to HTML table"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    html = ""
    
    # Handle different tensor dimensions
    if tensor.ndim == 1:
        # 1D tensor (vector)
        html += "<table class='tensor-table'>\n<tr>"
        # Add column headers
        for i in range(tensor.shape[0]):
            html += f"<th>[{i}]</th>"
        html += "</tr>\n<tr>"
        # Add data row
        for i in range(tensor.shape[0]):
            html += f"<td>{tensor[i]:.4f}</td>"
        html += "</tr>\n</table>"
        
    elif tensor.ndim == 2:
        # 2D tensor (matrix)
        html += "<table class='tensor-table'>\n<tr><th></th>"
        # Add column headers
        for i in range(tensor.shape[1]):
            html += f"<th>[{i}]</th>"
        html += "</tr>\n"
        # Add data rows
        for i in range(tensor.shape[0]):
            html += f"<tr><th>[{i}]</th>"
            for j in range(tensor.shape[1]):
                html += f"<td>{tensor[i, j]:.4f}</td>"
            html += "</tr>\n"
        html += "</table>"
        
    elif tensor.ndim == 3:
        # For 3D tensors, show each channel separately
        for c in range(tensor.shape[0]):
            html += f"<h5>Channel {c}</h5>\n"
            html += "<table class='tensor-table'>\n<tr><th></th>"
            # Add column headers
            for i in range(tensor.shape[2]):
                html += f"<th>[{i}]</th>"
            html += "</tr>\n"
            # Add data rows
            for i in range(tensor.shape[1]):
                html += f"<tr><th>[{i}]</th>"
                for j in range(tensor.shape[2]):
                    html += f"<td>{tensor[c, i, j]:.4f}</td>"
                html += "</tr>\n"
            html += "</table>\n"
            
    elif tensor.ndim == 4:
        # For 4D tensors, show first few channels of first sample
        html += "<p>Note: Showing first sample, first 3 channels (or fewer)</p>\n"
        
        max_channels = min(3, tensor.shape[1])
        for c in range(max_channels):
            html += f"<h5>Batch 0, Channel {c}</h5>\n"
            html += "<table class='tensor-table'>\n<tr><th></th>"
            # Add column headers
            for i in range(tensor.shape[3]):
                html += f"<th>[{i}]</th>"
            html += "</tr>\n"
            # Add data rows
            for i in range(tensor.shape[2]):
                html += f"<tr><th>[{i}]</th>"
                for j in range(tensor.shape[3]):
                    html += f"<td>{tensor[0, c, i, j]:.4f}</td>"
                html += "</tr>\n"
            html += "</table>\n"
            
    else:
        html += "<p>(Tensor with more than 4 dimensions, showing simplified view)</p>\n"
    
    return html


def get_computation_markdown(module: nn.Module, 
                          input_tensor: torch.Tensor,
                          output_tensor: torch.Tensor,
                          position: Optional[Tuple[int, ...]] = None) -> str:
    """
    Get detailed computation as markdown
    
    Args:
        module: The PyTorch module
        input_tensor: Input tensor to the module
        output_tensor: Output tensor from the module
        position: Position for which to generate the computation
        
    Returns:
        Markdown string with computation details
    """
    from .detailed_computation_utils import get_detailed_computation
    
    # Get computation details
    comp_results = get_detailed_computation(module, input_tensor, output_tensor, position)
    
    # Format as markdown
    return format_detailed_computation_as_markdown(comp_results)


def get_computation_html(module: nn.Module, 
                       input_tensor: torch.Tensor,
                       output_tensor: torch.Tensor,
                       position: Optional[Tuple[int, ...]] = None) -> str:
    """
    Get detailed computation as HTML
    
    Args:
        module: The PyTorch module
        input_tensor: Input tensor to the module
        output_tensor: Output tensor from the module
        position: Position for which to generate the computation
        
    Returns:
        HTML string with computation details
    """
    from .detailed_computation_utils import get_detailed_computation
    
    # Get computation details
    comp_results = get_detailed_computation(module, input_tensor, output_tensor, position)
    
    # Format as HTML
    return format_detailed_computation_as_html(comp_results)
