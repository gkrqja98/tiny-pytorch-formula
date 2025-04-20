"""
Formatters for detailed backward computation results
Provides functions to format backward computation results in Markdown, LaTeX, or HTML
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import html

from .formatters import format_tensor_as_markdown_table, tensor_to_html_table
from torch_formula.utils.html_renderer.renderer import HTMLRenderer
from torch_formula.utils.html_renderer.math_formatter import MathFormatter


def format_conv2d_backward_as_markdown(comp_results: Dict[str, Any]) -> str:
    """
    Format Conv2d backward computation results as markdown
    
    Args:
        comp_results: The backward computation results
        
    Returns:
        Markdown string with formatted backward computation
    """
    position = comp_results['position']
    md = f"### Backward Computation for Conv2d at Position: (batch={position[0]}, out_channel={position[1]}, y={position[2]}, x={position[3]})\n\n"
    
    # Gradient output value
    md += f"#### Gradient Output Value\n\n"
    md += f"{comp_results['grad_output_val']:.6f}\n\n"
    
    # Filter weights
    md += "#### Filter Weights\n\n"
    md += format_tensor_as_markdown_table(comp_results['filter_weights'], "Filter Weights")
    
    # Gradient propagation to input
    md += "#### Gradient Propagation to Input\n\n"
    positions = comp_results['input_grad']['positions']
    formulas = comp_results['input_grad']['formulas']
    values = comp_results['input_grad']['values']
    
    md += "| Position (y, x, c) | Formula | Gradient Value |\n"
    md += "| ----------------- | ------- | -------------- |\n"
    
    for i in range(len(positions)):
        pos = positions[i]
        formula = formulas[i]
        value = values[i]
        md += f"| ({pos[0]}, {pos[1]}, {pos[2]}) | ${formula}$ | {value:.6f} |\n"
    
    md += "\n"
    
    # Weight gradient computation
    md += "#### Weight Gradient Computation\n\n"
    positions = comp_results['weight_grad']['positions']
    formulas = comp_results['weight_grad']['formulas']
    values = comp_results['weight_grad']['values']
    
    md += "| Weight Position (out_c, in_c, ky, kx) | Formula | Gradient Value |\n"
    md += "| ------------------------------------ | ------- | -------------- |\n"
    
    for i in range(len(positions)):
        pos = positions[i]
        formula = formulas[i]
        value = values[i]
        md += f"| ({pos[0]}, {pos[1]}, {pos[2]}, {pos[3]}) | ${formula}$ | {value:.6f} |\n"
    
    md += "\n"
    
    # Bias gradient
    md += "#### Bias Gradient Computation\n\n"
    md += f"Bias Gradient for output channel {position[1]}: {comp_results['bias_grad']:.6f}\n\n"
    
    # General gradient formulas
    md += "### General Gradient Formulas\n\n"
    md += "#### Input Gradient Formula\n\n"
    md += f"${comp_results['general_input_grad_formula']}$\n\n"
    
    md += "#### Weight Gradient Formula\n\n"
    md += f"${comp_results['general_weight_grad_formula']}$\n\n"
    
    md += "#### Bias Gradient Formula\n\n"
    md += f"${comp_results['general_bias_grad_formula']}$\n\n"
    
    return md


def format_maxpool2d_backward_as_markdown(comp_results: Dict[str, Any]) -> str:
    """
    Format MaxPool2d backward computation results as markdown
    
    Args:
        comp_results: The backward computation results
        
    Returns:
        Markdown string with formatted backward computation
    """
    position = comp_results['position']
    md = f"### Backward Computation for MaxPool2d at Position: (batch={position[0]}, channel={position[1]}, y={position[2]}, x={position[3]})\n\n"
    
    # Gradient output value
    md += f"#### Gradient Output Value\n\n"
    md += f"{comp_results['grad_output_val']:.6f}\n\n"
    
    # Receptive field
    md += "#### Input Receptive Field\n\n"
    md += format_tensor_as_markdown_table(comp_results['receptive_field'], "Receptive Field")
    
    # Max position
    max_y, max_x = comp_results['max_position']
    md += f"Maximum value position in receptive field: ({max_y}, {max_x}) with value {comp_results['receptive_field'][max_y, max_x]:.6f}\n\n"
    
    # Gradient propagation to input
    md += "#### Gradient Propagation to Input\n\n"
    positions = comp_results['input_grad']['positions']
    formulas = comp_results['input_grad']['formulas']
    values = comp_results['input_grad']['values']
    
    md += "| Position (y, x) | Formula | Gradient Value |\n"
    md += "| -------------- | ------- | -------------- |\n"
    
    for i in range(len(positions)):
        pos = positions[i]
        formula = formulas[i]
        value = values[i]
        md += f"| ({pos[0]}, {pos[1]}) | ${formula}$ | {value:.6f} |\n"
    
    md += "\n"
    
    md += "All other positions in the receptive field receive zero gradient.\n\n"
    
    # Gradient map visualization
    md += "#### Gradient Map (Receptive Field)\n\n"
    md += format_tensor_as_markdown_table(comp_results['input_grad']['map'], "Gradient Map")
    
    # General gradient formula
    md += "### General Gradient Formula\n\n"
    md += f"${comp_results['general_input_grad_formula']}$\n\n"
    
    return md


def format_avgpool2d_backward_as_markdown(comp_results: Dict[str, Any]) -> str:
    """
    Format AvgPool2d backward computation results as markdown
    
    Args:
        comp_results: The backward computation results
        
    Returns:
        Markdown string with formatted backward computation
    """
    position = comp_results['position']
    md = f"### Backward Computation for AvgPool2d at Position: (batch={position[0]}, channel={position[1]}, y={position[2]}, x={position[3]})\n\n"
    
    # Gradient output value
    md += f"#### Gradient Output Value\n\n"
    md += f"{comp_results['grad_output_val']:.6f}\n\n"
    
    # Pool info
    md += f"Pool Size: {comp_results['pool_size']} (kernel: {comp_results['kernel_size']})\n\n"
    md += f"Distributed Gradient: {comp_results['distributed_grad']:.6f} ({comp_results['grad_output_val']:.6f} / {comp_results['pool_size']})\n\n"
    
    # Gradient propagation to input
    md += "#### Gradient Propagation to Input\n\n"
    positions = comp_results['input_grad']['positions']
    formulas = comp_results['input_grad']['formulas']
    values = comp_results['input_grad']['values']
    
    md += "| Position (y, x) | Formula | Gradient Value |\n"
    md += "| -------------- | ------- | -------------- |\n"
    
    for i in range(len(positions)):
        pos = positions[i]
        formula = formulas[i]
        value = values[i]
        md += f"| ({pos[0]}, {pos[1]}) | ${formula}$ | {value:.6f} |\n"
    
    md += "\n"
    
    # Gradient map visualization
    md += "#### Gradient Map (Receptive Field)\n\n"
    md += format_tensor_as_markdown_table(comp_results['input_grad']['map'], "Gradient Map")
    
    # General gradient formula
    md += "### General Gradient Formula\n\n"
    md += f"${comp_results['general_input_grad_formula']}$\n\n"
    
    return md


def format_linear_backward_as_markdown(comp_results: Dict[str, Any]) -> str:
    """
    Format Linear backward computation results as markdown
    
    Args:
        comp_results: The backward computation results
        
    Returns:
        Markdown string with formatted backward computation
    """
    position = comp_results['position']
    md = f"### Backward Computation for Linear at Position: (batch={position[0]}, output_feature={position[1]})\n\n"
    
    # Gradient output value
    md += f"#### Gradient Output Value\n\n"
    md += f"{comp_results['grad_output_val']:.6f}\n\n"
    
    # Input features
    md += "#### Input Features\n\n"
    md += format_tensor_as_markdown_table(comp_results['input_features'], "Input Features")
    
    # Output weights
    md += "#### Output Weights\n\n"
    md += format_tensor_as_markdown_table(comp_results['output_weights'], "Output Weights")
    
    # Gradient propagation to input
    md += "#### Gradient Propagation to Input\n\n"
    positions = comp_results['input_grad']['positions']
    formulas = comp_results['input_grad']['formulas']
    values = comp_results['input_grad']['values']
    
    md += "| Input Feature | Formula | Gradient Value |\n"
    md += "| ------------- | ------- | -------------- |\n"
    
    for i in range(len(positions)):
        pos = positions[i]
        formula = formulas[i]
        value = values[i]
        md += f"| {pos} | ${formula}$ | {value:.6f} |\n"
    
    md += "\n"
    
    # Weight gradient computation
    md += "#### Weight Gradient Computation\n\n"
    positions = comp_results['weight_grad']['positions']
    formulas = comp_results['weight_grad']['formulas']
    values = comp_results['weight_grad']['values']
    
    md += "| Weight Position (out_f, in_f) | Formula | Gradient Value |\n"
    md += "| ----------------------------- | ------- | -------------- |\n"
    
    for i in range(len(positions)):
        pos = positions[i]
        formula = formulas[i]
        value = values[i]
        md += f"| ({pos[0]}, {pos[1]}) | ${formula}$ | {value:.6f} |\n"
    
    md += "\n"
    
    # Bias gradient
    md += "#### Bias Gradient Computation\n\n"
    md += f"Bias Gradient for output feature {position[1]}: {comp_results['bias_grad']:.6f}\n\n"
    
    # General gradient formulas
    md += "### General Gradient Formulas\n\n"
    md += "#### Input Gradient Formula\n\n"
    md += f"${comp_results['general_input_grad_formula']}$\n\n"
    
    md += "#### Weight Gradient Formula\n\n"
    md += f"${comp_results['general_weight_grad_formula']}$\n\n"
    
    md += "#### Bias Gradient Formula\n\n"
    md += f"${comp_results['general_bias_grad_formula']}$\n\n"
    
    return md


def format_relu_backward_as_markdown(comp_results: Dict[str, Any]) -> str:
    """
    Format ReLU backward computation results as markdown
    
    Args:
        comp_results: The backward computation results
        
    Returns:
        Markdown string with formatted backward computation
    """
    position = comp_results['position']
    pos_str = ", ".join([str(p) for p in position])
    md = f"### Backward Computation for ReLU at Position: ({pos_str})\n\n"
    
    # Input, output, and gradient values
    md += "#### Values at Position\n\n"
    md += f"- Input Value: {comp_results['input_val']:.6f}\n"
    md += f"- Output Value: {comp_results['output_val']:.6f}\n"
    md += f"- Gradient Output Value: {comp_results['grad_output_val']:.6f}\n\n"
    
    # Gradient computation
    md += "#### Gradient Computation\n\n"
    md += f"Input Gradient = {comp_results['formula']}\n\n"
    md += f"Computed Gradient Value: {comp_results['grad_input']:.6f}\n\n"
    md += f"Explanation: {comp_results['explanation']}\n\n"
    
    # General gradient formula
    md += "### General Gradient Formula\n\n"
    md += f"${comp_results['general_formula']}$\n\n"
    
    return md


def format_backward_as_markdown(comp_results: Dict[str, Any]) -> str:
    """
    Format backward computation results as markdown
    
    Args:
        comp_results: The backward computation results
        
    Returns:
        Markdown string with formatted backward computation
    """
    module_type = comp_results['module_type']
    
    if module_type == 'Conv2d':
        return format_conv2d_backward_as_markdown(comp_results)
    elif module_type == 'Linear':
        return format_linear_backward_as_markdown(comp_results)
    elif module_type == 'MaxPool2d':
        return format_maxpool2d_backward_as_markdown(comp_results)
    elif module_type == 'AvgPool2d':
        return format_avgpool2d_backward_as_markdown(comp_results)
    elif module_type == 'ReLU':
        return format_relu_backward_as_markdown(comp_results)
    else:
        return f"Detailed backward formatting not implemented for module type: {module_type}\n\n"


def conv2d_backward_html(comp_results):
    """Generate HTML for Conv2d backward computation"""
    position = comp_results['position']
    
    # Use MathFormatter for specialized gradient formulas
    input_grad_formula = MathFormatter.format_input_gradient()
    weight_grad_formula = MathFormatter.format_weight_gradient()
    bias_grad_formula = MathFormatter.format_bias_gradient()
    
    html = f"""
    <div class="backward-computation">
    <h3>Backward Computation for Conv2d at Position: batch = {position[0]}, out_channel = {position[1]}, y = {position[2]}, x = {position[3]}</h3>
    
    <h4>Gradient Output Value</h4>
    <p>{comp_results['grad_output_val']:.6f}</p>
    
    <h4>Filter Weights</h4>
    {tensor_to_html_table(comp_results['filter_weights'])}
    
    <h4>Gradient Propagation to Input</h4>
    <table class="tensor-table">
    <tr><th>Position (y, x, c)</th><th>Formula</th><th>Gradient Value</th></tr>
    """
    
    positions = comp_results['input_grad']['positions']
    formulas = comp_results['input_grad']['formulas']
    values = comp_results['input_grad']['values']
    
    for i in range(len(positions)):
        pos = positions[i]
        formula = MathFormatter.format_tensor_element(formulas[i], display_mode=False)
        value = values[i]
        html += f"<tr><td>({pos[0]}, {pos[1]}, {pos[2]})</td><td>{formula}</td><td>{value:.6f}</td></tr>\n"
    
    html += """
    </table>
    
    <h4>Weight Gradient Computation</h4>
    <table class="tensor-table">
    <tr><th>Weight Position (out_c, in_c, ky, kx)</th><th>Formula</th><th>Gradient Value</th></tr>
    """
    
    positions = comp_results['weight_grad']['positions']
    formulas = comp_results['weight_grad']['formulas']
    values = comp_results['weight_grad']['values']
    
    for i in range(len(positions)):
        pos = positions[i]
        formula = MathFormatter.format_tensor_element(formulas[i], display_mode=False)
        value = values[i]
        html += f"<tr><td>({pos[0]}, {pos[1]}, {pos[2]}, {pos[3]})</td><td>{formula}</td><td>{value:.6f}</td></tr>\n"
    
    html += f"""
    </table>
    
    <h4>Bias Gradient Computation</h4>
    <p>Bias Gradient for output channel {position[1]}: {comp_results['bias_grad']:.6f}</p>
    
    <h3>General Gradient Formulas</h3>
    
    <div class="formula">
        <h4>Input Gradient Formula</h4>
        <div class="math-container">{input_grad_formula}</div>
    </div>
    
    <div class="formula">
        <h4>Weight Gradient Formula</h4>
        <div class="math-container">{weight_grad_formula}</div>
    </div>
    
    <div class="formula">
        <h4>Bias Gradient Formula</h4>
        <div class="math-container">{bias_grad_formula}</div>
    </div>
    </div>
    """
    
    return html


def maxpool2d_backward_html(comp_results):
    """Generate HTML for MaxPool2d backward computation"""
    position = comp_results['position']
    max_y, max_x = comp_results['max_position']
    
    # Format general formula
    general_formula_latex = r"\frac{\partial L}{\partial x_{i,j}} = \begin{cases} \frac{\partial L}{\partial y} & \text{if } (i,j) \text{ is the location of max value} \\ 0 & \text{otherwise} \end{cases}"
    general_formula = f"$${general_formula_latex}$$"
    
    html = f"""
    <div class="backward-computation">
    <h3>Backward Computation for MaxPool2d at Position: batch = {position[0]}, channel = {position[1]}, y = {position[2]}, x = {position[3]}</h3>
    
    <h4>Gradient Output Value</h4>
    <p>{comp_results['grad_output_val']:.6f}</p>
    
    <h4>Input Receptive Field</h4>
    {tensor_to_html_table(comp_results['receptive_field'])}
    
    <p>Maximum value position in receptive field: ({max_y}, {max_x}) with value {comp_results['receptive_field'][max_y, max_x]:.6f}</p>
    
    <h4>Gradient Propagation to Input</h4>
    <table class="tensor-table">
    <tr><th>Position (y, x)</th><th>Formula</th><th>Gradient Value</th></tr>
    """
    
    positions = comp_results['input_grad']['positions']
    formulas = comp_results['input_grad']['formulas']
    values = comp_results['input_grad']['values']
    
    for i in range(len(positions)):
        pos = positions[i]
        formula = MathFormatter.format_tensor_element(formulas[i], display_mode=False)
        value = values[i]
        html += f"<tr><td>({pos[0]}, {pos[1]})</td><td>{formula}</td><td>{value:.6f}</td></tr>\n"
    
    html += """
    </table>
    
    <p>All other positions in the receptive field receive zero gradient.</p>
    
    <h4>Gradient Map (Receptive Field)</h4>
    """
    
    html += tensor_to_html_table(comp_results['input_grad']['map'])
    
    html += f"""
    <div class="formula">
        <h4>General Gradient Formula</h4>
        <div class="math-container">{general_formula}</div>
    </div>
    </div>
    """
    
    return html


def avgpool2d_backward_html(comp_results):
    """Generate HTML for AvgPool2d backward computation"""
    position = comp_results['position']
    
    # Format general formula
    kernel_size = comp_results['kernel_size']
    general_formula_latex = r"\frac{\partial L}{\partial x_{i,j}} = \frac{1}{" + f"{kernel_size}^2" + r"} \cdot \frac{\partial L}{\partial y} \quad \text{for all } (i,j) \text{ in the receptive field}"
    general_formula = f"$${general_formula_latex}$$"
    
    html = f"""
    <div class="backward-computation">
    <h3>Backward Computation for AvgPool2d at Position: batch = {position[0]}, channel = {position[1]}, y = {position[2]}, x = {position[3]}</h3>
    
    <h4>Gradient Output Value</h4>
    <p>{comp_results['grad_output_val']:.6f}</p>
    
    <p>Pool Size: {comp_results['pool_size']} (kernel: {comp_results['kernel_size']})</p>
    <p>Distributed Gradient: {comp_results['distributed_grad']:.6f} ({comp_results['grad_output_val']:.6f} / {comp_results['pool_size']})</p>
    
    <h4>Gradient Propagation to Input</h4>
    <table class="tensor-table">
    <tr><th>Position (y, x)</th><th>Formula</th><th>Gradient Value</th></tr>
    """
    
    positions = comp_results['input_grad']['positions']
    formulas = comp_results['input_grad']['formulas']
    values = comp_results['input_grad']['values']
    
    for i in range(len(positions)):
        pos = positions[i]
        formula = MathFormatter.format_tensor_element(formulas[i], display_mode=False)
        value = values[i]
        html += f"<tr><td>({pos[0]}, {pos[1]})</td><td>{formula}</td><td>{value:.6f}</td></tr>\n"
    
    html += """
    </table>
    
    <h4>Gradient Map (Receptive Field)</h4>
    """
    
    html += tensor_to_html_table(comp_results['input_grad']['map'])
    
    html += f"""
    <div class="formula">
        <h4>General Gradient Formula</h4>
        <div class="math-container">{general_formula}</div>
    </div>
    </div>
    """
    
    return html


def linear_backward_html(comp_results):
    """Generate HTML for Linear backward computation"""
    position = comp_results['position']
    
    # Format general formulas
    input_grad_formula_latex = r"\frac{\partial L}{\partial x_i} = \sum_{j} \frac{\partial L}{\partial y_j} \cdot w_{j,i}"
    weight_grad_formula_latex = r"\frac{\partial L}{\partial w_{j,i}} = \frac{\partial L}{\partial y_j} \cdot x_i"
    bias_grad_formula_latex = r"\frac{\partial L}{\partial b_j} = \frac{\partial L}{\partial y_j}"
    
    input_grad_formula = f"$${input_grad_formula_latex}$$"
    weight_grad_formula = f"$${weight_grad_formula_latex}$$"
    bias_grad_formula = f"$${bias_grad_formula_latex}$$"
    
    html = f"""
    <div class="backward-computation">
    <h3>Backward Computation for Linear at Position: batch = {position[0]}, output_feature = {position[1]}</h3>
    
    <h4>Gradient Output Value</h4>
    <p>{comp_results['grad_output_val']:.6f}</p>
    
    <h4>Input Features</h4>
    {tensor_to_html_table(comp_results['input_features'])}
    
    <h4>Output Weights</h4>
    {tensor_to_html_table(comp_results['output_weights'])}
    
    <h4>Gradient Propagation to Input</h4>
    <table class="tensor-table">
    <tr><th>Input Feature</th><th>Formula</th><th>Gradient Value</th></tr>
    """
    
    positions = comp_results['input_grad']['positions']
    formulas = comp_results['input_grad']['formulas']
    values = comp_results['input_grad']['values']
    
    for i in range(len(positions)):
        pos = positions[i]
        formula = MathFormatter.format_tensor_element(formulas[i], display_mode=False)
        value = values[i]
        html += f"<tr><td>{pos}</td><td>{formula}</td><td>{value:.6f}</td></tr>\n"
    
    html += """
    </table>
    
    <h4>Weight Gradient Computation</h4>
    <table class="tensor-table">
    <tr><th>Weight Position (out_f, in_f)</th><th>Formula</th><th>Gradient Value</th></tr>
    """
    
    positions = comp_results['weight_grad']['positions']
    formulas = comp_results['weight_grad']['formulas']
    values = comp_results['weight_grad']['values']
    
    for i in range(len(positions)):
        pos = positions[i]
        formula = MathFormatter.format_tensor_element(formulas[i], display_mode=False)
        value = values[i]
        html += f"<tr><td>({pos[0]}, {pos[1]})</td><td>{formula}</td><td>{value:.6f}</td></tr>\n"
    
    html += f"""
    </table>
    
    <h4>Bias Gradient Computation</h4>
    <p>Bias Gradient for output feature {position[1]}: {comp_results['bias_grad']:.6f}</p>
    
    <h3>General Gradient Formulas</h3>
    
    <div class="formula">
        <h4>Input Gradient Formula</h4>
        <div class="math-container">{input_grad_formula}</div>
    </div>
    
    <div class="formula">
        <h4>Weight Gradient Formula</h4>
        <div class="math-container">{weight_grad_formula}</div>
    </div>
    
    <div class="formula">
        <h4>Bias Gradient Formula</h4>
        <div class="math-container">{bias_grad_formula}</div>
    </div>
    </div>
    """
    
    return html


def relu_backward_html(comp_results):
    """Generate HTML for ReLU backward computation"""
    position = comp_results['position']
    pos_str = ", ".join([str(p) for p in position])
    
    # Format formula
    formula = MathFormatter.format_tensor_element(comp_results['formula'], display_mode=True)
    
    # Format general formula
    general_formula_latex = r"\frac{\partial L}{\partial x} = \begin{cases} \frac{\partial L}{\partial y} & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}"
    general_formula = f"$${general_formula_latex}$$"
    
    html = f"""
    <div class="backward-computation">
    <h3>Backward Computation for ReLU at Position: ({pos_str})</h3>
    
    <h4>Values at Position</h4>
    <p>Input Value: {comp_results['input_val']:.6f}</p>
    <p>Output Value: {comp_results['output_val']:.6f}</p>
    <p>Gradient Output Value: {comp_results['grad_output_val']:.6f}</p>
    
    <h4>Gradient Computation</h4>
    <div class="formula">
        <div class="math-container">Input Gradient = {formula}</div>
    </div>
    <p>Computed Gradient Value: {comp_results['grad_input']:.6f}</p>
    <p>Explanation: {comp_results['explanation']}</p>
    
    <div class="formula">
        <h4>General Gradient Formula</h4>
        <div class="math-container">{general_formula}</div>
    </div>
    </div>
    """
    
    return html


def format_backward_as_html(comp_results: Dict[str, Any]) -> str:
    """
    Format backward computation results as HTML
    
    Args:
        comp_results: The backward computation results
        
    Returns:
        HTML string with formatted backward computation
    """
    module_type = comp_results['module_type']
    
    if module_type == 'Conv2d':
        return conv2d_backward_html(comp_results)
    elif module_type == 'Linear':
        return linear_backward_html(comp_results)
    elif module_type == 'MaxPool2d':
        return maxpool2d_backward_html(comp_results)
    elif module_type == 'AvgPool2d':
        return avgpool2d_backward_html(comp_results)
    elif module_type == 'ReLU':
        return relu_backward_html(comp_results)
    else:
        return f"<p>Detailed backward formatting not implemented for module type: {module_type}</p>"


def get_backward_markdown(module: nn.Module, 
                         input_tensor: torch.Tensor,
                         output_tensor: torch.Tensor,
                         grad_output: torch.Tensor,
                         position: Optional[Tuple[int, ...]] = None) -> str:
    """
    Get detailed backward computation as markdown
    
    Args:
        module: The PyTorch module
        input_tensor: Input tensor to the module
        output_tensor: Output tensor from the module
        grad_output: Gradient of the loss with respect to the output
        position: Position for which to generate the computation
        
    Returns:
        Markdown string with backward computation details
    """
    from .backward_computation_utils import get_detailed_backward
    
    # Get backward computation details
    comp_results = get_detailed_backward(module, input_tensor, output_tensor, grad_output, position)
    
    # Format as markdown
    return format_backward_as_markdown(comp_results)


def get_backward_html(module: nn.Module, 
                     input_tensor: torch.Tensor,
                     output_tensor: torch.Tensor,
                     grad_output: torch.Tensor,
                     position: Optional[Tuple[int, ...]] = None) -> str:
    """
    Get detailed backward computation as HTML
    
    Args:
        module: The PyTorch module
        input_tensor: Input tensor to the module
        output_tensor: Output tensor from the module
        grad_output: Gradient of the loss with respect to the output
        position: Position for which to generate the computation
        
    Returns:
        HTML string with backward computation details
    """
    from .backward_computation_utils import get_detailed_backward
    
    # Get backward computation details
    comp_results = get_detailed_backward(module, input_tensor, output_tensor, grad_output, position)
    
    # Format as HTML
    return format_backward_as_html(comp_results)
