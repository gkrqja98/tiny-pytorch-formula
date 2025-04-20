"""
Utility functions for torch_formula
"""
# Import from new modular structure
from .tensor_formatter import tensor_to_html_table, format_tensor_as_table
from .html_renderer import HTMLRenderer, render_template, create_comprehensive_report_html
from .visualization import visualize_tensor_data, visualize_gradient_flow, create_computation_graph

__all__ = [
    'tensor_to_html_table', 
    'format_tensor_as_table',
    'HTMLRenderer',
    'render_template',
    'create_comprehensive_report_html',
    'visualize_tensor_data',
    'visualize_gradient_flow',
    'create_computation_graph'
]
