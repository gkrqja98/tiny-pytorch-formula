"""
Package for detailed computation of neural network operations
"""
from .detailed_computation_utils import (
    sample_region,
    get_detailed_computation,
    conv2d_detailed_computation,
    maxpool2d_detailed_computation,
    avgpool2d_detailed_computation,
    linear_detailed_computation
)

from .formatters import (
    format_tensor_as_markdown_table,
    format_detailed_computation_as_markdown,
    format_detailed_computation_as_html,
    get_computation_markdown,
    get_computation_html
)

from .backward_computation_utils import (
    get_detailed_backward,
    conv2d_detailed_backward,
    maxpool2d_detailed_backward,
    avgpool2d_detailed_backward,
    linear_detailed_backward,
    relu_detailed_backward
)

from .backward_formatters import (
    format_backward_as_markdown,
    format_backward_as_html,
    get_backward_markdown,
    get_backward_html
)

__all__ = [
    # Forward computation utilities
    'sample_region',
    'get_detailed_computation',
    'conv2d_detailed_computation',
    'maxpool2d_detailed_computation',
    'avgpool2d_detailed_computation',
    'linear_detailed_computation',
    'format_tensor_as_markdown_table',
    'format_detailed_computation_as_markdown',
    'format_detailed_computation_as_html',
    'get_computation_markdown',
    'get_computation_html',
    
    # Backward computation utilities
    'get_detailed_backward',
    'conv2d_detailed_backward',
    'maxpool2d_detailed_backward',
    'avgpool2d_detailed_backward',
    'linear_detailed_backward',
    'relu_detailed_backward',
    'format_backward_as_markdown',
    'format_backward_as_html',
    'get_backward_markdown',
    'get_backward_html'
]
