"""
Utilities for formatting tensors and presenting them in various formats.
"""
from .tensor_formatter import (
    tensor_to_html_table,
    format_tensor_as_table,
    tensor_to_formatted_array
)

__all__ = [
    'tensor_to_html_table',
    'format_tensor_as_table',
    'tensor_to_formatted_array'
]
