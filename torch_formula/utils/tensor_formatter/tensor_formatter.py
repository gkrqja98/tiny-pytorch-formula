"""
Utilities for formatting tensors into HTML representations.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

def tensor_to_html_table(tensor: torch.Tensor, 
                       title: str = "Tensor Values",
                       max_rows: int = 10,
                       max_cols: int = 10) -> str:
    """
    Convert a tensor to an HTML table format
    
    Args:
        tensor: The tensor to convert
        title: Title of the table
        max_rows: Maximum number of rows to display
        max_cols: Maximum number of columns to display
        
    Returns:
        HTML formatted table
    """
    tensor = tensor.detach().cpu()
    
    html = f"<h3>{title} (Shape: {tensor.shape})</h3>\n\n"
    
    # Handle different tensor dimensions
    if tensor.dim() == 1:  # 1D tensor (vector)
        html += "<table class='tensor-table'>\n"
        html += "<tr><th>Index</th>"
        for i in range(min(tensor.shape[0], max_cols)):
            html += f"<th>[{i}]</th>"
        html += "</tr>\n<tr><td>Value</td>"
        
        for i in range(min(tensor.shape[0], max_cols)):
            html += f"<td>{tensor[i].item():.4f}</td>"
        html += "</tr>\n</table>\n"
        
    elif tensor.dim() == 2:  # 2D tensor (matrix)
        html += "<table class='tensor-table'>\n"
        html += "<tr><th>Index</th>"
        
        for j in range(min(tensor.shape[1], max_cols)):
            html += f"<th>[{j}]</th>"
        html += "</tr>\n"
        
        for i in range(min(tensor.shape[0], max_rows)):
            html += f"<tr><th>[{i}]</th>"
            for j in range(min(tensor.shape[1], max_cols)):
                html += f"<td>{tensor[i, j].item():.4f}</td>"
            html += "</tr>\n"
        html += "</table>\n"
        
    elif tensor.dim() == 3:  # 3D tensor (channels/height/width)
        # Each channel gets its own table section
        html += "<table class='tensor-table'>\n"
        
        for c in range(min(tensor.shape[0], 3)):  # Maximum of 3 channels for readability
            html += f"<tr><th colspan='{min(tensor.shape[2], max_cols) + 1}' class='tensor-header'>Channel [{c}]</th></tr>\n"
            html += "<tr><th>Index</th>"
            
            for w in range(min(tensor.shape[2], max_cols)):
                html += f"<th>[{w}]</th>"
            html += "</tr>\n"
            
            for h in range(min(tensor.shape[1], max_rows)):
                html += f"<tr><th>[{h}]</th>"
                for w in range(min(tensor.shape[2], max_cols)):
                    html += f"<td>{tensor[c, h, w].item():.4f}</td>"
                html += "</tr>\n"
                
            if c < min(tensor.shape[0], 3) - 1:
                html += f"<tr><td colspan='{min(tensor.shape[2], max_cols) + 1}' class='channel-separator'></td></tr>\n"
                
        html += "</table>\n"
        
    elif tensor.dim() == 4:  # 4D tensor (batch/channels/height/width)
        # Each batch and channel combination gets its own table section
        html += "<table class='tensor-table'>\n"
        
        for b in range(min(tensor.shape[0], 2)):  # Maximum of 2 batches for readability
            html += f"<tr><th colspan='{min(tensor.shape[3], max_cols) + 1}' class='tensor-header'>Batch [{b}]</th></tr>\n"
            
            for c in range(min(tensor.shape[1], 3)):  # Maximum of 3 channels for readability
                html += f"<tr><th colspan='{min(tensor.shape[3], max_cols) + 1}' class='tensor-header'>Channel [{c}]</th></tr>\n"
                html += "<tr><th>Index</th>"
                
                for w in range(min(tensor.shape[3], max_cols)):
                    html += f"<th>[{w}]</th>"
                html += "</tr>\n"
                
                for h in range(min(tensor.shape[2], max_rows)):
                    html += f"<tr><th>[{h}]</th>"
                    for w in range(min(tensor.shape[3], max_cols)):
                        html += f"<td>{tensor[b, c, h, w].item():.4f}</td>"
                    html += "</tr>\n"
                    
                if c < min(tensor.shape[1], 3) - 1:
                    html += f"<tr><td colspan='{min(tensor.shape[3], max_cols) + 1}' class='channel-separator'></td></tr>\n"
                    
            if b < min(tensor.shape[0], 2) - 1:
                html += f"<tr><td colspan='{min(tensor.shape[3], max_cols) + 1}' class='batch-separator'></td></tr>\n"
                
        html += "</table>\n"
        
    else:  # Higher dimensional tensors
        html += "<p>Note: Only showing first slice for high-dimensional tensor</p>\n"
        first_slice = tensor
        for _ in range(tensor.dim() - 2):
            first_slice = first_slice[0]
        html += tensor_to_html_table(first_slice, f"First Slice", max_rows, max_cols)
        
    return html

def format_tensor_as_table(tensor: torch.Tensor, 
                         title: str = "Tensor Values",
                         format_type: str = "html",
                         max_rows: int = 10,
                         max_cols: int = 10) -> str:
    """
    Format a tensor as a table in various formats
    
    Args:
        tensor: The tensor to format
        title: Title of the table
        format_type: Format type ('html', 'markdown', etc.)
        max_rows: Maximum number of rows to display
        max_cols: Maximum number of columns to display
        
    Returns:
        Formatted table as string
    """
    if format_type.lower() == 'html':
        return tensor_to_html_table(tensor, title, max_rows, max_cols)
    else:
        # Fallback to simple string representation
        return f"{title} (Shape: {tensor.shape})\n{tensor}"

def tensor_to_formatted_array(tensor: torch.Tensor, decimals: int = 4) -> str:
    """
    Convert a tensor to a formatted array string
    
    Args:
        tensor: The tensor to convert
        decimals: Number of decimal places to show
        
    Returns:
        Formatted array string
    """
    tensor = tensor.detach().cpu()
    tensor_np = tensor.numpy()
    
    # Use numpy to format array
    with np.printoptions(precision=decimals, suppress=True, threshold=np.inf):
        return str(tensor_np)
