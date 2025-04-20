"""
Utilities for formatting tensors and presenting them in various formats
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

class TensorFormatter:
    """Formats tensor data for presentation in tables and reports"""
    
    @staticmethod
    def tensor_to_markdown_table(tensor: torch.Tensor, 
                               title: str = "Tensor Values",
                               max_rows: int = 10,
                               max_cols: int = 10) -> str:
        """
        Convert a tensor to a markdown table format
        
        Args:
            tensor: The tensor to convert
            title: Title of the table
            max_rows: Maximum number of rows to display
            max_cols: Maximum number of columns to display
            
        Returns:
            Markdown formatted table
        """
        tensor = tensor.detach().cpu()
        
        # Handle different tensor dimensions
        if tensor.dim() == 1:  # 1D tensor (vector)
            data = tensor.numpy()
            if len(data) > max_cols:
                # Sample if too long
                indices = np.round(np.linspace(0, len(data) - 1, max_cols)).astype(int)
                data = data[indices]
                col_headers = [f"[{i}]" for i in indices]
            else:
                col_headers = [f"[{i}]" for i in range(len(data))]
            
            # Generate markdown table
            md = f"### {title} (Shape: {tensor.shape})\n\n"
            md += "| " + " | ".join(col_headers) + " |\n"
            md += "| " + " | ".join(["---"] * len(col_headers)) + " |\n"
            md += "| " + " | ".join([f"{x:.4f}" for x in data]) + " |\n\n"
            
        elif tensor.dim() == 2:  # 2D tensor (matrix)
            data = tensor.numpy()
            rows, cols = data.shape
            
            # Sample rows and columns if too large
            if rows > max_rows:
                row_indices = np.round(np.linspace(0, rows - 1, max_rows)).astype(int)
                data = data[row_indices, :]
                row_headers = [f"[{i}]" for i in row_indices]
            else:
                row_headers = [f"[{i}]" for i in range(rows)]
            
            if cols > max_cols:
                col_indices = np.round(np.linspace(0, cols - 1, max_cols)).astype(int)
                data = data[:, col_indices]
                col_headers = [f"[{i}]" for i in col_indices]
            else:
                col_headers = [f"[{i}]" for i in range(cols)]
            
            # Generate markdown table
            md = f"### {title} (Shape: {tensor.shape})\n\n"
            md += "| Index | " + " | ".join(col_headers) + " |\n"
            md += "| --- | " + " | ".join(["---"] * len(col_headers)) + " |\n"
            
            for i, row in enumerate(data):
                md += f"| {row_headers[i]} | " + " | ".join([f"{x:.4f}" for x in row]) + " |\n"
            md += "\n"
            
        else:  # Higher dimensional tensors
            # Show the first slice of each dimension
            md = f"### {title} (Shape: {tensor.shape})\n\n"
            md += "> Note: Only showing first slice for high-dimensional tensor\n\n"
            
            # Get the first slice
            first_slice = tensor
            for _ in range(tensor.dim() - 2):
                first_slice = first_slice[0]
                
            # Recursively format the first slice
            slice_md = TensorFormatter.tensor_to_markdown_table(
                first_slice, 
                title=f"First Slice", 
                max_rows=max_rows, 
                max_cols=max_cols
            )
            md += slice_md
            
        return md
    
    @staticmethod
    def create_summary_table(tensor_dict: Dict[str, torch.Tensor]) -> pd.DataFrame:
        """
        Create a summary table of tensor statistics
        
        Args:
            tensor_dict: Dictionary of tensor names and values
            
        Returns:
            DataFrame with tensor statistics
        """
        rows = []
        
        for name, tensor in tensor_dict.items():
            # Convert tensor to numpy for statistics calculation
            tensor_np = tensor.detach().cpu().numpy()
            
            # Calculate statistics
            mean_val = np.mean(tensor_np)
            std_val = np.std(tensor_np)
            min_val = np.min(tensor_np)
            max_val = np.max(tensor_np)
            norm_val = np.linalg.norm(tensor_np)
            
            rows.append({
                'Name': name,
                'Shape': str(list(tensor.shape)),
                'Size': tensor.numel(),
                'Mean': mean_val,
                'Std': std_val,
                'Min': min_val,
                'Max': max_val,
                'Norm': norm_val
            })
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def value_substitution_example(module: torch.nn.Module, 
                                  input_tensor: torch.Tensor,
                                  output_tensor: torch.Tensor,
                                  position: Tuple[int, ...] = None) -> str:
        """
        Generate a value substitution example for a specific module
        
        Args:
            module: The PyTorch module
            input_tensor: Input tensor to the module
            output_tensor: Output tensor from the module
            position: Position in the tensor for which to generate the example (if None, use first element)
            
        Returns:
            String with the value substitution example
        """
        # Handle different module types
        if isinstance(module, torch.nn.Conv2d):
            return TensorFormatter._conv2d_value_substitution(
                module, input_tensor, output_tensor, position
            )
        elif isinstance(module, torch.nn.Linear):
            return TensorFormatter._linear_value_substitution(
                module, input_tensor, output_tensor, position
            )
        elif isinstance(module, torch.nn.MaxPool2d):
            return TensorFormatter._maxpool_value_substitution(
                module, input_tensor, output_tensor, position
            )
        elif isinstance(module, torch.nn.AvgPool2d):
            return TensorFormatter._avgpool_value_substitution(
                module, input_tensor, output_tensor, position
            )
        
        return f"Value substitution not implemented for {type(module).__name__}"
    
    @staticmethod
    def _conv2d_value_substitution(module, input_tensor, output_tensor, position=None):
        """Generate value substitution for Conv2d module"""
        # Simplified implementation for now
        return "Conv2d calculation example (simplified)"
    
    @staticmethod
    def _linear_value_substitution(module, input_tensor, output_tensor, position=None):
        """Generate value substitution for Linear module"""
        # Simplified implementation for now
        return "Linear calculation example (simplified)"
    
    @staticmethod
    def _maxpool_value_substitution(module, input_tensor, output_tensor, position=None):
        """Generate value substitution for MaxPool2d module"""
        # Simplified implementation for now
        return "MaxPool2d calculation example (simplified)"
    
    @staticmethod
    def _avgpool_value_substitution(module, input_tensor, output_tensor, position=None):
        """Generate value substitution for AvgPool2d module"""
        # Simplified implementation for now
        return "AvgPool2d calculation example (simplified)"
