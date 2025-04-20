"""
Specialized formatter for mathematical formulas in PyTorch analysis.
"""

class MathFormatter:
    """Class for handling specialized math formula formatting"""
    
    @staticmethod
    def format_convolution_forward(display_mode=True):
        """
        Returns the general formula for convolution forward pass
        
        Args:
            display_mode: Whether to use display mode (centered) or inline mode
            
        Returns:
            Formatted LaTeX formula
        """
        formula = r"y_{n,c_{out},h_{out},w_{out}} = \sum_{c_{in}} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}"
        
        if display_mode:
            return f"$${formula}$$"
        else:
            return f"${formula}$"
    
    @staticmethod
    def format_convolution_simplified(display_mode=True):
        """
        Returns the simplified formula for convolution forward pass
        
        Args:
            display_mode: Whether to use display mode (centered) or inline mode
            
        Returns:
            Formatted LaTeX formula
        """
        formula = r"y_{n,c_{out},h_{out},w_{out}} = \sum_{c_{in},k_h,k_w} x_{n,c_{in},h_{in}+k_h,w_{in}+k_w} \cdot w_{c_{out},c_{in},k_h,k_w} + b_{c_{out}}"
        
        if display_mode:
            return f"$${formula}$$"
        else:
            return f"${formula}$"
    
    @staticmethod
    def format_input_gradient(display_mode=True):
        """
        Returns the formula for input gradient in backpropagation
        
        Args:
            display_mode: Whether to use display mode (centered) or inline mode
            
        Returns:
            Formatted LaTeX formula
        """
        formula = r"\frac{\partial L}{\partial x_{n,c_{in},h_{in},w_{in}}} = \sum_{c_{out}} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot w_{c_{out},c_{in},k_h,k_w}"
        
        if display_mode:
            return f"$${formula}$$"
        else:
            return f"${formula}$"
    
    @staticmethod
    def format_weight_gradient(display_mode=True):
        """
        Returns the formula for weight gradient in backpropagation
        
        Args:
            display_mode: Whether to use display mode (centered) or inline mode
            
        Returns:
            Formatted LaTeX formula
        """
        formula = r"\frac{\partial L}{\partial w_{c_{out},c_{in},k_h,k_w}} = \sum_{n} \sum_{h_{out},w_{out}} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot x_{n,c_{in},h_{in}+k_h,w_{in}+k_w}"
        
        if display_mode:
            return f"$${formula}$$"
        else:
            return f"${formula}$"
    
    @staticmethod
    def format_bias_gradient(display_mode=True):
        """
        Returns the formula for bias gradient in backpropagation
        
        Args:
            display_mode: Whether to use display mode (centered) or inline mode
            
        Returns:
            Formatted LaTeX formula
        """
        formula = r"\frac{\partial L}{\partial b_{c_{out}}} = \sum_{n} \sum_{h_{out},w_{out}} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}}"
        
        if display_mode:
            return f"$${formula}$$"
        else:
            return f"${formula}$"
    
    @staticmethod
    def format_value_substitution(values_str, display_mode=True):
        """
        Format a value substitution expression for convolution
        
        Args:
            values_str: String containing the values to substitute
            display_mode: Whether to use display mode (centered) or inline mode
            
        Returns:
            Formatted LaTeX formula
        """
        # Clean up the input string - replace plain text operators with LaTeX ones
        values_str = values_str.replace('×', r' \times ')
        values_str = values_str.replace('*', r' \times ')
        values_str = values_str.replace('−', '-')  # Replace unicode minus with hyphen
        
        if display_mode:
            return f"$${values_str}$$"
        else:
            return f"${values_str}$"
    
    @staticmethod
    def format_tensor_element(tensor_expr, display_mode=True):
        """
        Format a tensor element expression like (-0.0014 × 0.3000)
        
        Args:
            tensor_expr: Expression for the tensor element calculation
            display_mode: Whether to use display mode (centered) or inline mode
            
        Returns:
            Formatted LaTeX formula
        """
        # Clean up the input string - replace plain text operators with LaTeX ones
        tensor_expr = tensor_expr.replace('×', r' \times ')
        tensor_expr = tensor_expr.replace('*', r' \times ')
        tensor_expr = tensor_expr.replace('−', '-')  # Replace unicode minus with hyphen
        
        if display_mode:
            return f"$${tensor_expr}$$"
        else:
            return f"${tensor_expr}$"
