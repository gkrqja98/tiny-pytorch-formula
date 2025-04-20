"""
Workflow for analyzing PyTorch models.
"""
import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any

class AnalysisWorkflow:
    """PyTorch model forward/backward analysis workflow"""
    
    def __init__(self, model: nn.Module, input_shape: tuple, output_dir: str = "./output"):
        """
        Initialize the analysis workflow
        
        Args:
            model: PyTorch model to analyze
            input_shape: Shape of the input tensor
            output_dir: Directory to save the analysis results
        """
        self.model = model
        self.input_shape = input_shape
        self.output_dir = output_dir
        
        # Import here to avoid circular imports
        from torch_formula.core.model_analyzer import ModelAnalyzer
        self.analyzer = ModelAnalyzer(model, input_shape)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def run_full_analysis(self, input_tensor: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Run full model analysis
        
        Args:
            input_tensor: Input tensor for the model. If None, a random tensor is created
            
        Returns:
            Dictionary with analysis results
        """
        # Import the HTML renderer
        from torch_formula.utils.html_renderer import HTMLRenderer
        
        # Run the model analyzer
        output, loss = self.analyzer.run_model(input_tensor)
        
        # Get all layers summary
        all_layers_html = self.analyzer.get_all_layers_summary(format="html", include_detailed_computation=True)
        
        # Create HTML content
        content = "<h1>Complete Forward and Backward Analysis</h1>\n"
        content += f"<p><strong>Input Shape:</strong> {self.input_shape}</p>\n"
        content += f"<p><strong>Output Shape:</strong> {output.shape}</p>\n"
        content += f"<p><strong>Loss Value:</strong> {loss.item():.6f}</p>\n"
        content += all_layers_html
        
        # Wrap in template and save
        html_content = HTMLRenderer.wrap_in_template(content, "PyTorch Model Analysis")
        html_path = os.path.join(self.output_dir, "full_model_analysis.html")
        HTMLRenderer.save_html(html_content, html_path)
        
        print(f"Full model analysis saved to: {html_path}")
        
        return {
            "output": output,
            "loss": loss,
            "html_path": html_path
        }
    
    def analyze_specific_layer(self, layer_name: str, position: Optional[Tuple[int, ...]] = None) -> Dict[str, Any]:
        """
        Analyze a specific layer of the model
        
        Args:
            layer_name: Name of the layer to analyze
            position: Position tuple for detailed analysis
            
        Returns:
            Dictionary with analysis results
        """
        # Import the HTML renderer
        from torch_formula.utils.html_renderer import HTMLRenderer
        
        # Get module type
        module_type = "unknown"
        for name, module in self.model.named_modules():
            if name == layer_name:
                module_type = type(module).__name__
        
        # Get forward computation
        forward_html = self.analyzer.get_detailed_computation(layer_name, position, format="html")
        
        # Get backward computation
        backward_html = self.analyzer.get_detailed_backward(layer_name, position, format="html")
        
        # Create HTML content
        pos_str = "N/A" if position is None else str(position)
        content = f"<h1>Analysis for {module_type} Layer '{layer_name}'</h1>\n"
        content += f"<p><strong>Position:</strong> {pos_str}</p>\n"
        
        content += "<div class='forward-pass'>\n"
        content += "<h2>Forward Pass</h2>\n"
        content += forward_html
        content += "</div>\n"
        
        content += "<div class='backward-pass'>\n"
        content += "<h2>Backward Pass</h2>\n"
        content += backward_html
        content += "</div>\n"
        
        # Wrap in template and save
        html_content = HTMLRenderer.wrap_in_template(content, f"{layer_name} Analysis")
        html_path = os.path.join(self.output_dir, f"{layer_name}_analysis.html")
        HTMLRenderer.save_html(html_content, html_path)
        
        print(f"Layer analysis for {layer_name} saved to: {html_path}")
        
        return {
            "layer_name": layer_name,
            "position": position,
            "html_path": html_path
        }
