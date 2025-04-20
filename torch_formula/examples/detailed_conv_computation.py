"""
Example script demonstrating detailed computation for Conv2d layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pandas as pd

# Add the parent directory to the path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_formula.models import TinyCNN
from torch_formula.core.model_analyzer import ModelAnalyzer
from torch_formula.utils.detailed_computation import get_computation_markdown, get_computation_html

def main():
    """Main function to demonstrate detailed computation for Conv2d layers"""
    print("Creating TinyCNN model...")
    model = TinyCNN()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Fix random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create a simple 8x8 input with a pattern for easier visualization
    input_tensor = torch.zeros(1, 1, 8, 8)
    # Add a pattern (e.g., cross)
    input_tensor[0, 0, 3:5, :] = 1.0  # Horizontal line
    input_tensor[0, 0, :, 3:5] = 1.0  # Vertical line
    # Add some random noise
    input_tensor += 0.1 * torch.randn_like(input_tensor)
    
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Analyze model
    print("Analyzing model...")
    analyzer = ModelAnalyzer(model, input_tensor.shape)
    output, loss = analyzer.run_model(input_tensor)
    
    print(f"Model output shape: {output.shape}")
    print(f"Loss: {loss.item()}")
    
    # Get detailed computation for each layer
    layer_names = ["conv1", "pool1", "conv2", "pool2", "conv3", "conv_out"]
    
    for layer_name in layer_names:
        print(f"\nGenerating detailed computation for {layer_name}...")
        
        # Get detailed computation
        markdown_content = analyzer.get_detailed_computation(layer_name, format='markdown')
        html_content = analyzer.get_detailed_computation(layer_name, format='html')
        
        # Save to files
        md_file = os.path.join(output_dir, f"{layer_name}_detailed.md")
        html_file = os.path.join(output_dir, f"{layer_name}_detailed.html")
        
        with open(md_file, "w") as f:
            f.write(f"# Detailed Computation for {layer_name}\n\n")
            f.write(markdown_content)
        
        with open(html_file, "w") as f:
            f.write("<html><head><title>Detailed Computation</title>")
            f.write("<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML'></script>")
            f.write("<style>body { font-family: Arial, sans-serif; margin: 20px; }</style>")
            f.write("</head><body>")
            f.write(f"<h1>Detailed Computation for {layer_name}</h1>")
            f.write(html_content)
            f.write("</body></html>")

    # Get comprehensive layer summary
    print("\nGenerating comprehensive layer summary...")
    comprehensive_md = analyzer.get_all_layers_summary(format='markdown', include_detailed_computation=True)
    
    with open(os.path.join(output_dir, "comprehensive_layer_summary.md"), "w") as f:
        f.write("# Comprehensive Layer Analysis for TinyCNN\n\n")
        f.write(comprehensive_md)

    # Generate HTML report with all layers
    print("Generating HTML report...")
    html_report = """
    <html>
    <head>
        <title>TinyCNN Layer Analysis</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .section { margin-bottom: 30px; }
            .tensor-table { border-collapse: collapse; margin: 10px 0; }
            .tensor-table th, .tensor-table td { border: 1px solid #ddd; padding: 8px; text-align: right; }
            .tensor-table th { background-color: #f2f2f2; }
            .nav { position: fixed; top: 10px; right: 10px; background: #fff; 
                  border: 1px solid #ddd; padding: 10px; max-height: 80vh; overflow-y: auto; }
            .nav ul { padding-left: 20px; }
            .nav a { text-decoration: none; }
        </style>
    </head>
    <body>
        <h1>TinyCNN Layer Analysis</h1>
        
        <div class="nav">
            <h3>Navigation</h3>
            <ul>
    """
    
    # Add navigation links
    for layer_name in layer_names:
        html_report += f'<li><a href="#{layer_name}">{layer_name}</a></li>\n'
    
    html_report += """
            </ul>
        </div>
        
        <div class="content">
    """
    
    # Add detailed computation for each layer
    for layer_name in layer_names:
        html_report += f'<div id="{layer_name}" class="section">\n'
        html_report += analyzer.get_layer_summary(layer_name, format='html', include_detailed_computation=True)
        html_report += '</div>\n'
    
    html_report += """
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "layer_analysis_report.html"), "w") as f:
        f.write(html_report)
    
    print(f"All files saved to {output_dir}")
    
    # Clean up
    analyzer.cleanup()
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
