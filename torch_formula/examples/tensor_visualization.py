"""
Example script demonstrating tensor visualization and detailed computation presentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pandas as pd
import json

# Add the parent directory to the path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_formula.models import TinyCNN
from torch_formula.core.model_analyzer import ModelAnalyzer
from torch_formula.utils.detailed_computation import (
    sample_region,
    format_tensor_as_markdown_table,
    format_detailed_computation_as_html
)

def create_visualizations(tensor, output_dir, name):
    """Create visualizations for a tensor"""
    if tensor.dim() <= 2:
        # For 1D or 2D tensors
        df = pd.DataFrame(tensor.cpu().numpy())
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(df, cmap='viridis')
        plt.colorbar(im, ax=ax)
        plt.title(f"{name} Values")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name.lower().replace(' ', '_')}_heatmap.png"))
        plt.close()
    
    elif tensor.dim() >= 3:
        # For 3D+ tensors, visualize first few channels
        max_channels = min(4, tensor.shape[1])
        
        fig, axes = plt.subplots(1, max_channels, figsize=(15, 4))
        if max_channels == 1:
            axes = [axes]  # Make it iterable
            
        for i in range(max_channels):
            im = axes[i].imshow(tensor[0, i].cpu().numpy(), cmap='viridis')
            axes[i].set_title(f"Channel {i}")
            plt.colorbar(im, ax=axes[i])
        
        plt.suptitle(f"{name} - Channel Visualizations")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name.lower().replace(' ', '_')}_channels.png"))
        plt.close()

def generate_html_report(analyzer, layer_name, output_dir):
    """Generate a detailed HTML report for a layer"""
    if layer_name not in analyzer.activations:
        print(f"Layer {layer_name} not found")
        return
    
    # Get relevant data
    module_type = ""
    for name, module in analyzer.model.named_modules():
        if name == layer_name:
            module_type = type(module).__name__
            break
    
    input_tensor = analyzer.activations[layer_name]['input']
    output_tensor = analyzer.activations[layer_name]['output']
    
    # Create visualizations
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    create_visualizations(input_tensor, vis_dir, f"{layer_name}_input")
    create_visualizations(output_tensor, vis_dir, f"{layer_name}_output")
    
    # Generate detailed computation
    detailed_comp = analyzer.get_detailed_computation(layer_name, format='html')
    
    # Create HTML report
    html_report = f"""
    <html>
    <head>
        <title>{layer_name} - Detailed Analysis</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            .section {{ margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .tensor-table {{ border-collapse: collapse; margin: 10px 0; width: 100%; }}
            .tensor-table th, .tensor-table td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
            .tensor-table th {{ background-color: #f2f2f2; }}
            .visualization {{ text-align: center; margin: 20px 0; }}
            .formula {{ padding: 10px; background-color: #f9f9f9; border-left: 3px solid #2196F3; margin: 10px 0; }}
            .computation {{ padding: 15px; background-color: #f5f5f5; border-radius: 5px; margin: 15px 0; }}
            .result {{ font-weight: bold; color: #2196F3; }}
            h2, h3 {{ color: #333; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
            .nav {{ position: fixed; top: 10px; right: 10px; background: #fff; 
                  border: 1px solid #ddd; padding: 10px; border-radius: 5px; 
                  box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <div class="nav">
            <a href="#overview">Overview</a> | 
            <a href="#input">Input</a> | 
            <a href="#computation">Computation</a> | 
            <a href="#output">Output</a>
        </div>
        
        <h1>{layer_name} - {module_type} Layer Analysis</h1>
        
        <div id="overview" class="section">
            <h2>Layer Overview</h2>
            <p><strong>Layer Name:</strong> {layer_name}</p>
            <p><strong>Layer Type:</strong> {module_type}</p>
            <p><strong>Input Shape:</strong> {input_tensor.shape}</p>
            <p><strong>Output Shape:</strong> {output_tensor.shape}</p>
        </div>
        
        <div id="input" class="section">
            <h2>Input Tensor Analysis</h2>
            <p>Statistical summary of the input tensor:</p>
            <ul>
                <li><strong>Min:</strong> {input_tensor.min().item():.6f}</li>
                <li><strong>Max:</strong> {input_tensor.max().item():.6f}</li>
                <li><strong>Mean:</strong> {input_tensor.mean().item():.6f}</li>
                <li><strong>Std Dev:</strong> {input_tensor.std().item():.6f}</li>
            </ul>
            
            <div class="visualization">
                <h3>Input Tensor Visualization</h3>
                <img src="visualizations/{layer_name}_input_channels.png" alt="{layer_name} Input Channels">
            </div>
        </div>
        
        <div id="computation" class="section">
            <h2>Detailed Computation</h2>
            {detailed_comp}
        </div>
        
        <div id="output" class="section">
            <h2>Output Tensor Analysis</h2>
            <p>Statistical summary of the output tensor:</p>
            <ul>
                <li><strong>Min:</strong> {output_tensor.min().item():.6f}</li>
                <li><strong>Max:</strong> {output_tensor.max().item():.6f}</li>
                <li><strong>Mean:</strong> {output_tensor.mean().item():.6f}</li>
                <li><strong>Std Dev:</strong> {output_tensor.std().item():.6f}</li>
            </ul>
            
            <div class="visualization">
                <h3>Output Tensor Visualization</h3>
                <img src="visualizations/{layer_name}_output_channels.png" alt="{layer_name} Output Channels">
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(os.path.join(output_dir, f"{layer_name}_detailed_analysis.html"), "w") as f:
        f.write(html_report)

def main():
    """Main function to demonstrate tensor visualization and detailed computation presentation"""
    print("Creating TinyCNN model...")
    model = TinyCNN()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "tensor_visualization")
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
    
    # Save input visualization
    create_visualizations(input_tensor, output_dir, "Input_Tensor")
    
    # Analyze model
    print("Analyzing model...")
    analyzer = ModelAnalyzer(model, input_tensor.shape)
    output, loss = analyzer.run_model(input_tensor)
    
    print(f"Model output shape: {output.shape}")
    
    # Generate HTML reports for each layer
    layer_names = ["conv1", "pool1", "conv2", "pool2", "conv3", "conv_out"]
    
    for layer_name in layer_names:
        print(f"Generating HTML report for {layer_name}...")
        generate_html_report(analyzer, layer_name, output_dir)
    
    # Create main index page
    index_html = """
    <html>
    <head>
        <title>TinyCNN - Tensor Visualization & Detailed Computation</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            .card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin: 15px 0; }
            .card h2 { margin-top: 0; color: #333; }
            .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
            a { text-decoration: none; color: #2196F3; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>TinyCNN - Tensor Visualization & Detailed Computation</h1>
        
        <div class="card">
            <h2>Model Overview</h2>
            <p>This report shows detailed computation and tensor visualization for the TinyCNN model.</p>
            <p>Input Shape: (1, 1, 8, 8)</p>
            <p>Output Shape: (1, 2)</p>
            <div style="text-align: center;">
                <img src="Input_Tensor_channels.png" alt="Input Tensor" style="max-width: 400px;">
            </div>
        </div>
        
        <h2>Layer Analysis</h2>
        <div class="grid">
    """
    
    for layer_name in layer_names:
        index_html += f"""
            <div class="card">
                <h2>{layer_name}</h2>
                <p>Click below to view detailed analysis:</p>
                <p><a href="{layer_name}_detailed_analysis.html">View {layer_name} Analysis</a></p>
            </div>
        """
    
    index_html += """
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(index_html)
    
    print(f"All files saved to {output_dir}")
    
    # Clean up
    analyzer.cleanup()
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
