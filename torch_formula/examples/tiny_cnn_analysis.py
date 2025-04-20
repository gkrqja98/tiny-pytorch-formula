"""
Example script demonstrating the analysis of TinyCNN model
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_formula.models import TinyCNN
from torch_formula.core.model_analyzer import ModelAnalyzer
from torch_formula.viz.computation_graph import (
    create_computation_graph, visualize_computation_graph,
    create_layerwise_graph, visualize_layerwise_graph
)

def main():
    """Main function to demonstrate the functionality"""
    print("Creating TinyCNN model...")
    model = TinyCNN()
    
    # Create sample input
    input_shape = (1, 1, 8, 8)
    input_tensor = torch.randn(*input_shape)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Analyzing model...")
    analyzer = ModelAnalyzer(model, input_shape)
    output, loss = analyzer.run_model(input_tensor)
    
    print(f"Model output shape: {output.shape}")
    print(f"Loss: {loss.item()}")
    
    # Get layer summaries
    print("\nLayer Summaries:")
    all_layers = analyzer.get_all_layers_summary(format='markdown')
    
    # Save layer summaries to file
    with open(os.path.join(output_dir, "layer_summaries.md"), "w") as f:
        f.write("# TinyCNN Layer Summaries\n\n")
        f.write(all_layers)
    
    print("Layer summaries saved to", os.path.join(output_dir, "layer_summaries.md"))
    
    # Visualize computation graph
    print("\nCreating computation graph visualization...")
    G = create_computation_graph(model, input_shape)
    visualize_computation_graph(G, filename=os.path.join(output_dir, "computation_graph"))
    
    # Visualize layerwise graph
    print("Creating layerwise graph visualization...")
    G, forward_layers, backward_layers = create_layerwise_graph(model, input_shape)
    visualize_layerwise_graph(G, forward_layers, backward_layers, 
                            filename=os.path.join(output_dir, "layerwise_graph"))
    
    print("Visualizations saved to", output_dir)
    
    # Clean up resources
    analyzer.cleanup()
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
