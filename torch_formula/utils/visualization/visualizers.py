"""
Visualization functions for PyTorch model analysis.
"""
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

def visualize_tensor_data(tensor: torch.Tensor, output_path: str) -> str:
    """
    Visualize tensor data with heatmaps
    
    Args:
        tensor: The tensor to visualize
        output_path: Path to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    # Ensure extension
    if not output_path.endswith('.png'):
        output_path += '.png'
    
    # Create directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Detach and move to CPU
    tensor = tensor.detach().cpu()
    
    if tensor.dim() == 4:  # Batched image tensor
        # Use the first batch
        tensor = tensor[0]
    
    if tensor.dim() == 3:  # Channel/Height/Width
        # Create a figure with subplots for each channel
        n_channels = tensor.shape[0]
        fig, axes = plt.subplots(1, n_channels, figsize=(n_channels * 4, 4))
        
        # Handle the case where there's only one channel
        if n_channels == 1:
            axes = [axes]
            
        for c in range(n_channels):
            im = axes[c].imshow(tensor[c].numpy(), cmap='viridis')
            axes[c].set_title(f'Channel {c}')
            axes[c].axis('off')
            fig.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04)
            
    elif tensor.dim() == 2:  # Height/Width
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(tensor.numpy(), cmap='viridis')
        ax.set_title('Tensor Values')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    else:  # 1D or higher dims - use a basic line plot or fallback
        if tensor.dim() == 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(tensor.numpy())
            ax.set_title('Tensor Values')
            ax.grid(True)
        else:
            # Fallback for higher dimensions - flatten and plot
            fig, ax = plt.subplots(figsize=(10, 6))
            flattened = tensor.flatten().numpy()
            ax.plot(flattened)
            ax.set_title(f'Flattened Tensor (Shape: {tensor.shape})')
            ax.grid(True)
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def visualize_gradient_flow(model: nn.Module, input_tensor: torch.Tensor, output_path: str) -> str:
    """
    Visualize gradient flow through the model
    
    Args:
        model: The PyTorch model
        input_tensor: Input tensor to the model
        output_path: Path to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    # Ensure extension
    if not output_path.endswith('.png'):
        output_path += '.png'
    
    # Create directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Enable gradient tracking
    input_tensor.requires_grad_(True)
    
    # Forward pass
    output = model(input_tensor)
    
    # Create a target and loss for backward pass
    if output.dim() > 0:
        target = torch.zeros_like(output)
        if output.numel() > 0:
            target.flatten()[0] = 1.0  # Set a target for the first element
    else:
        target = torch.tensor([0.0])  # Fallback for scalar output
    
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    
    # Backward pass
    loss.backward()
    
    # Collect gradients
    gradients = {}
    grad_norms = []
    layer_names = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach().cpu()
            gradients[name] = grad
            norm = torch.norm(grad).item()
            grad_norms.append(norm)
            layer_names.append(name.replace('.weight', '').replace('.bias', ''))
    
    # Visualize gradient norms
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(grad_norms)), grad_norms, color='skyblue')
    plt.title('Gradient Norms Across Layers', fontsize=14)
    plt.ylabel('Gradient Norm', fontsize=12)
    plt.xlabel('Layer', fontsize=12)
    plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_computation_graph(model: nn.Module, input_shape: Tuple[int, ...], output_path: str) -> str:
    """
    Create a computation graph for the model
    
    Args:
        model: The PyTorch model
        input_shape: Shape of the input tensor
        output_path: Path to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    # Ensure extension
    if not output_path.endswith('.png'):
        output_path += '.png'
    
    # Create directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Create a figure
    plt.figure(figsize=(12, 10))
    
    # Create a graph representation
    G = create_model_graph(model)
    
    try:
        import networkx as nx
        
        # Create positions
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        plt.title("Model Computation Graph")
        plt.axis('off')
        
    except ImportError:
        plt.figtext(0.5, 0.5, "NetworkX library is required for graph visualization", 
                   ha="center", va="center", fontsize=12, color="red")
        plt.title("Computation Graph (NetworkX required)")
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_model_graph(model: nn.Module):
    """
    Create a graph representation of the model
    
    Args:
        model: The PyTorch model
        
    Returns:
        NetworkX graph object
    """
    try:
        import networkx as nx
        
        G = nx.DiGraph()
        
        # Add input node
        G.add_node("input", type="input")
        
        # Track previous nodes for connection
        prev_nodes = ["input"]
        
        # Process named modules
        for name, module in model.named_modules():
            if name == "":  # Skip the model itself
                continue
                
            # Check if this is a "leaf" module (not containing other modules)
            is_leaf = True
            for child_name, _ in model.named_modules():
                if child_name != name and child_name.startswith(name + "."):
                    is_leaf = False
                    break
            
            if is_leaf:
                # Add the module as a node
                module_type = type(module).__name__
                G.add_node(name, type=module_type)
                
                # Connect from previous nodes
                for prev in prev_nodes:
                    G.add_edge(prev, name)
                
                # Update previous nodes
                if name.count(".") == 0:  # Only top-level modules reset the chain
                    prev_nodes = [name]
                else:
                    # For nested modules, maintain the chain
                    if prev_nodes[-1] != name:
                        prev_nodes.append(name)
        
        # Add output node and connect to last nodes
        G.add_node("output", type="output")
        for node in prev_nodes:
            G.add_edge(node, "output")
        
        return G
        
    except ImportError:
        # Create an empty graph if NetworkX is not available
        class DummyGraph:
            def __init__(self):
                self.nodes = {}
                self.edges = []
            
            def add_node(self, name, **attrs):
                self.nodes[name] = attrs
            
            def add_edge(self, u, v):
                self.edges.append((u, v))
        
        return DummyGraph()
