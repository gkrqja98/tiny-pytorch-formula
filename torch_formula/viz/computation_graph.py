"""
Visualization tools for neural network computation graphs
"""
import torch
import torch.nn as nn
import torch.fx as fx
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional, Dict, List, Any

def create_computation_graph(model: nn.Module, input_shape: Tuple[int, ...]) -> nx.DiGraph:
    """
    Create a computation graph from a PyTorch model
    
    Args:
        model: PyTorch model
        input_shape: Shape of the input tensor
        
    Returns:
        NetworkX directed graph representing the computation flow
    """
    # Use torch.fx to trace the model
    try:
        traced_model = fx.symbolic_trace(model)
    except Exception as e:
        print(f"Warning: Could not trace model with fx: {e}")
        return create_manual_graph(model)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for all operations
    for node in traced_model.graph.nodes:
        # Node attributes
        attrs = {
            'type': node.op,
            'target': str(node.target),
            'args': str(node.args),
            'kwargs': str(node.kwargs)
        }
        
        # Add forward node
        G.add_node(node.name, **attrs, direction='forward')
        
        # Add backward node for operations that have gradients
        if node.op != 'placeholder' and node.op != 'output':
            G.add_node(f"{node.name}_backward", **attrs, direction='backward')
    
    # Add edges between nodes
    for node in traced_model.graph.nodes:
        for input_node in node.all_input_nodes:
            # Forward edge
            G.add_edge(input_node.name, node.name, direction='forward')
            
            # Backward edge (in reverse direction)
            if node.op != 'output' and input_node.op != 'placeholder':
                G.add_edge(f"{node.name}_backward", f"{input_node.name}_backward", direction='backward')
    
    return G

def create_manual_graph(model: nn.Module) -> nx.DiGraph:
    """
    Create a computation graph manually when fx tracing fails
    
    Args:
        model: PyTorch model
        
    Returns:
        NetworkX directed graph
    """
    G = nx.DiGraph()
    
    # Add input node
    G.add_node('input', type='placeholder', direction='forward')
    
    # Add nodes and edges for each module
    prev_node = 'input'
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AvgPool2d)) and name:
            # Add forward node
            G.add_node(name, type=type(module).__name__, direction='forward')
            G.add_edge(prev_node, name, direction='forward')
            prev_node = name
            
            # Add backward node
            G.add_node(f"{name}_backward", type=type(module).__name__, direction='backward')
    
    # Add output node
    G.add_node('output', type='output', direction='forward')
    G.add_edge(prev_node, 'output', direction='forward')
    
    # Add backward edges
    nodes = [node for node in G.nodes if G.nodes[node]['direction'] == 'forward' 
             and G.nodes[node]['type'] != 'placeholder' and G.nodes[node]['type'] != 'output']
    
    # Add backward edges in reverse order
    for i in range(len(nodes) - 1, 0, -1):
        if i < len(nodes) and f"{nodes[i]}_backward" in G and f"{nodes[i-1]}_backward" in G:
            G.add_edge(f"{nodes[i]}_backward", f"{nodes[i-1]}_backward", direction='backward')
    
    # Add edge from output to first backward node
    if nodes and f"{nodes[-1]}_backward" in G:
        G.add_edge('output', f"{nodes[-1]}_backward", direction='connection')
    
    return G

def visualize_computation_graph(G: nx.DiGraph, filename: Optional[str] = None) -> plt.Figure:
    """
    Visualize the computation graph
    
    Args:
        G: NetworkX directed graph
        filename: If provided, save the visualization to this file
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(12, 8))
    
    # Separate forward and backward nodes
    forward_nodes = [n for n, d in G.nodes(data=True) if d.get('direction') == 'forward']
    backward_nodes = [n for n, d in G.nodes(data=True) if d.get('direction') == 'backward']
    
    # Calculate positions for nodes
    pos = {}
    
    # Position forward nodes in top half
    if forward_nodes:
        forward_pos = nx.spring_layout(G.subgraph(forward_nodes))
        for node, position in forward_pos.items():
            pos[node] = np.array([position[0], position[1] * 0.4 + 0.5])
    
    # Position backward nodes in bottom half
    if backward_nodes:
        backward_pos = nx.spring_layout(G.subgraph(backward_nodes))
        for node, position in backward_pos.items():
            pos[node] = np.array([position[0], position[1] * 0.4])
    
    # Draw forward nodes
    nx.draw_networkx_nodes(G, pos,
                         nodelist=forward_nodes,
                         node_color='skyblue',
                         node_size=800,
                         alpha=0.8)
    
    # Draw backward nodes
    nx.draw_networkx_nodes(G, pos,
                         nodelist=backward_nodes,
                         node_color='salmon',
                         node_size=800,
                         alpha=0.8)
    
    # Draw edges with different colors based on direction
    forward_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('direction') == 'forward']
    backward_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('direction') == 'backward']
    connection_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('direction') == 'connection']
    
    nx.draw_networkx_edges(G, pos,
                         edgelist=forward_edges,
                         edge_color='blue',
                         arrows=True,
                         width=1.5)
    
    nx.draw_networkx_edges(G, pos,
                         edgelist=backward_edges,
                         edge_color='red',
                         arrows=True,
                         width=1.5)
    
    nx.draw_networkx_edges(G, pos,
                         edgelist=connection_edges,
                         edge_color='purple',
                         style='dashed',
                         arrows=True,
                         width=1.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=10, label='Forward Pass'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', markersize=10, label='Backward Pass'),
        Line2D([0], [0], color='blue', lw=2, label='Forward Flow'),
        Line2D([0], [0], color='red', lw=2, label='Backward Flow'),
        Line2D([0], [0], color='purple', lw=2, linestyle='--', label='Connection')
    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    
    # Set title and layout
    plt.title('Neural Network Computation Graph: Forward and Backward Pass')
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure if filename is provided
    if filename:
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def create_layerwise_graph(model: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[nx.DiGraph, List[str], List[str]]:
    """
    Create a layerwise computation graph
    
    Args:
        model: PyTorch model
        input_shape: Shape of the input tensor
        
    Returns:
        Tuple of (graph, forward_layers, backward_layers)
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add input node
    G.add_node('input', label=f'Input\n{input_shape}', type='data', direction='forward')
    
    # Add nodes for each layer
    prev_layer = 'input'
    forward_layers = []
    backward_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AvgPool2d)) and name:
            # Layer info
            layer_type = type(module).__name__
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            # Add forward node
            G.add_node(name, 
                      label=f'{name}\n{layer_type}\nParams: {params}', 
                      type='layer',
                      direction='forward')
            
            G.add_edge(prev_layer, name, direction='forward')
            prev_layer = name
            forward_layers.append(name)
            
            # Add backward node
            backward_name = f'{name}_backward'
            G.add_node(backward_name,
                      label=f'∇{name}\n∇{layer_type}',
                      type='gradient',
                      direction='backward')
            backward_layers.append(backward_name)
    
    # Add loss node
    G.add_node('loss', label='Loss', type='loss', direction='forward')
    G.add_edge(prev_layer, 'loss', direction='forward')
    forward_layers.append('loss')
    
    # Add backward connections in reverse order
    for i in range(len(backward_layers) - 1):
        G.add_edge(backward_layers[i], backward_layers[i + 1], direction='backward')
    
    # Add input gradient node
    G.add_node('dinput', label='∇Input', type='gradient', direction='backward')
    if backward_layers:
        G.add_edge(backward_layers[-1], 'dinput', direction='backward')
    
    # Connect forward and backward passes
    G.add_edge('loss', backward_layers[0], direction='connection', style='dashed')
    
    return G, ['input'] + forward_layers, backward_layers + ['dinput']

def visualize_layerwise_graph(G: nx.DiGraph, 
                            forward_layers: List[str], 
                            backward_layers: List[str], 
                            filename: Optional[str] = None) -> plt.Figure:
    """
    Visualize the layerwise computation graph
    
    Args:
        G: NetworkX directed graph
        forward_layers: List of forward layer nodes
        backward_layers: List of backward layer nodes
        filename: If provided, save the visualization to this file
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(15, 8))
    
    # Calculate positions
    pos = {}
    
    # x positions based on layer order
    num_layers = max(len(forward_layers), len(backward_layers))
    for i, layer in enumerate(forward_layers):
        pos[layer] = (i / (num_layers - 1), 0.75)
    
    for i, layer in enumerate(backward_layers):
        # Reversed order for backward layers
        pos[layer] = ((num_layers - i - 1) / (num_layers - 1), 0.25)
    
    # Node styles based on type
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        node_type = G.nodes[node].get('type', '')
        
        if node_type == 'data':
            node_colors.append('lightgreen')
            node_sizes.append(1500)
        elif node_type == 'layer':
            node_colors.append('lightblue')
            node_sizes.append(2000)
        elif node_type == 'loss':
            node_colors.append('orange')
            node_sizes.append(1200)
        elif node_type == 'gradient':
            node_colors.append('lightcoral')
            node_sizes.append(1800)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                         node_color=node_colors,
                         node_size=node_sizes,
                         alpha=0.8)
    
    # Draw edges
    forward_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('direction') == 'forward']
    backward_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('direction') == 'backward']
    connection_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('direction') == 'connection']
    
    nx.draw_networkx_edges(G, pos,
                         edgelist=forward_edges,
                         edge_color='blue',
                         arrows=True,
                         width=2.0,
                         arrowsize=20)
    
    nx.draw_networkx_edges(G, pos,
                         edgelist=backward_edges,
                         edge_color='red',
                         arrows=True,
                         width=2.0,
                         arrowsize=20)
    
    nx.draw_networkx_edges(G, pos,
                         edgelist=connection_edges,
                         edge_color='purple',
                         style='dashed',
                         arrows=True,
                         width=1.5,
                         arrowsize=15)
    
    # Draw labels
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Data'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Layer'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Loss'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=10, label='Gradient'),
        Line2D([0], [0], color='blue', lw=2, label='Forward Flow'),
        Line2D([0], [0], color='red', lw=2, label='Backward Flow'),
        Line2D([0], [0], color='purple', lw=2, linestyle='--', label='Connection')
    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    
    # Set title and layout
    plt.title('Neural Network Flow: Forward and Backward Pass', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure if filename is provided
    if filename:
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    
    return plt.gcf()
