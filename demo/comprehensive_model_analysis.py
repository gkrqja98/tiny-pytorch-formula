"""
Comprehensive PyTorch Model Analysis

This script performs a complete analysis of PyTorch models including:
- Full forward and backward pass analysis
- Step-by-step computation visualization
- Detailed mathematical formulations with LaTeX
- Gradient flow visualization
- Interactive HTML report generation
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modular components
from torch_formula.models import TinyCNN
from torch_formula.core.workflow import AnalysisWorkflow
from torch_formula.utils.visualization import (
    visualize_tensor_data,
    visualize_gradient_flow,
    create_computation_graph
)
from torch_formula.utils.html_renderer.templates import create_comprehensive_report_html

def main():
    """Main function for comprehensive analysis"""
    parser = argparse.ArgumentParser(description="Comprehensive PyTorch Model Analysis")
    parser.add_argument("--model", type=str, default="tiny_cnn", 
                       choices=["tiny_cnn", "custom"],
                       help="Model to analyze")
    parser.add_argument("--position", type=str, default="0,0,4,4",
                       help="Position for detailed analysis for conv1 (comma-separated)")
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse position for conv1 layer
    position_conv1 = tuple(map(int, args.position.split(',')))
    
    # Setup model
    model = setup_model(args.model)
    
    # Setup input
    input_tensor = create_patterned_input(size=8)
    
    # Create analysis workflow
    workflow = AnalysisWorkflow(model, input_tensor.shape, args.output_dir)
    
    # Run comprehensive analysis
    run_comprehensive_analysis(workflow, model, input_tensor, position_conv1, args.output_dir)
    
def run_comprehensive_analysis(workflow, model, input_tensor, position_conv1, output_dir):
    """Run a comprehensive analysis of the model"""
    print("Starting comprehensive model analysis...")
    
    # Step 1: Full model analysis
    print("Analyzing full model architecture...")
    model_result = workflow.run_full_analysis(input_tensor)
    
    # Step 2: Layer-specific detailed analysis
    print("Performing detailed layer analysis...")
    # Define layers to analyze with appropriate positions for each layer
    layers_to_analyze = {
        "conv1": position_conv1,                       # Original position (e.g., 0,0,4,4)
        "pool1": (position_conv1[0], position_conv1[1], position_conv1[2]//2, position_conv1[3]//2),  # Half size (e.g., 0,0,2,2)
        "conv2": (position_conv1[0], position_conv1[1], position_conv1[2]//2, position_conv1[3]//2),  # Same as pool1
        "conv_out": (position_conv1[0], position_conv1[1], position_conv1[2]//4, position_conv1[3]//4)  # Quarter size (e.g., 0,0,1,1)
    }
    
    layer_results = {}
    
    for layer_name, position in layers_to_analyze.items():
        print(f"  Analyzing layer: {layer_name} at position {position}")
        try:
            result = workflow.analyze_specific_layer(layer_name, position)
            layer_results[layer_name] = result
        except Exception as e:
            print(f"    Error analyzing layer {layer_name}: {str(e)}")
            continue
    
    # Step 3: Generate visualizations
    print("Creating visualizations...")
    viz_results = generate_visualizations(model, input_tensor, output_dir)
    
    # Step 4: Create comprehensive HTML report
    print("Compiling comprehensive HTML report...")
    create_comprehensive_report(model_result, layer_results, viz_results, output_dir)
    
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"Open {os.path.join(output_dir, 'comprehensive_analysis.html')} to view the full report")

def generate_visualizations(model, input_tensor, output_dir):
    """Generate various visualizations"""
    results = {}
    
    # Generate tensor visualizations
    print("  Generating tensor visualization...")
    tensor_viz_path = visualize_tensor_data(input_tensor, 
                                         os.path.join(output_dir, "input_tensor_viz"))
    results["tensor_viz"] = tensor_viz_path
    
    # Generate gradient flow visualization
    print("  Generating gradient flow visualization...")
    grad_flow_path = visualize_gradient_flow(model, input_tensor, 
                                          os.path.join(output_dir, "gradient_flow"))
    results["gradient_flow"] = grad_flow_path
    
    # Generate computation graph
    print("  Generating computation graph...")
    comp_graph_path = create_computation_graph(model, input_tensor.shape,
                                            os.path.join(output_dir, "computation_graph"))
    results["computation_graph"] = comp_graph_path
    
    return results

def create_comprehensive_report(model_result, layer_results, viz_results, output_dir):
    """Create a comprehensive HTML report with all analysis results"""
    # Generate the comprehensive report
    html_path = os.path.join(output_dir, "comprehensive_analysis.html")
    create_comprehensive_report_html(
        model_result=model_result,
        layer_results=layer_results,
        visualizations=viz_results,
        output_path=html_path,
        title="Comprehensive PyTorch Model Analysis"
    )
    
    return html_path

def setup_model(model_name):
    """Setup and initialize model with weights"""
    if model_name == "tiny_cnn":
        model = TinyCNN()
        
        # Set specific weights for better visualization
        torch.manual_seed(42)
        with torch.no_grad():
            # First filter
            model.conv1.weight[0, 0, :, :] = torch.tensor([
                [0.3, -0.5, 0.2],
                [0.7, 0.4, -0.1],
                [-0.3, 0.8, 0.5]
            ])
            
            # Second filter
            model.conv1.weight[1, 0, :, :] = torch.tensor([
                [0.2, 0.5, -0.3],
                [-0.4, 0.6, 0.1],
                [0.7, -0.2, 0.4]
            ])
            
            # Bias values
            model.conv1.bias[0] = 0.1
            model.conv1.bias[1] = -0.2
        
    elif model_name == "custom":
        # Create a custom model implementation here
        model = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(2, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16, 10)
        )
    
    return model

def create_patterned_input(size=8):
    """Create an input tensor with a pattern for better visualization"""
    # Create input tensor
    input_tensor = torch.zeros(1, 1, size, size)
    
    # Fill with a pattern
    for i in range(size):
        for j in range(size):
            # Create a pattern: gradual increase with variations
            val = (i + j) / (2*size - 2)  # normalized between 0 and 1
            
            # Add variations
            if (i + j) % 2 == 0:
                val += 0.2
            else:
                val -= 0.1
            
            input_tensor[0, 0, i, j] = val
    
    # Add a cross pattern in the center
    center_start = size//2 - 1
    center_end = size//2 + 1
    input_tensor[0, 0, center_start:center_end, :] += 0.3  # Horizontal line
    input_tensor[0, 0, :, center_start:center_end] += 0.3  # Vertical line
    
    # Ensure values are within a reasonable range
    input_tensor = torch.clamp(input_tensor, -1.0, 1.0)
    
    return input_tensor

if __name__ == "__main__":
    main()
