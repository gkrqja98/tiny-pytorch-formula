# PyTorch Formula

A Python package for analyzing, visualizing, and formulating PyTorch model operations. This project helps deep learning researchers and practitioners understand the mathematical formulations and computational flow of neural network operations, including both forward and backward passes.

## Features

- **Model Analysis**: Extract detailed information about model layers, parameters, and operations
- **Mathematical Formulations**: Generate mathematical formulas for forward and backward passes
- **Value Substitution**: Show detailed computation with actual values (e.g., `(1.2 × 0.3) + (0.5 × -0.5) + ...`)
- **Gradient Analysis**: Analyze backward pass computations and gradients
- **Visualization**: Create visual representations of computation graphs and data flow
- **Tensor Formatting**: Format tensor data into readable tables and reports
- **Step-by-Step Computation**: Show detailed steps of tensor operations with value substitution
- **HTML Reports**: Generate comprehensive interactive HTML reports with full analysis

## Project Structure

```
pytorch-formula/
├── core/             # Core functionality for model analysis
│   ├── model_analyzer.py  # Main ModelAnalyzer class
│   └── workflow/     # Analysis workflow utilities
├── models/           # Model definitions for testing and examples
│   └── tiny_cnn.py   # TinyCNN model definition
├── utils/            # Utility functions for tensor operations
│   ├── tensor_formatter/  # Enhanced tensor formatting utilities
│   ├── html_renderer/     # HTML rendering utilities using Jinja2
│   ├── visualization/     # Visualization utilities
│   └── detailed_computation/  # Detailed computation utilities
├── viz/              # Visualization tools
│   └── computation_graph.py  # Graph visualization utilities
├── examples/         # Legacy example scripts
└── demo/             # New modular demonstration scripts
    ├── comprehensive_model_analysis.py  # Complete model analysis
    ├── single_layer_analysis.py         # Detailed layer analysis
    └── README.md                        # Demo documentation
```

## New Demo Scripts

The new `demo/` folder contains modular, well-structured scripts that showcase the library's capabilities in an organized way. These scripts are recommended for new users:

```bash
# Run comprehensive analysis
python demo/comprehensive_model_analysis.py --model tiny_cnn --output-dir ./output

# Run specific layer analysis
python demo/single_layer_analysis.py --layer conv1 --position 0,0,4,4
```

See the `demo/README.md` file for detailed documentation on these new scripts.

## Usage Examples

### Basic Model Analysis

```python
from torch_formula.models import TinyCNN
from torch_formula.core.model_analyzer import ModelAnalyzer

# Create model and sample input
model = TinyCNN()
input_shape = (1, 1, 8, 8)

# Analyze model
analyzer = ModelAnalyzer(model, input_shape)
output, loss = analyzer.run_model()

# Get layer summaries
summaries = analyzer.get_all_layers_summary(format='html')
```

### Using the Analysis Workflow

```python
from torch_formula.models import TinyCNN
from torch_formula.core.workflow import AnalysisWorkflow

# Create model and workflow
model = TinyCNN()
workflow = AnalysisWorkflow(model, (1, 1, 8, 8), output_dir="./output")

# Run full model analysis
result = workflow.run_full_analysis()

# Analyze specific layer
layer_result = workflow.analyze_specific_layer("conv1", position=(0, 0, 4, 4))
```

### Forward and Backward Pass Analysis

```python
# Get detailed forward computation for a specific layer
detailed_html = analyzer.get_detailed_computation("conv1", position=(0, 0, 4, 4), format="html")

# Get detailed backward computation for the same layer
backward_html = analyzer.get_detailed_backward("conv1", position=(0, 0, 4, 4), format="html")
```

### Using the Visualization Tools

```python
from torch_formula.utils.visualization import (
    visualize_tensor_data,
    visualize_gradient_flow,
    create_computation_graph
)

# Create visualizations
visualize_tensor_data(input_tensor, "input_visualization.png")
visualize_gradient_flow(model, input_tensor, "gradient_flow.png")
create_computation_graph(model, input_tensor.shape, "computation_graph.png")
```

## Running the Legacy Examples

The original examples are still available in the `examples/` directory:

```bash
# Basic model analysis
cd torch_formula/examples
python tiny_cnn_analysis.py

# Detailed forward computation with value substitution
python detailed_value_computation.py --mode all

# Detailed backward computation
python detailed_backward_computation.py --mode all

# Complete forward and backward analysis
python backpropagation_analysis.py
```

This will generate analysis reports and visualizations in the `examples/output` directory.

## Example Output: HTML Reports

The new implementation produces comprehensive HTML reports that include:
- Interactive collapsible sections for different aspects of the analysis
- Tabbed interface for navigating between different layers
- Mathematical formulas rendered with MathJax
- Enhanced tensor tables with clear visualization of batches and channels
- Computation graphs and gradient flow visualizations

## Requirements

- PyTorch >= 1.7.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0
- NetworkX >= 2.5 (for graph visualization)
- Jinja2 >= 3.0.0 (for HTML rendering)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pytorch-formula.git
cd pytorch-formula

# Install dependencies
pip install -r requirements.txt

# Install Jinja2 for HTML rendering
pip install jinja2

# Install the package in development mode
pip install -e .
```

## Future Improvements

Planned improvements for the package include:

1. Support for more complex layer types (e.g., LSTM, Attention, Transformers)
2. Integration with PyTorch's native visualization tools
3. Improved aesthetics for LaTeX formula rendering
4. Jupyter notebook integration for interactive exploration
5. Acceleration for large-scale models
6. Support for distributed computation analysis

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
