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

## Project Structure

```
torch_formula/
├── core/             # Core functionality for model analysis
│   └── model_analyzer.py  # Main ModelAnalyzer class
├── models/           # Model definitions for testing and examples
│   └── tiny_cnn.py   # TinyCNN model definition
├── utils/            # Utility functions for tensor operations
│   ├── tensor_formatter.py            # Basic tensor formatting utilities
│   └── detailed_computation/          # Detailed computation utilities
│       ├── detailed_computation_utils.py  # Core computation functions
│       ├── backward_computation_utils.py  # Backward computation functions
│       ├── formatters.py              # Output formatting functions
│       └── backward_formatters.py     # Backward computation formatters
├── viz/              # Visualization tools
│   └── computation_graph.py           # Graph visualization utilities
└── examples/         # Example scripts
    ├── tiny_cnn_analysis.py           # Basic model analysis
    ├── detailed_computation_example.py # Simple computation examples
    ├── detailed_conv_computation.py    # Conv2d computation analysis
    ├── tensor_visualization.py         # Tensor visualization examples
    ├── detailed_value_computation.py   # Value substitution examples
    ├── detailed_backward_computation.py # Backward computation examples
    └── backpropagation_analysis.py     # Full forward/backward analysis
```

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
summaries = analyzer.get_all_layers_summary(format='markdown')
print(summaries)
```

### Forward Pass Detailed Computation

```python
# Get detailed forward computation for a specific layer
detailed_md = analyzer.get_detailed_computation("conv1", format="markdown")

# Or analyze a specific position in a layer
position = (0, 0, 4, 4)  # batch, out_channel, height, width
detailed_md = analyzer.get_detailed_computation("conv1", position=position, format="markdown")
```

### Backward Pass Detailed Computation

```python
# Get detailed backward computation for a specific layer
backward_md = analyzer.get_detailed_backward("conv1", position=position, format="markdown")

# Analyze both forward and backward passes
from torch_formula.utils.detailed_computation import (
    get_computation_markdown, 
    get_backward_markdown
)

# Forward pass
forward_md = get_computation_markdown(conv_layer, input_tensor, output_tensor, position)

# Backward pass
backward_md = get_backward_markdown(conv_layer, input_tensor, output_tensor, grad_output, position)
```

### Gradient Flow Analysis

```python
# Analyze gradient flow through the model
from torch_formula.examples.backpropagation_analysis import visualize_gradient_flow

gradients = visualize_gradient_flow(model, input_tensor, output_dir="./output")
```

## Running the Examples

To run the provided examples:

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

## Example Output: Backward Computation

For Conv2d layer gradient computation, you'll see output like this:

```
### Backward Computation for Conv2d at Position: (batch=0, out_channel=0, y=4, x=4)

#### Gradient Output Value
0.235700

#### Filter Weights
| Index | [0]     | [1]     | [2]     |
| ----- | ------- | ------- | ------- |
| [0]   | 0.3000  | -0.5000 | 0.2000  |
| [1]   | 0.7000  | 0.4000  | -0.1000 |
| [2]   | -0.3000 | 0.8000  | 0.5000  |

#### Gradient Propagation to Input
| Position (y, x, c) | Formula | Gradient Value |
| ----------------- | ------- | -------------- |
| (3, 3, 0) | $(0.2357 \times 0.3000)$ | 0.070710 |
| (3, 4, 0) | $(0.2357 \times -0.5000)$ | -0.117850 |
...

### General Gradient Formula
$\frac{\partial L}{\partial x_{n,c_{in},h_{in},w_{in}}} = \sum_{c_{out}} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} \frac{\partial L}{\partial y_{n,c_{out},h_{out},w_{out}}} \cdot w_{c_{out},c_{in},k_h,k_w}$
```

## Requirements

- PyTorch >= 1.7.0
- NumPy >= 1.19.0
- Pandas >= 1.2.0
- Matplotlib >= 3.3.0
- NetworkX >= 2.5
- tabulate >= 0.8.0 (for Markdown tables)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pytorch-formula.git
cd pytorch-formula

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Future Improvements

Planned improvements for the package include:

1. Interactive visualizations using d3.js or Plotly
2. Support for more complex layer types (e.g., LSTM, Attention)
3. Integration with PyTorch's native visualization tools
4. Improved aesthetics for LaTeX formula rendering
5. Jupyter notebook integration for interactive exploration
6. Acceleration for large-scale models
7. Support for distributed computation analysis

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
