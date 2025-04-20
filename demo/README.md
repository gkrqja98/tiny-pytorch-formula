# PyTorch Formula Demo

This folder contains demonstration scripts showing how to use the modular PyTorch Formula library for analyzing deep learning models.

## Available Demos

### 1. Comprehensive Model Analysis

`comprehensive_model_analysis.py` - A complete analysis of PyTorch models including:
- Full forward and backward pass analysis
- Step-by-step computation visualization with mathematical formulas
- Detailed layer analysis with value substitution
- Gradient flow visualization
- Interactive HTML report generation

Usage:
```bash
python comprehensive_model_analysis.py --model tiny_cnn --position 0,0,4,4 --output-dir ./output
```

Arguments:
- `--model`: Model to analyze (tiny_cnn or custom)
- `--position`: Position for detailed analysis (comma-separated)
- `--output-dir`: Directory to save analysis results

### 2. Single Layer Analysis

`single_layer_analysis.py` - Detailed analysis of a specific layer in a model:
- Forward and backward computation
- Mathematical formulas (general form, value substitution, calculation results)
- Tensor visualization
- Step-by-step computation breakdown

Usage:
```bash
python single_layer_analysis.py --layer conv1 --position 0,0,4,4 --output-dir ./output/layer_analysis
```

Arguments:
- `--layer`: Layer name to analyze (conv1, pool1, conv2, etc.)
- `--position`: Position for detailed analysis (comma-separated)
- `--output-dir`: Directory to save analysis results

## Requirements

Make sure you have the following libraries installed:
```bash
pip install torch numpy matplotlib jinja2 networkx
```

## Viewing Results

After running the scripts, open the generated HTML files in a web browser to view the detailed analysis:
- Comprehensive analysis: `output/comprehensive_analysis.html`
- Layer analysis: `output/layer_analysis/{layer_name}_analysis.html`

The HTML reports include interactive elements like collapsible sections and tabs for better navigation.
