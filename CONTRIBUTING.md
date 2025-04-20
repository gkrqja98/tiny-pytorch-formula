# Contributing to PyTorch Formula

Thank you for considering contributing to PyTorch Formula! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How Can I Contribute?

### Reporting Bugs

- Check the issue tracker to see if the bug has already been reported
- If not, create a new issue with a clear description and steps to reproduce

### Suggesting Enhancements

- First, check if the enhancement has already been suggested
- If not, create a new issue describing the enhancement

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add or update tests as needed
5. Run tests to ensure they pass
6. Update documentation as needed
7. Commit your changes (`git commit -m 'Add some amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/pytorch-formula.git
cd pytorch-formula

# Install dependencies
pip install -r requirements.txt

# Install for development
pip install -e .
```

## Coding Style

- Follow PEP 8 guidelines
- Use descriptive variable names
- Add docstrings to functions and classes
- Keep code modular and maintainable

## Testing

- Add appropriate tests for your changes
- Ensure all tests pass before submitting a pull request

## Documentation

- Update the README.md if necessary
- Add or update docstrings for new or modified code
- If adding new features, add examples to the examples directory

## Adding New Layer Support

If you're adding support for a new layer type:

1. Add the forward computation function in `detailed_computation_utils.py`
2. Add the backward computation function in `backward_computation_utils.py`
3. Add the formatting functions in `formatters.py` and `backward_formatters.py`
4. Add an example in the examples directory
5. Update the documentation

Thank you for your contribution!
