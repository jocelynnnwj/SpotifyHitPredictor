# Contributing to SpotifyHitPredictor

Thank you for your interest in contributing to SpotifyHitPredictor! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### 1. Fork the Repository
- Fork the repository to your GitHub account
- Clone your fork locally

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Your Changes
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 4. Test Your Changes
```bash
# Install dependencies
pip install -r requirements.txt

# Run the analysis
python main.py --quick

# Run tests (if available)
python -m pytest tests/
```

### 5. Commit Your Changes
```bash
git add .
git commit -m "Add: brief description of your changes"
```

### 6. Push and Create a Pull Request
```bash
git push origin feature/your-feature-name
```

## 📋 Coding Standards

### Python Code Style
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise

### Example Function Documentation
```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        param1 (str): Description of parameter 1
        param2 (int): Description of parameter 2
        
    Returns:
        bool: Description of return value
        
    Raises:
        ValueError: When parameters are invalid
    """
    # Function implementation
    pass
```

### File Organization
- Keep related functionality in the same module
- Use descriptive file names
- Organize imports: standard library, third-party, local

## 🧪 Testing

### Writing Tests
- Create tests for new functionality
- Use descriptive test names
- Test both success and failure cases
- Aim for good test coverage

### Example Test
```python
def test_feature_selection():
    """Test feature selection functionality."""
    # Arrange
    X_train = create_test_data()
    y_train = create_test_target()
    
    # Act
    selected_features = select_features(X_train, y_train)
    
    # Assert
    assert len(selected_features) > 0
    assert all(feature in X_train.columns for feature in selected_features)
```

## 📝 Documentation

### Code Documentation
- Add docstrings to all public functions and classes
- Include type hints where appropriate
- Provide examples for complex functions

### README Updates
- Update README.md if you add new features
- Include usage examples
- Update installation instructions if needed

## 🚀 Types of Contributions

### Bug Reports
- Use the issue template
- Provide detailed steps to reproduce
- Include error messages and stack traces
- Specify your environment (OS, Python version, etc.)

### Feature Requests
- Describe the feature clearly
- Explain the use case
- Suggest implementation approach if possible

### Code Improvements
- Performance optimizations
- Code refactoring
- Better error handling
- Additional model algorithms

## 🔍 Review Process

1. **Automated Checks**: All PRs must pass automated tests
2. **Code Review**: At least one maintainer must approve
3. **Documentation**: Ensure documentation is updated
4. **Testing**: Verify that tests pass and new tests are added

## 📞 Getting Help

- Open an issue for bugs or feature requests
- Join our discussions for general questions
- Check existing issues and PRs first

## 🎯 Project Goals

SpotifyHitPredictor aims to:
- Provide accurate song popularity prediction
- Maintain clean, well-documented code
- Support multiple machine learning approaches
- Be accessible to beginners and experts alike

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to SpotifyHitPredictor! 🎵 