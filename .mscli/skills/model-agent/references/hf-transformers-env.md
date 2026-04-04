# HF Transformers Environment Notes

## MindOne setup
```bash
# Install from source (recommended for development)
cd mindone
pip install -e .

# Install with optional dependencies
pip install -e ".[dev]"  # All dev tools
pip install -e ".[lint,tests]"  # Linting and testing only
pip install -e ".[training]"  # Training utilities
```

## Transformers setup
```bash
cd transformers
pip install -e .
```

## Code quality & formatting
**transformers/** uses a Makefile:
```bash
# Format code (run on feature branches)
make style                    # Format all files with ruff
make modified_only_fixup      # Format only modified files (auto-detects branch)

# Check code quality
make quality                  # Run all linting checks

# Repository consistency checks
make repo-consistency         # Validate copies, dummies, inits, configs, etc.

# Auto-generate code
make autogenerate_code        # Update dependency tables

# Fix everything
make fixup                    # Runs modified_only_fixup + extra_style_checks + autogenerate_code + repo-consistency

# Fix marked code copies
make fix-copies               # Update marked code sections across files
```

**mindone/** uses pyproject.toml configuration:
- Formatting: Black (line length 120)
- Import sorting: isort with MindSpore-specific sections

## Testing

**mindone/**:
```bash
# Run all the transformers model tests with pytest
pytest tests/transformers_tests/models -v

# Run specific test model like `cohere`
pytest tests/transformers_tests/models/cohere/ -v
```
