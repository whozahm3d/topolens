# Contributing to Topolens

Thank you for your interest in contributing to **Topolens**! We welcome contributions from the scientific computing, computer vision, and graph learning communities.

This document outlines the guidelines and standards for contributing to this project. Following these practices ensures smooth collaboration, reproducible science, and high code quality.

---

## 📜 Code of Conduct

By participating in this project, you agree to foster an open, welcoming, and respectful environment. Please report any unacceptable behavior to the project maintainers.

---

## 🛠️ How to Contribute

### 1. Reporting Issues or Bugs
If you encounter a bug, dataset inconsistency, or unexpected behavior:
1. Search existing [Issues](https://github.com/whozahm3d/topolens/issues) to ensure it hasn't already been reported.
2. Open a new issue with a clear title and description. Include:
   - Your operating system and Python/PyTorch versions.
   - Exact steps or commands to reproduce the issue.
   - Full tracebacks or error logs if applicable.

### 2. Suggesting Feature Enhancements
We welcome proposals for new graph rendering backends, baseline architectures, or interpretability tools.
- Open an issue titled `[Feature Request]: <Brief Summary>`.
- Explain the motivation, proposed implementation, and expected impact on evaluation metrics or project goals.

### 3. Submitting Code Changes (Pull Requests)
1. **Fork the Repository**: Create a personal fork on GitHub.
2. **Clone & Set Up Environment**:
   ```bash
   git clone https://github.com/<your-username>/topolens.git
   cd topolens
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Create a Feature Branch**:
   ```bash
   git checkout -b feat/your-feature-name
   ```
4. **Make Your Changes**: Write clean, modular, and documented code.
5. **Verify Evaluation & Code Safety**: Ensure existing evaluation pipelines run without errors.
6. **Commit & Push**: Commit using conventional commit standards (see below) and push to your fork.
7. **Open a Pull Request**: Provide a detailed summary of changes and reference relevant issues.

---

## 📐 Coding Standards & Guidelines

To maintain code readability and prevent subtle PyTorch runtime bugs, please adhere to these coding principles:

### Python & Formatting
- **PEP 8 Compliance**: Follow standard Python code style guidelines.
- **Type Annotations**: Use type hints (`from __future__ import annotations`, `Path | str`, `List[dict]`) for function signatures.
- **Docstrings**: Provide concise docstrings for all functions, modules, and model classes explaining arguments and return types.

### PyTorch & Tensor Safety
- **Cross-Device Execution**: Always explicitly handle GPU/CPU transfers:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  ```
- **Safe Checkpoint Loading**: When loading `.pt` checkpoints, always specify `map_location="cpu"` and set `weights_only=False` if the checkpoint contains custom metadata (such as normalization tuples or config dicts):
  ```python
  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  ```
- **Target Transformations**: Topolens trains on log-transformed targets (`log1p`). Predictions must be decoded safely using `safe_counts()` in `evaluation/evaluate.py` to prevent overflow or NaN propagation.

### Reproducibility & Randomness
- Do not introduce unseeded `random.random()`, `np.random`, or `torch.randint` calls.
- Always initialize seeds using `topolens_utils.set_random_seeds(seed)` or derive deterministic seeds via SHA-256 hashes of input strings (as seen in `data/generate_synthetic.py`).

---

## 📌 Commit Message Conventions

We enforce [Conventional Commits](https://www.conventionalcommits.org/) to keep the git history clean and meaningful. Format commit messages as:

```
<type>(<scope>): <short description>
```

### Supported Types:
- `feat`: A new feature (e.g., adding a new GNN baseline or render mode).
- `fix`: A bug fix (e.g., fixing a checkpoint loader or Streamlit UI warning).
- `docs`: Documentation updates (e.g., updating README or docstrings).
- `refactor`: Code restructuring without changing functionality.
- `chore`: Maintenance tasks, dependency updates, or gitignore tweaks.
- `eval`: Benchmark, evaluation script, or result table additions.

### Examples:
```bash
git commit -m "feat(models): add ResNet18 backbone option for visual regression"
git commit -m "fix(eval): enforce weights_only=False in checkpoint loader"
git commit -m "docs(readme): update experimental benchmark table with held-out metrics"
```

---

## 🧪 Testing Checklist Before Submitting a PR

Before creating your Pull Request, verify the following:

- [ ] `streamlit run app/app.py` launches cleanly without warnings or missing import errors.
- [ ] `python evaluation/evaluate.py` completes successfully on test and held-out splits.
- [ ] No git-ignored model checkpoints (`*.pt`) or large temporary image folders were accidentally committed.
- [ ] All new functions have appropriate type annotations and docstrings.
- [ ] Commit messages follow the Conventional Commits format.

---

## 💬 Need Help?

If you have questions about the codebase structure or research methodology, feel free to open a Q&A issue on GitHub or reach out to the project maintainers. Thank you for contributing to **Topolens**!
