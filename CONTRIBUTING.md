# Contributing to DexDenso Teleop

Thank you for your interest in contributing to **DexDenso Teleop**! We welcome contributions from robotics researchers, computer vision developers, and simulation engineers worldwide.

---

## Code of Conduct

All contributors and participants are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please treat all members of the community with respect and courtesy.

---

## Development Workflow

### 1. Fork & Clone
```bash
git clone https://github.com/MrTonyIT/dexdenso-teleop.git
cd dexdenso-teleop
```

### 2. Environment Setup
Create a dedicated virtual environment with Python 3.10+:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Linux / macOS
source venv/bin/activate

pip install -r requirements.txt -r requirements-dev.txt
```

### 3. Branching Guidelines
Create a descriptive branch for your work:
```bash
git checkout -b feature/analytical-ik-optimization
# or
git checkout -b fix/udp-packet-drop
```

### 4. Code Standards & Style
- **Formatting**: Strictly follow PEP 8.
- **Typing**: Use Python type hints where applicable (`typing.Tuple`, `typing.Optional`, etc.).
- **Comments & Documentation**: All code, docstrings, variable names, and commit messages must be written in **100% technical English**.
- **Performance**: Avoid operations inside the primary vision or networking loops that could introduce frame drops (>16.6ms latency).

### 5. Running Tests
Ensure all unit and integration tests pass before submitting a pull request:
```bash
# Compilation check
python -m py_compile *.py

# Full pytest test runner
python -m pytest -v
```

---

## Commit Guidelines

We enforce **Conventional Commits**:
- `feat: add dual-quaternion orientation smoothing`
- `fix: resolve socket timeout during simulated robot homing`
- `docs: update NVIDIA Isaac Sim 4.0 configuration guide`
- `perf: optimize DirectShow camera frame retrieval`
- `test: add unit coverage for singularity avoidance`

---

## Submitting Pull Requests

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Open a Pull Request against the `main` branch.
3. Describe the problem your change solves and link any relevant issues.
4. Verify that all GitHub Actions CI checks pass.

Thank you for helping push real-time industrial teleoperation forward!
