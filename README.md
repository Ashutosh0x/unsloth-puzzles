# Unsloth Puzzles

Solutions for the [Unsloth Puzzles Challenge](https://github.com/unslothai/puzzles).

## Project Structure

```
unsloth-puzzles/
├── notebooks/          # Original challenge notebook
├── challenges/         # Solutions by challenge
│   ├── A_nf4_triton/      # NF4 → Triton kernel
│   ├── B_fsdp2_qlora/     # FSDP2 + QLoRA
│   ├── C_torch_compile/   # Graph break elimination
│   ├── D_github_issues/   # Unsloth GitHub contributions
│   └── E_memory_backprop/ # Memory-efficient backprop
├── kaggle/             # Kaggle notebooks for GPU execution
├── tests/              # pytest validation
└── docs/               # PROBLEMS.md, DESIGN.md
```

## Challenges

| Challenge | Points | Status | Environment |
|-----------|--------|--------|-------------|
| A) NF4 → Triton | 14 | 🔲 | Kaggle T4 |
| B) FSDP2 + QLoRA | 10 | 🔲 | Kaggle 2×T4 |
| C) torch.compile | 9 | 🔲 | Kaggle T4 |
| D) GitHub Issues | 12 | 🔲 | Local |
| E) Memory Backprop | 10 | 🔲 | Local + GPU |

## Setup

```bash
# Install Kaggle CLI
pip install kaggle

# Set API token
export KAGGLE_API_TOKEN=your_token_here

# Verify
python -m kaggle competitions list
```

## Execution

### GPU Challenges (A, B, C)
```bash
# Push notebook to Kaggle
python -m kaggle kernels push -p kaggle/A_kernel

# Check output
python -m kaggle kernels output username/kernel-name -p outputs/
```

### Local Challenges (D, E)
```bash
# Run tests
pytest tests/ -v
```
