# Empty on purpose: pytest adds this file's directory (the repo root) to
# sys.path, which is what lets tests/test_ops.py do `from kernels import ...`
# and `from model import ...` without an installed package.
