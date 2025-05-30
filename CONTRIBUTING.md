# Contributing

Thank you for your interest in Knightmare! This document captures the
expectations for proposing changes.

## Setup

```bash
git clone https://github.com/CarloTUM/Knightmare.git
cd Knightmare
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
pre-commit install
```

## Workflow

1. Open an issue describing the change you have in mind, especially for
   non-trivial work. This avoids duplicate effort.
2. Create a feature branch off `master` named `feat/<topic>`,
   `fix/<topic>`, `docs/<topic>`, or similar.
3. Make your change. Add or update tests under `tests/`.
4. Run the local checks:

   ```bash
   make lint
   make test
   ```

5. Commit using
   [Conventional Commits](https://www.conventionalcommits.org/) — for
   example `feat(mcts): add transposition table`.
6. Open a pull request. Fill in the PR template; link the issue.

## Coding style

- Code is formatted by `black` and linted by `ruff`. The CI workflow will
  reject style violations; run `make format` before pushing.
- Type-annotate new public functions; `mypy` is part of the linter pass.
- Public APIs live in `selfrl_chess/__init__.py`. Be conservative when
  adding to the public surface.
- Tests are mandatory for new functionality. Aim for branch coverage on
  the search and encoding code-paths (these are easy to get wrong).

## Reporting bugs

Please use the issue template. Include:

- Python and PyTorch versions (`knightmare info` output).
- Minimal repro with random seed if applicable.
- Stack trace or unexpected output.

## Security

Please report security issues privately — see [SECURITY.md](SECURITY.md).
