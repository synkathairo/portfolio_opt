# Repository Guidelines

## Project Structure & Module Organization

Core application code lives in `src/portfolio_opt/` for the custom allocator and `src/cvxportfolio_impl/` for the parallel `cvxportfolio` experiment. Keep business logic separated by concern:

- `cli.py`: command-line entrypoint
- `optimizer.py`: `cvxpy` portfolio optimization
- `alpaca.py`: Alpaca REST integration
- `rebalance.py`: convert target weights into orders
- `model.py`, `config.py`, `types.py`: input loading, settings, shared types

Example inputs live in `examples/`. The notebook `cvxportfolio_exampletutorial.ipynb` is reference material, not production code.

## Build, Test, and Development Commands

Use `uv` for local setup and dependency management:

```bash
uv sync
source .venv/bin/activate
```

When dependencies change, regenerate `uv.lock` with:

```bash
uv lock
```

Run a dry rebalance:

```bash
uv run portfolio-opt --model examples/sample_model.json --dry-run
```

Run the `cvxportfolio` comparison path:

```bash
uv run cvxportfolio-backtest --model examples/sample_universe.json --lookback-days 126 --backtest-days 252
```

Current best practical `cvxportfolio` candidate:

```bash
uv run cvxportfolio-backtest --model examples/sample_universe.json --lookback-days 126 --backtest-days 252 --risk-aversion 0.5 --mean-shrinkage 0.5 --momentum-window 84 --min-cash-weight 0.05 --min-invested-weight 0.4 --linear-trade-cost 0.001
```

Prime and reuse the local Alpaca cache for repeatable backtests:

```bash
uv run cvxportfolio-backtest --model examples/sample_universe.json --lookback-days 126 --backtest-days 252 --use-cache --refresh-cache
uv run cvxportfolio-backtest --model examples/sample_universe.json --lookback-days 126 --backtest-days 252 --offline
```

Quick syntax verification:

```bash
python3 -m compileall src
```

Static type checking:

```bash
uvx ty check
```

**Before committing any code changes, always run `uvx ty check` and verify the plotter works by piping a test backtest into it.**

If tests are added later, prefer `pytest`.
The current lightweight test suite uses `pytest` under `tests/` for deterministic helper behavior and cache setup.

## Coding Style & Naming Conventions

Target Python 3.10+ and keep code straightforward and typed where practical. Follow these conventions:

- 4-space indentation
- `snake_case` for functions, variables, and modules
- `PascalCase` for dataclasses and other types
- small, single-purpose modules

Add comments only where the intent is not obvious from the code. Keep Alpaca and optimization concerns decoupled.

## Testing Guidelines

There is no test suite yet. Add tests under `tests/` with filenames like `test_optimizer.py` and `test_rebalance.py`. Focus first on deterministic logic:

- weight normalization and bound handling
- rebalance threshold behavior
- model file validation

Mock Alpaca API calls instead of hitting live endpoints in unit tests.

## Commit & Pull Request Guidelines

Current history uses short, simple commit subjects such as `single period opt`. Keep commit titles concise and descriptive. Prefer one logical change per commit.

Pull requests should include:

- a short summary of behavior changes
- any new commands or env vars
- verification performed (for example `python3 -m compileall src`)
- sample output or screenshots only if the CLI/user-facing behavior changed materially

## Security & Configuration Tips

Never commit Alpaca credentials. Use environment variables like `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`, and default to paper trading unless a change explicitly requires live trading.
The CLIs default plotting and font caches into `.cache/` so local runs stay inside the repo workspace.
