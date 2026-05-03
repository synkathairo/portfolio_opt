from __future__ import annotations

from mypyc.build import mypycify
from setuptools import setup


setup(
    ext_modules=mypycify(
        [
            "src/portfolio_opt/black_litterman.py",
            "src/portfolio_opt/backtest.py",
            "src/portfolio_opt/estimation.py",
            "src/portfolio_opt/model.py",
            "src/portfolio_opt/optimizer.py",
            "src/portfolio_opt/rebalance.py",
            "src/portfolio_opt/types.py",
        ],
        opt_level="3",
    ),
)
