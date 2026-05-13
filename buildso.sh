#!/bin/bash
nice -n 10 uv run python -m compileall src
nice -n 10 uv run python setup.py build_ext --inplace
