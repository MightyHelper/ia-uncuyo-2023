#!/usr/bin/python3
try:
    from lib.random_discrete_agent import RandomAgent
except ModuleNotFoundError as e:
    path = "PYTHONPATH"
    print(f"\x1b[31;1mCould not find python module '{e.name}'. Please add it to env variable {path}\x1b[0m")
    import os
    print(f"Current {path}: {os.getenv(path)}")
    exit(-1)
import pandas as pd
import numpy as np
