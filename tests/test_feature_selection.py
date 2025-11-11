import numpy as np
import pandas as pd
import pytest

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ficaria.feature_selection import *