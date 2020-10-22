import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import sys
from loader import FileLoader
from describe import Describe

def sigmoid(z):
    return 1 / (1 + np.exp(-z))