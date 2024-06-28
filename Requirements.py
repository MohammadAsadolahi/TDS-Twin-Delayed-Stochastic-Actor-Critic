import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import animation, rc
from IPython.display import Math, HTML
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import copy
import pandas as pd
import json,os
!pip install gymnasium[mujoco]
import gymnasium as gym
