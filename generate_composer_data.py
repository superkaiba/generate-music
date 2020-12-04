from Data import Data
import numpy as np
import os
from params import *

for composer in ["beethoven", "burgmueller", "chopin", "haydn", "mendelssohn", "mozart", "schubert", "schumann", "tchaikovsky"]:
    test_data = Data(composer)
    np.save(f"data/{composer}-data", test_data.data)

