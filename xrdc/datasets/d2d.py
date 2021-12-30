import pandas as pd
from ..utils.utils import utils



df = pd.read_csv(utils.resource_path("inputs/YijinXRD.dat"), sep = '\t')
qq = df.iloc[:, 0]

patterns = df.iloc[:, 2:]
patterns = (patterns.values.T)[:, 1:]

for i in range(len(patterns)):
    patterns[i] = patterns[i] - i * 1000

m2d = patterns
