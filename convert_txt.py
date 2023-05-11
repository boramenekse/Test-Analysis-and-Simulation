import pathlib
import numpy as np

root = pathlib.Path.cwd()
results = root.joinpath('results')
sample = '1DS3'
stuff = np.loadtxt(results.joinpath('PP_1D2D', sample, f'{sample}.txt'))
stuff[:, 0] = stuff[:, 0] - stuff[0, 0]
np.savetxt(results.joinpath('PP_1D2D', sample, f'{sample}_crack_lengths.txt'), stuff)
