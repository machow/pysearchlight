import numpy as np
from pysearchlight import run_searchlight
run_searchlight(np.load('test_centers.npy')[()]['centers'][:50], 
                'pattern_indx_diag.npy', 
                '/path/to/data/main_intact/nii', 
                'Slumlord+Overview_intact_mingap1000_medbreak', 
                center_kwargs = np.load('out/searchlight/test_centers.npy')[()]['kwargs'])
