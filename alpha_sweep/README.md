# alpha_plotter

## Requirements
Download wing coords in [XFOIL / UIUC format](https://m-selig.ae.illinois.edu/ads/coord_database.html), e.g. [NACA2408](https://m-selig.ae.illinois.edu/ads/coord/naca2408.dat) to `data` subdirectory, and data csv with columns `alpha, dp0, alpha, dp1, ...` (alpha columns should be the same).

## Running
Script help with: `python alpha_sweep.py -h`

Basic run with:
`python alpha_sweep.py -p data/Data_Nick.csv -w data/naca2408.dat`

