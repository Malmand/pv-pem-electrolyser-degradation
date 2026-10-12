# PV-Powered PEM Electrolyser: Performance and Degradation under Solar Intermittency

Python model of a 1 MW PEM electrolyser coupled directly to PV, simulated hour by hour for a full year in two climates: **Muscat, Oman** (hot, arid) and **Brighton, UK** (temperate, maritime). It quantifies how the variability of solar input affects hydrogen output, efficiency and stack degradation.

This is the code behind the peer-reviewed article:

> Al-Mandhari, M., Ghosh, A. (2025). Modelling solar intermittency effects on PEM electrolyser performance and degradation: a comparison of Oman and UK. *Energies*, 18(23), 6131. https://doi.org/10.3390/en18236131 (**Editor's Choice**)

The work started as my BEng thesis at the University of Exeter.

## What the model does

| Component | Approach |
|---|---|
| PV generation | pvlib ModelChain with PVWatts inverter and loss models; tilt and azimuth optimised for each site |
| Electrochemistry | Reversible voltage with activation, ohmic and concentration overpotentials |
| Thermal behaviour | Lumped thermal capacity model of the stack temperature |
| Degradation | Catalyst (active-area loss), membrane thinning and interfacial resistance growth, using literature-derived rates; voltage degradation reported in µV/h |
| Operation | Minimum-load cut-off (5%), start-stop cycle counting, Faradaic efficiency with gas crossover |
| Validation | Polarisation curves at 60 °C and 80 °C, efficiency and degradation rates checked against published data with MAE, RMSE and R², plus k-fold cross-validation |
| Reference case | Constant irradiance of 350 W/m², isolating the effect of intermittency |

Main settings (in `SimulationConfig`): 1 MW DC nominal power, 20 stacks of 100 cells, 100 cm² active area per cell, operation up to 80 °C.

## How to run

```bash
pip install -r requirements.txt
python pv_pem_simulation.py
# or with your own file locations:
python pv_pem_simulation.py --brighton path/to/brighton.xlsx --muscat path/to/muscat.csv --out results
```

All CSV outputs and figures are written to `results/` by default.

### Input data

The weather files are not included in the repository. Place them in `data/` with these names, or pass their paths as arguments:

| File | Source | Expected format |
|---|---|---|
| `data/brighton_ghi_hourly_2019.xlsx` | Hourly global horizontal irradiance for Brighton, 2019, from the NREL National Solar Radiation Database | Columns `Year`, `Month`, `Day`, `Hour`, `GHI`; air temperature uses monthly means set in the script |
| `data/muscat_tmy_2005_2023.csv` | Typical meteorological year for 23.249° N, 56.448° E from [PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/en/) (2005 to 2023) | Standard PVGIS TMY CSV export |

## Outputs

| File | Content |
|---|---|
| `simulation_summary.csv` | Annual hydrogen, energy, efficiency and degradation per location |
| `brighton_results.csv`, `muscat_results.csv` | Hourly time series of the electrolyser state |
| `scaled_degradation_rates.csv` | Hourly voltage degradation rate (µV/h) |
| `validation_metrics.csv` | Model validation metrics |
| `fig_main_*.png`, `fig_appA_*.png` | Article and appendix figures |

## Repository structure

```
pv-pem-electrolyser-degradation/
├── pv_pem_simulation.py   model, validation and figure generation
├── requirements.txt
├── LICENSE                MIT
└── README.md
```

## Citation

If you use this code, please cite the article above.

## Author

Mohamed Al-Mandhari · [github.com/Malmand](https://github.com/Malmand)
