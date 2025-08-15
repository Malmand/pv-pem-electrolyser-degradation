# pv-pem-electrolyser-degradation
Python code and datasets for modelling solar intermittency effects on PEM electrolyser performance and degradation in Oman and UK conditions.

# PV–PEM Electrolyser Degradation Simulation

This repository contains the Python implementation used in the study:

> **"Modeling Solar Intermittency Effects on PEM Electrolyser Performance and Degradation: A Comparison of Oman and UK Conditions"**

The model simulates the performance and degradation of a proton exchange membrane (PEM) electrolyser directly coupled to photovoltaic (PV) generation under two distinct climatic conditions: Muscat, Oman (hot-arid) and Brighton, UK (temperate-maritime).

---

## 📄 Overview

The simulation integrates:
- **PV modelling** using location-specific irradiance data.
- **Electrochemical modelling** of PEM electrolysers, including activation, ohmic, and concentration overpotentials.
- **Thermal modelling** via a lumped thermal capacity approach.
- **Degradation modelling** for catalyst, membrane, and interfacial resistance, based on literature-derived empirical rates.

---

## ⚙️ Requirements

- Python **3.10** or later  
- Main packages:  
numpy
pandas
pvlib
matplotlib
seaborn
tqdm
scipy
scikit-learn
numba

csharp
Copy
Edit
Install with:
```bash
pip install -r requirements.txt
