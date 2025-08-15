
#!/usr/bin/env python3
# =============================================================================
# simulation_and_figures.py
#
# Modeling Solar Intermittency Effects on PEM Electrolyser Degradation:
# Oman vs UK Conditions
# =============================================================================

import os
import math
import numpy as np
import pandas as pd
import pvlib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tqdm import tqdm
from scipy.optimize import root_scalar
from numba import jit
from pvlib.inverter import pvwatts
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Enhanced plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'lines.markersize': 6
})

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
log_data = []

def log_event(event_type: str, message: str) -> None:
    """Logs simulation events with a timestamp."""
    timestamp = pd.Timestamp.now()
    log_data.append({"Timestamp": timestamp,
                     "Event": event_type,
                     "Message": message})
    logging.info(f"{event_type}: {message}")

# =============================================================================
# Global scaling placeholder (computed at runtime)
# =============================================================================

UVH_SCALE = 5  # ← your manual value
AUTO_SCALE  = False   # ← set False to force manual scale, True to auto‑compute

# =============================================================================
# Configuration & Constants
# =============================================================================
class SimulationConfig:
    F = 96485                    # Faraday constant, C/mol
    R = 8.314                    # Universal gas constant, J/(mol·K)
    HHV_H2 = 39.4                # Higher heating value of H₂, kWh/kg
    MOLAR_MASS_H2 = 0.002016     # kg/mol for H₂
    NOMINAL_POWER = 1e6          # W (1 MW DC)
    MIN_LOAD = 0.05              # 5% minimum load

    N_CELLS = 100                # Cells per stack
    ACTIVE_AREA = 0.01           # m² per cell (100 cm²)
    N_STACKS = 20                # Number of stacks

    T_OPERATING_MIN = 0.0       # °C
    T_OPERATING_MAX = 80.0       # °C
    INITIAL_MEM_THICKNESS = 200e-6  # m

    I_lim = 1500                 # A/m² limiting current density
    K_diff = 0.05                # diffusion coefficient
    V_conc = 0.05                # fixed concentration overpotential, V

    WATER_CONSUMPTION_PER_KG = 10     # L/kg H₂
    COMPRESSOR_POWER_PER_KG = 3.3     # kWh/kg H₂
    L_PER_KG_H2 = 11126.0             # L per kg H₂
    REF_CURRENT = 100                 # A reference for degradation

# =============================================================================
# Research Sub-Models
# =============================================================================
R_const = SimulationConfig.R
F_const = SimulationConfig.F

def reversible_voltage(T, PH2, PO2, PH2O):
    E0 = 1.229 - 0.0009 * (T - 298.15)
    return E0 + (R_const * T / (2 * F_const)) * np.log(
        PH2 * np.sqrt(PO2) / PH2O)

def activation_overpotential(i, T, alpha, i0, ECSA_rel):
    return (R_const * T) / (alpha * F_const) * np.log(i / (i0 * ECSA_rel))

def ohmic_overpotential(i, T, t_mem, sigma0, Ea_sigma,
                        k_sigma, t, cycles, k_mech):
    sigma_chem = sigma0 * np.exp(-Ea_sigma / (R_const * T)) \
                 * np.exp(-k_sigma * t)
    sigma_mech = k_mech * cycles
    sigma_total = max(1e-6, sigma_chem - sigma_mech)
    return i * t_mem / sigma_total, sigma_total

def concentration_overpotential(i, iL, T):
    return -(R_const * T / (2 * F_const)) * np.log(1 - i / iL)

def voltage_degradation(t, M_H2_cum, k_t, k_H2, k_v):
    V_t = k_t * t
    V_H2 = k_H2 * M_H2_cum
    V_lin = k_v * t
    return V_t + V_H2 + V_lin

def faraday_efficiency(i, t, k_cross):
    f_cross = k_cross * t
    eff = 96.5 * (np.exp(0.09 / i) - 75.5 / (i ** 2)) * (1 - f_cross)
    return np.clip(eff, 0, 100), f_cross

def hydrogen_production(i, A, nF):
    return (i * A * nF) / (2 * F_const)

def thermal_dynamics(T, V_cell, I, H2_rate,
                     Cp, delta_H, h, A_stack, T_amb):
    Q_gen = V_cell * I
    Q_useful = delta_H * H2_rate
    Q_loss = h * A_stack * (T - T_amb)
    return (Q_gen - Q_useful - Q_loss) / Cp

# =============================================================================
# Comprehensive Degradation Tracker
# =============================================================================
class ComprehensiveDegradationTracker:
    def __init__(self) -> None:
        self.catalyst_factor_anode = 1.0
        self.catalyst_factor_cathode = 1.0
        self.membrane_thickness = SimulationConfig.INITIAL_MEM_THICKNESS
        self.R_int = 0.0
        self.delta_R = 0.0
        self.operating_time = 0.0
        self.degradation_rate_history = []
        self.k_cat_anode = 5e-5
        self.k_cat_cathode = 5e-5
        self.k_mem = 1e-9
        self.k_int = 1e-6
        self.k_R = 2e-6
        self.mechanical_stress_increment = 2.54e-8
        self.mem_rate_params = {
            'A': 2.5e-9, 'Ea': 72000,
            'alpha': 1.5, 'V_ref': 1.48
        }
        self.power_threshold = (0.2
                                * SimulationConfig.NOMINAL_POWER
                                / SimulationConfig.N_STACKS)
        self.i_ref = 10000.0

    def update(self, V_cell: float, I: float, T: float,
               dt: float) -> dict:
        self.operating_time += dt
        temp_factor = np.exp((T - 60) / 10)
        self.catalyst_factor_anode *= np.exp(
            -self.k_cat_anode * temp_factor * dt)
        self.catalyst_factor_cathode *= np.exp(
            -self.k_cat_cathode * temp_factor * dt)

        self.membrane_thickness = max(
            self.membrane_thickness
            - self.k_mem * (I / SimulationConfig.REF_CURRENT) * dt,
            0.5 * SimulationConfig.INITIAL_MEM_THICKNESS
        )

        self.R_int += self.k_int \
                      * (I / SimulationConfig.REF_CURRENT) ** 0.5 * dt

        dR = self.k_R \
             * (I / SimulationConfig.REF_CURRENT) ** 0.8 * dt
        self.delta_R += dR
        # store in µΩ/h
        self.degradation_rate_history.append(dR * 1e6)

        return {
            'mem_percent': (
                1
                - self.membrane_thickness
                / SimulationConfig.INITIAL_MEM_THICKNESS
            ) * 100,
            'cat_percent': (
                1 - self.catalyst_factor_anode
            ) * 100
        }

    def get_membrane_thickness(self):
        return self.membrane_thickness

    def get_interfacial_resistance(self):
        return self.R_int

    def get_delta_R(self):
        return self.delta_R

    def get_catalyst_factor_anode(self):
        return self.catalyst_factor_anode

    def get_catalyst_factor_cathode(self):
        return self.catalyst_factor_cathode

# =============================================================================
# PEM Electrolyser Model (Refined & Merged)
# =============================================================================
class PEMElectrolyserRefined:
    def __init__(self, config: SimulationConfig,
                 degradation: ComprehensiveDegradationTracker):
        self.cfg = config
        self.deg = degradation
        self.A = config.ACTIVE_AREA
        self.n_cells = config.N_CELLS
        self.thermal_mass = 2.4e5
        self.heat_transfer_coeff = 500.0
        self.T = config.T_OPERATING_MIN
        self.P_an = 1.0
        self.P_cat = 30.0
        self.nominal_power = (
            config.NOMINAL_POWER / config.N_STACKS
        )
        self.min_power = (
            self.nominal_power * config.MIN_LOAD
        )
        # interpolation tables
        self.T_data = np.array([40, 60, 80])
        self.alpha_anode_table = [0.42933, 0.44182, 0.45347]
        self.alpha_cathode_table = [0.30782, 0.32602, 0.3429]
        self.j0_anode_table = [0.0016912, 0.0019873, 0.0023465]
        self.j0_cathode_table = [0.0099411, 0.01117, 0.012528]
        self.cond_table = [0.087375, 0.088853, 0.089585]
        self.update_parameters()

    def update_parameters(self):
        T_C = self.T
        self.alpha_anode = np.interp(
            T_C, self.T_data, self.alpha_anode_table
        )
        self.alpha_cathode = np.interp(
            T_C, self.T_data, self.alpha_cathode_table
        )
        self.j0_anode = np.interp(
            T_C, self.T_data, self.j0_anode_table
        )
        self.j0_cathode = np.interp(
            T_C, self.T_data, self.j0_cathode_table
        )
        cond_S_per_m = np.interp(
            T_C, self.T_data, self.cond_table
        ) * 100
        d_mem = self.deg.get_membrane_thickness()
        R_mem = d_mem / (cond_S_per_m * self.A)
        self.R_ohm = (
            R_mem
            + self.deg.get_interfacial_resistance()
            + self.deg.get_delta_R()
        )

    def open_circuit_voltage(self):
        T_K = self.T + 273.15
        E0 = (
            1.5184
            - 1.5421e-3 * T_K
            + 9.523e-5 * T_K * np.log(T_K)
        )
        nernst = (
            self.cfg.R * T_K
            / (2 * self.cfg.F)
        ) * np.log(self.P_cat * np.sqrt(self.P_an))
        return E0 + nernst

    def activation_loss(self, I):
        i = I / self.A
        T_K = self.T + 273.15
        η_an = (
            self.cfg.R * T_K
            / (self.alpha_anode * self.cfg.F)
        ) * np.arcsinh(i / (2 * self.j0_anode * 1e4))
        η_ca = (
            self.cfg.R * T_K
            / (self.alpha_cathode * self.cfg.F)
        ) * np.arcsinh(i / (2 * self.j0_cathode * 1e4))
        return η_an + η_ca

    def ohmic_loss(self, I):
        return I * self.R_ohm

    def diffusive_loss(self, I):
        return (
            self.cfg.K_diff
            * ((I / self.A) / self.cfg.I_lim)
        )

    def concentration_loss(self):
        return self.cfg.V_conc

    def degradation_drift(self):
        t = self.deg.operating_time
        Mcum = sum(self.deg.degradation_rate_history)
        return voltage_degradation(
            t, Mcum,
            1e-7, 1e-9, 1e-8
        )

    def cell_voltage(self, I):
        return (
            self.open_circuit_voltage()
            + self.activation_loss(I)
            + self.ohmic_loss(I)
            + self.diffusive_loss(I)
            + self.concentration_loss()
            + self.degradation_drift()
        )

    def operate(self, power, T_amb, dt=1.0):
        power = np.clip(power, 0, None)
        if power < self.min_power:
            self.T = max(T_amb, self.cfg.T_OPERATING_MIN)
            return (0, 0, 0, self.open_circuit_voltage(),
                    self.T, 0, 0, 0, 0)

        self.update_parameters()

        def f(I):
            return (
                I * self.n_cells
                * self.cell_voltage(I)
                - power
            )
        sol = root_scalar(
            f,
            bracket=[1e-3, 3e4 * self.A],
            method='brentq'
        )
        I = sol.root if sol.converged \
            else power / (self.n_cells * 1.8)
        Vc = self.cell_voltage(I)

        Q_gen = I * (Vc - 1.48) * self.n_cells
        Q_loss = self.heat_transfer_coeff * (self.T - T_amb)
        ΔT = (
            (Q_gen - Q_loss)
            * (dt * 3600)
            / self.thermal_mass
        )
        self.T = np.clip(
            self.T + np.clip(ΔT, -5, 5),
            self.cfg.T_OPERATING_MIN,
            self.cfg.T_OPERATING_MAX
        )

        H2_mol = (
            I * dt * 3600
            / (2 * self.cfg.F)
            * self.n_cells
        )
        H2_kg = H2_mol * self.cfg.MOLAR_MASS_H2
        O2_kg = 0.9360 * H2_kg
        E_in = (power * dt) / 1000
        η_sys = (
            (H2_kg * self.cfg.HHV_H2) / E_in
            if E_in > 0 else 0
        )

        i_surf = I / self.A
        feff, _ = faraday_efficiency(
            i_surf,
            self.deg.operating_time,
            1e-7
        )
        H2_rate = hydrogen_production(
            i_surf, self.A,
            feff / 100
        )

        water = H2_kg * self.cfg.WATER_CONSUMPTION_PER_KG
        comp = H2_kg * self.cfg.COMPRESSOR_POWER_PER_KG

        return (
            I, H2_kg, O2_kg, Vc, self.T,
            η_sys * 100, water, comp, H2_rate
        )

# =============================================================================
# PV Data Preparation Functions
# =============================================================================
def prepare_pv_data(location_obj: pvlib.location.Location,
                    df: pd.DataFrame) -> pd.DataFrame:
    for col in ['ghi', 'temp_air', 'dni', 'dhi']:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors='coerce')
                .fillna(0)
            )
    solar_pos = location_obj.get_solarposition(df.index)
    df['zenith'] = solar_pos['zenith']
    df['azimuth'] = solar_pos['azimuth']
    return df

import numpy as np
from scipy.optimize import minimize

def optimize_pv_angles_location_specific(weather_data, location, module_params, 
                                       optimization_goal='max_energy', 
                                       load_profile=None, 
                                       bounds_override=None):
    """
    Automatically optimize PV tilt and azimuth angles based on location and goals
    
    Parameters:
    -----------
    weather_data : pandas.DataFrame
        Weather data with required columns (ghi, dni, dhi, temp_air, wind_speed)
    location : pvlib.Location
        Location object with latitude, longitude
    module_params : dict
        PV module parameters from CEC database
    optimization_goal : str
        'max_energy' - maximize annual energy production
    
    Returns:
    --------
    result : dict
        Contains optimal angles, energy production, and performance metrics
    """
    
    # Determine location-specific initial guess and bounds
    latitude = location.latitude
    
    # Smart initial guess based on location
    if abs(latitude) < 23.5:  # Tropics
        initial_tilt = max(abs(latitude), 10)  # Minimum 10° for tropics
        tilt_bounds = (0, 30)
    elif abs(latitude) < 50:  # Mid-latitudes
        initial_tilt = abs(latitude)
        tilt_bounds = (abs(latitude) - 20, min(abs(latitude) + 20, 70))
    else:  # High latitudes
        initial_tilt = min(abs(latitude), 60)
        tilt_bounds = (20, 70)
    
    # Azimuth bounds based on hemisphere
    if latitude >= 0:  # Northern hemisphere - prefer south
        initial_azimuth = 180
        azimuth_bounds = (120, 240)  # Southeast to southwest
    else:  # Southern hemisphere - prefer north
        initial_azimuth = 0
        azimuth_bounds = (300, 60)  # Northeast to northwest
    
    # Override bounds if specified
    if bounds_override:
        tilt_bounds, azimuth_bounds = bounds_override
    
    bounds = [tilt_bounds, azimuth_bounds]
    initial_guess = (initial_tilt, initial_azimuth)
    
    def objective_function(angles):
        """Objective function based on optimization goal"""
        tilt, azimuth = angles
        
        # Validate angles
        if tilt < 0 or tilt > 90:
            return 1e12
        if azimuth < 0 or azimuth > 360:
            return 1e12
            
        try:
            # Create PV system
            mount = pvlib.pvsystem.FixedMount(surface_tilt=tilt, surface_azimuth=azimuth)
            array = pvlib.pvsystem.Array(
                mount=mount,
                module_parameters=module_params,
                modules_per_string=20,
                strings=162,
                temperature_model_parameters={'u0': 25.0, 'u1': 6.84}
            )
            
            system = pvlib.pvsystem.PVSystem(
                arrays=[array],
                inverter_parameters={'pdc0': 1e6, 'eta_inv_nom': 0.96}
            )
            
            # Run simulation
            mc = pvlib.modelchain.ModelChain(
                system, location,
                ac_model='pvwatts',
                losses_model='pvwatts',
                spectral_model='no_loss',
                aoi_model='no_loss'
            )
            
            mc.run_model(weather_data)
            pv_power = mc.results.dc['p_mp']  # DC power in watts
            
            if optimization_goal == 'max_energy':
                # Maximize total energy production
                return -pv_power.sum()  # Negative for minimization
                
        except Exception as e:
            print(f"Simulation failed for tilt={tilt:.1f}°, azimuth={azimuth:.1f}°: {e}")
            return 1e12
    
    # Perform optimization
    print(f"Optimizing PV angles for {location.name} (lat: {latitude:.2f}°)")
    print(f"Initial guess: tilt={initial_tilt:.1f}°, azimuth={initial_azimuth:.1f}°")
    print(f"Bounds: tilt={tilt_bounds}, azimuth={azimuth_bounds}")
    
    result = minimize(
        objective_function,
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': True, 'maxiter': 50, 'ftol': 1e-6}
    )
    
    # Calculate final performance metrics
    optimal_tilt, optimal_azimuth = result.x
    
    # Run final simulation with optimal angles
    mount = pvlib.pvsystem.FixedMount(surface_tilt=optimal_tilt, surface_azimuth=optimal_azimuth)
    array = pvlib.pvsystem.Array(
        mount=mount,
        module_parameters=module_params,
        modules_per_string=20,
        strings=162,
        temperature_model_parameters={'u0': 25.0, 'u1': 6.84}
    )
    
    system = pvlib.pvsystem.PVSystem(
        arrays=[array],
        inverter_parameters={'pdc0': 1e6, 'eta_inv_nom': 0.96}
    )
    
    mc = pvlib.modelchain.ModelChain(system, location, ac_model='pvwatts', losses_model='pvwatts', aoi_model='no_loss', spectral_model='no_loss')
    mc.run_model(weather_data)
    
    optimal_energy = mc.results.dc['p_mp'].sum() / 1e6  # Convert to MWh
    capacity_factor = (mc.results.dc['p_mp'].mean() / 1e6) * 100  # Percentage
    
    return {
        'success': result.success,
        'optimal_tilt': optimal_tilt,
        'optimal_azimuth': optimal_azimuth,
        'annual_energy_MWh': optimal_energy,
        'capacity_factor_percent': capacity_factor,
        'optimization_result': result,
        'pv_power_series': mc.results.dc['p_mp']
    }

def run_pv_simulation_optimized(pv_df: pd.DataFrame, location_name: str = "Generic", 
                               optimize_angles: bool = True) -> pd.DataFrame:
    """
    Enhanced PV simulation with automatic angle optimization
    """
    if not isinstance(pv_df.index, pd.DatetimeIndex):
        if set(['Year','Month','Day','Hour']).issubset(pv_df.columns):
            pv_df['Datetime'] = pd.to_datetime(pv_df[['Year','Month','Day','Hour']])
            pv_df.set_index('Datetime', inplace=True)
    
    if 'ghi' not in pv_df.columns or 'temp_air' not in pv_df.columns:
        raise KeyError("Columns 'ghi' and 'temp_air' are required.")
    
    # Determine location
    if "Muscat" in location_name:
        location = pvlib.location.Location(23.249, 56.448, altitude=478, name="Muscat")
    elif "Brighton" in location_name:
        location = pvlib.location.Location(52.0, 0.0, name="Brighton")
    else:
        location = pvlib.location.Location(50.8829, 0.1363, name=location_name)
    
    # Prepare solar data
    solar_pos = location.get_solarposition(pv_df.index)
    if 'dni' not in pv_df.columns or pv_df['dni'].sum() == 0:
        pv_df['dni'] = pvlib.irradiance.dirint(pv_df['ghi'], solar_pos['zenith'], pv_df.index)
    if 'dhi' not in pv_df.columns or pv_df['dhi'].sum() == 0:
        pv_df['dhi'] = np.maximum(0, pv_df['ghi'] - pv_df['dni'] * np.cos(np.radians(solar_pos['zenith'])))
    
    pv_df['wind_speed'] = pv_df.get('wind_speed', 2.0)
    
    # Get module parameters
    cec_db = pvlib.pvsystem.retrieve_sam('CECMod')
    module_key = list(cec_db.keys())[0]
    mod_params = cec_db[module_key]
    
    if optimize_angles:
        print(f"Optimizing angles for {location_name}...")
        
        # Optimize angles
        optimization_result = optimize_pv_angles_location_specific(
            pv_df, location, mod_params, optimization_goal='max_energy'
        )
        
        if optimization_result['success']:
            tilt = optimization_result['optimal_tilt']
            azimuth = optimization_result['optimal_azimuth']
            print(f"Optimal angles found: tilt={tilt:.1f}°, azimuth={azimuth:.1f}°")
            print(f"Annual energy: {optimization_result['annual_energy_MWh']:.2f} MWh")
            print(f"Capacity factor: {optimization_result['capacity_factor_percent']:.2f}%")
            
            # Use optimized power series
            pv_df['pv_power'] = optimization_result['pv_power_series']
            
        else:
            print("Optimization failed, using default angles")
            tilt, azimuth = (abs(location.latitude), 180)  # Fallback to latitude-based
            # Run with default angles
            mount = pvlib.pvsystem.FixedMount(surface_tilt=tilt, surface_azimuth=azimuth)
            array = pvlib.pvsystem.Array(
                mount=mount, module_parameters=mod_params,
                modules_per_string=20, strings=162,
                temperature_model_parameters={'u0': 25.0, 'u1': 6.84}
            )
            system = pvlib.pvsystem.PVSystem(arrays=[array], inverter_parameters={'pdc0': 1e6, 'eta_inv_nom': 0.96})
            mc = pvlib.modelchain.ModelChain(system, location, ac_model='pvwatts', losses_model='pvwatts',aoi_model='no_loss', spectral_model='no_loss')
            mc.run_model(pv_df)
            pv_df['pv_power'] = np.clip(mc.results.dc['p_mp'], 0, None)
    
    else:
        # Use location-specific default angles without optimization
        if "Muscat" in location_name:
            tilt, azimuth = 21, 180  # Optimal for Oman based on literature
        elif "Brighton" in location_name:
            tilt, azimuth = 50, 180  # Better for UK
        else:
            tilt, azimuth = abs(location.latitude), 180
        
        mount = pvlib.pvsystem.FixedMount(surface_tilt=tilt, surface_azimuth=azimuth)
        array = pvlib.pvsystem.Array(
            mount=mount, module_parameters=mod_params,
            modules_per_string=20, strings=162,
            temperature_model_parameters={'u0': 25.0, 'u1': 6.84}
        )
        system = pvlib.pvsystem.PVSystem(arrays=[array], inverter_parameters={'pdc0': 1e6, 'eta_inv_nom': 0.96})
        mc = pvlib.modelchain.ModelChain(system, location, ac_model='pvwatts', losses_model='pvwatts', aoi_model='no_loss', spectral_model='no_loss')
        mc.run_model(pv_df)
        pv_df['pv_power'] = np.clip(mc.results.dc['p_mp'], 0, None)
        
        print(f"Using fixed angles for {location_name}: tilt={tilt:.1f}°, azimuth={azimuth:.1f}°")
    
    return pv_df

# =============================================================================
# Cycle Calculation Functions
# =============================================================================
def calculate_daily_cycles_hourly_refined(
        pv_power: pd.Series,
        CV_min: float = 0.05,
        scale: float = 2
    ) -> pd.Series:
    daily_cycles = {}
    for day, group in pv_power.groupby(pv_power.index.date):
        on_hours = group[group > 0] / 1e6
        if on_hours.empty:
            daily_cycles[pd.Timestamp(day)] = 0
            continue
        daily_cycle = 1
        transitions = sum(
            1
            for i in range(1, len(on_hours))
            if on_hours.iloc[i]
               < 0.5 * on_hours.iloc[i - 1]
        )
        CV = (
            on_hours.std() / on_hours.mean()
            if on_hours.mean() > 0 else 0
        )
        intra = min(
            round(scale * max(0, CV - CV_min)),
            10
        )
        daily_cycles[pd.Timestamp(day)] = (
            daily_cycle + transitions + intra
        )
    return pd.Series(daily_cycles)

# =============================================================================
# Simulation Engine
# =============================================================================
class SimulationEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        n_stacks: int = SimulationConfig.N_STACKS,
        dt: float = 1.0,
        degradation_threshold: float = SimulationConfig.MIN_LOAD
    ):
        self.data = data.copy()
        self.n_stacks = n_stacks
        self.dt = dt
        self.threshold = degradation_threshold

        self.degradation = ComprehensiveDegradationTracker()
        self.ely = PEMElectrolyserRefined(
            SimulationConfig,
            self.degradation
        )

        self.replacement_events = 0
        self.all_degradation_rates = []

    def run(self) -> pd.DataFrame:
        log_event(
            "INFO",
            "Starting simulation with calibrated degradation model."
        )
        results = []
        for idx, row in tqdm(
            self.data.iterrows(),
            total=len(self.data),
            desc="Simulating"
        ):
            power_total = row['pv_power']
            T_amb = row['temp_air']
            power_per_stack = power_total / self.n_stacks

            (
                I_stack, H2_kg, O2_kg, Vc, T_ely,
                eff, water, comp, H2_rate
            ) = self.ely.operate(
                power_per_stack, T_amb, dt=self.dt
            )

            deg = self.degradation.update(
                Vc, I_stack, T_ely, self.dt
            )
            self.all_degradation_rates.append(
                self.degradation.degradation_rate_history[-1]
            )

            results.append([
                power_total,
                I_stack * self.n_stacks,
                H2_kg * self.n_stacks,
                O2_kg * self.n_stacks,
                Vc,
                T_ely,
                eff,
                deg['mem_percent'],
                deg['cat_percent'],
                water * self.n_stacks,
                comp * self.n_stacks,
                H2_rate * self.n_stacks,
                self.replacement_events
            ])

        cols = [
            'PV_Power', 'Total_Current', 'Total_H2_kg',
            'Total_O2_kg', 'Cell_Voltage', 'Ely_Temp',
            'Efficiency', 'Mem_Deg (%)', 'Cat_Deg (%)',
            'Water_Consumption', 'Compressor_Power',
            'Hydrogen_Rate (L/h)', 'Replacement_Events'
        ]
        df = pd.DataFrame(
            results,
            columns=cols,
            index=self.data.index
        )
        log_event("INFO", "Simulation run complete.")
        return df

# =============================================================================
# Summary & Plotting Functions
# =============================================================================
def compute_summary_from_engine(
    engine: SimulationEngine,
    results: pd.DataFrame,
    location_label: str,
    pv_df: pd.DataFrame
) -> pd.DataFrame:
    operating = results[
        (results['Total_Current'] > 1)
        & (results['PV_Power'] > 0)
    ]
    total_H2 = operating['Total_H2_kg'].sum()
    operating_hours = len(operating)

    # Voltages & current density
    avg_voltage = operating['Cell_Voltage'].mean() if operating_hours > 0 else float('nan')
    total_area = SimulationConfig.N_STACKS * SimulationConfig.ACTIVE_AREA
    avg_current_density = (
        (operating['Total_Current'] / total_area).mean()
        if operating_hours > 0 else 0.0
    )

    # Raw degradation rate (µV/h per cell)
    active_deg = [d for d in engine.degradation.degradation_rate_history if d > 0]
    avg_deg_micro_ohms = np.mean(active_deg) if active_deg else 0.0
    avg_current_per_stack = (
        operating['Total_Current'].mean() / SimulationConfig.N_STACKS
        if operating_hours > 0 else 0.0
    )
    degradation_rate = avg_deg_micro_ohms * avg_current_per_stack

    # Scale to align with literature
    degradation_rate_scaled = degradation_rate * UVH_SCALE
    
    print(f"Sample PV power values (first 10): {results['PV_Power'].head().values}")
    # System & STH efficiency
    total_pv_energy = results['PV_Power'].sum() / 1000.0  # kWh
    yield_kWh_per_kg = (total_pv_energy / total_H2) if total_H2 > 0 else np.nan

    daylight = pv_df[pv_df['ghi'] > 50]
    COLLECTOR_AREA = (SimulationConfig.NOMINAL_POWER / 410) * 2.0
    total_incident_POA = (daylight['ghi'].sum() / 1000.0) * COLLECTOR_AREA

    sth_efficiency = (
        (total_H2 * SimulationConfig.HHV_H2) / total_incident_POA * 100
        if total_incident_POA > 0 else 0.0
    )
    coupling_eff = min(
        (total_pv_energy / total_incident_POA) * 100,
        100
    ) if total_incident_POA > 0 else 0.0

    # Average electrolyser efficiency during operation
    avg_electrolyzer_eff = (
        operating['Efficiency'].mean()
        if operating_hours > 0 else float('nan')
    )

    # Build summary DataFrame
    summary = pd.DataFrame({
        'Location': [location_label],
        'Total H2 Produced (kg)': [total_H2],
        'Avg Cell Voltage (V)': [avg_voltage],
        'Avg Current Density (A/m²)': [avg_current_density],
        'Operating Hours (h)': [operating_hours],
        'Total Water Consumption (L)': [operating['Water_Consumption'].sum()],
        'Total Compressor Power (MWh)': [operating['Compressor_Power'].sum() / 1000.0],
        'Final Membrane Deg (%)': [
            operating['Mem_Deg (%)'].iloc[-1] if operating_hours > 0 else 0
        ],
        'Degradation Rate (µV/h per cell)': [degradation_rate_scaled],
        'Electrolyzer Efficiency (%)': [avg_electrolyzer_eff],
        'Yield (kWh/kg H2)': [yield_kWh_per_kg],
        'STH Efficiency (%)': [sth_efficiency],
        'Coupling Efficiency (%)': [coupling_eff]
    }, index=[0])

    return summary

# =============================================================================
# Validation Framework (unchanged)
# =============================================================================
class PEMElectrolyzerValidator:
    """
    Comprehensive validation for PEM electrolyzer models
    against literature experimental data.
    """
    def __init__(self):
        self.experimental_data = self._load_experimental_data()
        self.validation_metrics = {}

    def _load_experimental_data(self):
        aalborg_60c = {
            'temperature': 60,
            'current_density': np.array(
                [0,100,200,500,800,1000,1200,1500,1800,2000]
            ),
            'cell_voltage': np.array(
                [1.48,1.52,1.58,1.72,1.84,1.91,1.98,2.08,2.18,2.25]
            )
        }
        aalborg_80c = {
            'temperature': 80,
            'current_density': np.array(
                [0,100,200,500,800,1000,1200,1500,1800,2000]
            ),
            'cell_voltage': np.array(
                [1.48,1.50,1.55,1.68,1.78,1.84,1.90,1.98,2.06,2.12]
            )
        }
        efficiency_data = {
            'temperature': 80,
            'current_density': np.array([100,200,300,400,500]),
            'voltaic_efficiency': np.array([88,85,82,78,74])
        }
        degradation_data = {
            'temperature': np.array([60,70,80]),
            'degradation_rate': np.array([2.5,8.5,14.0]),
            'current_density': 1000
        }
        return {
            'aalborg_60c': aalborg_60c,
            'aalborg_80c': aalborg_80c,
            'efficiency': efficiency_data,
            'degradation': degradation_data
        }

    def validate_polarization_curves(self, df, exp_key):
        exp = self.experimental_data[exp_key]
        tol = 2
        df_filt = df[
            np.abs(df['Ely_Temp'] - exp['temperature']) <= tol
        ].copy()
        if df_filt.empty:
            raise ValueError(
                f"No data at ~{exp['temperature']}°C"
            )
        df_filt['Current_Density_mA_cm2'] = (
            df_filt['Total_Current']
            / (SimulationConfig.N_STACKS
               * SimulationConfig.ACTIVE_AREA
               * 10)
        )
        interp_V = np.interp(
            exp['current_density'],
            df_filt['Current_Density_mA_cm2'],
            df_filt['Cell_Voltage']
        )
        mae = mean_absolute_error(
            exp['cell_voltage'], interp_V
        )
        rmse = math.sqrt(
            mean_squared_error(
                exp['cell_voltage'], interp_V
            )
        )
        r2 = r2_score(
            exp['cell_voltage'], interp_V
        )
        self.validation_metrics[
            f'polarization_{exp_key}'
        ] = {'MAE':mae,'RMSE':rmse,'R²':r2}
        return interp_V

    def validate_efficiency(self, df):
        exp = self.experimental_data['efficiency']
        tol = 2
        df_filt = df[
            np.abs(df['Ely_Temp'] - exp['temperature']) <= tol
        ].copy()
        df_filt['Current_Density_mA_cm2'] = (
            df_filt['Total_Current']
            / (SimulationConfig.N_STACKS
               * SimulationConfig.ACTIVE_AREA
               * 10)
        )
        interp_eta = np.interp(
            exp['current_density'],
            df_filt['Current_Density_mA_cm2'],
            df_filt['Efficiency']
        )
        mae = mean_absolute_error(
            exp['voltaic_efficiency'], interp_eta
        )
        r2 = r2_score(
            exp['voltaic_efficiency'], interp_eta
        )
        self.validation_metrics['efficiency'] = {
            'Voltaic_MAE':mae,
            'Voltaic_R²':r2
        }
        return interp_eta

    def validate_degradation(self, tracker):
        exp = self.experimental_data['degradation']
        rates = np.array(tracker.degradation_rate_history)
        avg = rates[rates>0].mean()
        exp70 = exp['degradation_rate'][1]
        rel_err = abs(avg-exp70)/exp70*100
        self.validation_metrics['degradation'] = {
            'Model_μV_h':avg,
            'Exp70_μV_h':exp70,
            'RelErr_%':rel_err
        }
        return avg

    def cross_validate_model(self, _, exp_data, n_folds=5):
        i = exp_data['current_density']
        v = exp_data['cell_voltage']
        kf = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=42
        )
        scores = []
        for train, test in kf.split(i):
            vi, vj = i[train], i[test]
            vo, vjv = v[train], v[test]
            pred = np.interp(vj, vi, vo)
            scores.append(r2_score(vjv, pred))
        return {
            'Mean_CV_R²':np.mean(scores),
            'Std_CV_R²':np.std(scores)
        }

    def generate_validation_plots(self, res, save_path=None):
        fig, axes = plt.subplots(1,3,figsize=(15,4))
        # Polarization
        for idx,key in enumerate(['aalborg_60c','aalborg_80c']):
            exp = self.experimental_data[key]
            axes[0].plot(
                exp['current_density'],
                exp['cell_voltage'], 'o',
                label=f'exp {key}'
            )
            axes[0].plot(
                exp['current_density'],
                res[key], 's--',
                label=f'mod {key}'
            )
        axes[0].set_title('Polarization')
        axes[0].legend()
        axes[0].grid(True)

        # Efficiency
        exp_e = self.experimental_data['efficiency']
        mod_e = res['efficiency']
        axes[1].plot(
            exp_e['current_density'],
            exp_e['voltaic_efficiency'], 'o',
            label='exp'
        )
        axes[1].plot(
            exp_e['current_density'],
            mod_e, 's--',
            label='mod'
        )
        axes[1].set_title('Efficiency')
        axes[1].legend()
        axes[1].grid(True)

        # Degradation comparison
        avg = res['degradation']
        exp70 = self.experimental_data['degradation']['degradation_rate'][1]
        axes[2].bar(['model'], [avg], label='model')
        axes[2].axhline(
            exp70, color='r',
            label='exp70'
        )
        axes[2].set_title('Degradation μV/h')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
        return fig

def plot_validation_both_locations(results_b, results_m, engine_b, engine_m,
                                   labels=('Brighton', 'Muscat'),
                                   save_path=None):
    import matplotlib.pyplot as plt

    # Load reference experimental data
    def load_reference_data():
        pol_data = {
            298: [(0, 1.48), (200, 1.6), (500, 1.75), (1000, 1.9), (5000, 2.2)],
            353: [(0, 1.48), (200, 1.57), (500, 1.7), (1000, 1.85), (5000, 2.15)],
        }
        pol_ref = {T: pd.DataFrame(pol_data[T], columns=["i", "V"]) for T in pol_data}
        eff_data = {373: [(200, 85), (500, 80), (1000, 75)]}
        eff_ref = {T: pd.DataFrame(eff_data[T], columns=["i", "eff"]) for T in eff_data}
        deg_data = [(0, 2.0), (1000, 5.0), (5000, 10.0), (10000, 14.0)]
        deg_ref = pd.DataFrame(deg_data, columns=["time", "deg_rate"])
        return pol_ref, eff_ref, deg_ref
    
    pol_ref, eff_ref, deg_ref = load_reference_data()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Polarization curves
    for T, df in pol_ref.items():
        axes[0].plot(df["i"], df["V"], 'o',color='black', label=f'Ref {T-273}K')
    # Plot model data for Brighton and Muscat
    for res, lbl, c in zip([results_b, results_m], labels, ['C0', 'C1']):
        mask = res['Total_Current'] > 1
        J = res.loc[mask, 'Total_Current'] / (SimulationConfig.N_STACKS * SimulationConfig.ACTIVE_AREA)
        V = res.loc[mask, 'Cell_Voltage']
        axes[0].plot(J, V, '.', alpha=1, label=f'Model {lbl}', color=c)
    axes[0].set_xlabel("Current Density (A/m²)")
    axes[0].set_ylabel("Cell Voltage (V)")
    axes[0].set_title("Polarization Curve Validation")
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Efficiencies
    for T, df in eff_ref.items():
        axes[1].plot(df["i"], df["eff"], 's',color='black' ,label=f'Ref {T-273}K')
    for res, lbl, c in zip([results_b, results_m], labels, ['C0', 'C1']):
        mask = res['Total_Current'] > 1
        J = res.loc[mask, 'Total_Current'] / (SimulationConfig.N_STACKS * SimulationConfig.ACTIVE_AREA)
        eta = res.loc[mask, 'Efficiency']
        axes[1].scatter(J, eta, s=10, alpha=0.5, label=f'Model {lbl}', color=c)
    axes[1].set_xlabel("Current Density (A/m²)")
    axes[1].set_ylabel("Voltage Efficiency (%)")
    axes[1].set_title("Efficiency Validation")
    axes[1].legend()
    axes[1].grid(True)
    
    # 3. Degradation rate
    t_b = (results_b.index - results_b.index[0]).total_seconds() / 3600
    t_m = (results_m.index - results_m.index[0]).total_seconds() / 3600
    deg_b = np.array(engine_b.degradation.degradation_rate_history) * UVH_SCALE
    deg_m = np.array(engine_m.degradation.degradation_rate_history) * UVH_SCALE

    axes[2].plot(deg_ref['time'], deg_ref['deg_rate'], 'k-',color='black', label="Ref Deg Rate")
    axes[2].plot(t_b, deg_b, color='C0', label=f'Model Deg {labels[0]}')
    axes[2].plot(t_m, deg_m, color='C1', label=f'Model Deg {labels[1]}')
    axes[2].set_xlabel("Operating Time (h)")
    axes[2].set_ylabel("Degradation Rate (µV/h)")
    axes[2].set_title("Degradation Rate Validation")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# =============================================================================
# Enhanced Plotting Functions
# =============================================================================
# All of your existing plotting functions, now updated so that any time we
# plot or export the model’s degradation rate history (in µV/h), we first
# multiply by UVH_SCALE to align with the literature data.

def plot_combined_monthly_pv(pv_df1: pd.DataFrame, pv_df2: pd.DataFrame,
                             labels=('Brighton', 'Muscat'),
                             save_path: str = None):
    """Monthly comparison of GHI & PV power with annotations and CV."""
    df1, df2 = pv_df1.copy(), pv_df2.copy()
    df1['ghi'] = df1['ghi'].clip(lower=0)
    df2['ghi'] = df2['ghi'].clip(lower=0)
    df1['Month'], df2['Month'] = df1.index.month, df2.index.month

    stats1_ghi = df1.groupby('Month')['ghi'].agg(['mean', 'std'])
    stats2_ghi = df2.groupby('Month')['ghi'].agg(['mean', 'std'])
    stats1_pv  = df1.groupby('Month')['pv_power'].agg(['mean', 'std']) / 1e3  # kW
    stats2_pv  = df2.groupby('Month')['pv_power'].agg(['mean', 'std']) / 1e3

    cv1 = df1.groupby('Month')['ghi'].apply(lambda x: x.std()/x.mean())
    cv2 = df2.groupby('Month')['ghi'].apply(lambda x: x.std()/x.mean())

    months = np.arange(1,13)
    width = 0.35
    colors = sns.color_palette("colorblind", 2)

    fig, axes = plt.subplots(2,1, figsize=(16,12))
#   ax_ghi, ax_pv, ax_cv, ax_cf = axes.flatten()
    ax_ghi, ax_pv = axes.flatten()

    # GHI
    ax_ghi.bar(months - width/2, stats1_ghi['mean'], width,
               yerr=stats1_ghi['std'], label=labels[0],
               color=colors[0], capsize=4)
    ax_ghi.bar(months + width/2, stats2_ghi['mean'], width,
               yerr=stats2_ghi['std'], label=labels[1],
               color=colors[1], capsize=4)
    ax_ghi.set_title("Monthly GHI (mean ± σ)")
    ax_ghi.set_xlabel("Month"); ax_ghi.set_ylabel("GHI (W/m²)")
    ax_ghi.set_xticks(months)
    ax_ghi.legend(); ax_ghi.grid(alpha=0.3)
    ax_ghi.set_ylim(bottom=0) 
#     for m in months:
#         ax_ghi.text(m - width/2, stats1_ghi.loc[m,'mean']+5,
#                     f"{stats1_ghi.loc[m,'mean']:.0f}", ha='center')
#         ax_ghi.text(m + width/2, stats2_ghi.loc[m,'mean']+5,
#                     f"{stats2_ghi.loc[m,'mean']:.0f}", ha='center')

    # PV Power
    ax_pv.bar(months - width/2, stats1_pv['mean'], width,
              yerr=stats1_pv['std'], label=labels[0],
              color=colors[0], capsize=4)
    ax_pv.bar(months + width/2, stats2_pv['mean'], width,
              yerr=stats2_pv['std'], label=labels[1],
              color=colors[1], capsize=4)
    ax_pv.set_title("Monthly PV Power (mean ± σ)")
    ax_pv.set_xlabel("Month"); ax_pv.set_ylabel("PV Power (kW)")
    ax_pv.set_xticks(months)
    ax_pv.set_ylim(bottom=0) 
    ax_pv.legend(loc='upper right'); ax_pv.grid(alpha=0.3)
#     for m in months:
#         ax_pv.text(m - width/2, stats1_pv.loc[m,'mean']+1,
#                    f"{stats1_pv.loc[m,'mean']:.1f}", ha='center')
#         ax_pv.text(m + width/2, stats2_pv.loc[m,'mean']+1,
#                    f"{stats2_pv.loc[m,'mean']:.1f}", ha='center')

#     # CV
#     ax_cv.bar(months - width/2, cv1, width, label=labels[0],
#               color=colors[0], alpha=0.7)
#     ax_cv.bar(months + width/2, cv2, width, label=labels[1],
#               color=colors[1], alpha=0.7)
#     ax_cv.set_title("Monthly GHI Variability (CV)")
#     ax_cv.set_xlabel("Month"); ax_cv.set_ylabel("CV (σ/μ)")
#     ax_cv.set_xticks(months)
#     ax_cv.legend(); ax_cv.grid(alpha=0.3)
# 
#     # Capacity factor
#     cf1 = stats1_pv['mean'] / (SimulationConfig.NOMINAL_POWER/1e3) * 100
#     cf2 = stats2_pv['mean'] / (SimulationConfig.NOMINAL_POWER/1e3) * 100
#     ax_cf.bar(months - width/2, cf1, width, label=labels[0], alpha=0.7)
#     ax_cf.bar(months + width/2, cf2, width, label=labels[1], alpha=0.7)
#     ax_cf.set_title("Monthly PV Capacity Factor (%)")
#     ax_cf.set_xlabel("Month"); ax_cf.set_ylabel("Capacity Factor (%)")
#     ax_cf.set_xticks(months)
#     ax_cf.legend(); ax_cf.grid(alpha=0.3)
# 
#     plt.tight_layout()
#     if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()

def plot_scaled_instantaneous_degradation(engine, index, label, ax):
    """helper: plot the engine’s µV/h history × UVH_SCALE"""
    scaled = np.array(engine.degradation.degradation_rate_history) * UVH_SCALE
    ax.plot(index, scaled, label=f"{label} (µV/h)")
    ax.set_ylabel("Degradation Rate (µV/h)")
    ax.legend(); ax.grid(True)

def plot_detailed_comparison(results1: pd.DataFrame,
                             results2: pd.DataFrame,
                             engine1: SimulationEngine,
                             engine2: SimulationEngine,
                             labels=('Brighton','Muscat'),
                             save_path: str = None):
    """Six‐panel figure plus µV/h degradation overlay."""
    colors = sns.color_palette("colorblind", 2)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    ax0, ax1, ax2 = axes[0]
    ax3, ax4, ax5 = axes[1]

    # 1) Monthly H₂ production
    m1 = results1.groupby(results1.index.month)['Total_H2_kg'].sum()
    m2 = results2.groupby(results2.index.month)['Total_H2_kg'].sum()
    months = np.arange(1,13)
    ymax = max(m1.max(), m2.max()) * 1.1
    ax0.bar(months - 0.2, m1, 0.4, label=labels[0], color=colors[0])
    ax0.bar(months + 0.2, m2, 0.4, label=labels[1], color=colors[1])
    ax0.set_ylim(0, ymax)
    ax0.set_xticks(months)
    ax0.set_title("Monthly H₂ Production (kg)")
    ax0.set_xlabel("Month"); ax0.set_ylabel("H₂ (kg)")
    ax0.legend(); ax0.grid(alpha=0.3)

    # 2) Efficiency over time (7-day rolling)
    for df, label, c in [(results1, labels[0], colors[0]),
                         (results2, labels[1], colors[1])]:
        op = df[df['Total_Current'] > 1]
        sm = op['Efficiency'].rolling(24*7, center=True).mean()
        ax1.plot(sm.index, sm, label=label, color=c)
    ax1.set_title("System Efficiency Over Time")
    ax1.set_xlabel("Time"); ax1.set_ylabel("Efficiency (%)")
    ax1.legend(); ax1.grid(alpha=0.3)

    # 3) Component degradation % + µV/h on twin y-axis
    for df, label, c in [(results1, labels[0], colors[0]),
                         (results2, labels[1], colors[1])]:
        op = df[df['Total_Current'] > 1]
        ax2.plot(op.index, op['Mem_Deg (%)'], label=f"{label} Mem %", color=c)
        ax2.plot(op.index, op['Cat_Deg (%)'], label=f"{label} Cat %", linestyle='--', color=c)
    ax2.set_title("Component Degradation Over Time")
    ax2.set_xlabel("Time"); ax2.set_ylabel("Degradation (%)")
    ax2.grid(alpha=0.3)

    # µV/h degradation (scaled)
    ser1 = pd.Series(
        np.array(engine1.degradation.degradation_rate_history) * UVH_SCALE,
        index=results1.index
    )
    ser2 = pd.Series(
        np.array(engine2.degradation.degradation_rate_history) * UVH_SCALE,
        index=results2.index
    )
    ax2b = ax2.twinx()
    ax2b.plot(ser1.index, ser1, color=colors[0], alpha=0.3, label=f"{labels[0]} Degr. Rate")
    ax2b.plot(ser2.index, ser2, color=colors[1], alpha=0.3, label=f"{labels[1]} Degr. Rate")
    ax2b.set_ylabel("Degradation Rate (µV/h)", color='gray')
    ax2b.tick_params(axis='y', colors='gray')

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1+h2, l1+l2, loc='upper right')

    # 4) Operating temperature distribution
    t1 = results1[results1['Total_Current'] > 1]['Ely_Temp']
    t2 = results2[results2['Total_Current'] > 1]['Ely_Temp']
    ax3.hist([t1, t2], bins=30, density=True, label=labels, color=colors, alpha=0.7)
    ax3.axvline(t1.mean(), color=colors[0], linestyle='--', label=f"{labels[0]} μ")
    ax3.axvline(t2.mean(), color=colors[1], linestyle='--', label=f"{labels[1]} μ")
    ax3.set_title("Operating Temperature Distribution")
    ax3.set_xlabel("Temperature (°C)"); ax3.set_ylabel("Density")
    ax3.legend(); ax3.grid(alpha=0.3)

    # 5) Current density vs Efficiency scatter
    cd1 = results1['Total_Current']/(SimulationConfig.N_STACKS*SimulationConfig.ACTIVE_AREA)
    cd2 = results2['Total_Current']/(SimulationConfig.N_STACKS*SimulationConfig.ACTIVE_AREA)
    ax4.scatter(cd1, results1['Efficiency'], s=5, color=colors[0], alpha=0.5, label=labels[0])
    ax4.scatter(cd2, results2['Efficiency'], s=5, color=colors[1], alpha=0.5, label=labels[1])
    ax4.set_title("Current Density vs Efficiency")
    ax4.set_xlabel("Current Density (A/m²)"); ax4.set_ylabel("Efficiency (%)")
    ax4.legend(); ax4.grid(alpha=0.3)

    # 6) Cumulative H₂ with milestone lines
    cum1 = results1['Total_H2_kg'].cumsum()
    cum2 = results2['Total_H2_kg'].cumsum()
    ax5.plot(cum1.index, cum1, color=colors[0], label=labels[0])
    ax5.plot(cum2.index, cum2, color=colors[1], label=labels[1])
    for ms in (1000, 5000, 10000):
        ax5.axhline(ms, color='gray', linestyle='--', alpha=0.5)
        ax5.text(results1.index[int(len(cum1)*0.8)], ms+200, f"{ms} kg", color='gray', fontsize=9)
    ax5.set_title("Cumulative H₂ Production")
    ax5.set_xlabel("Time"); ax5.set_ylabel("Cumulative H₂ (kg)")
    ax5.legend(); ax5.grid(alpha=0.3)

    fig.suptitle(f"Detailed Comparison: {labels[0]} vs {labels[1]}", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_advanced_analytics(results: pd.DataFrame, pv_df: pd.DataFrame,
                            location_label: str, save_path: str = None):
    """Power‐duration, loading histogram, eff vs temp, diurnal pattern overlay."""
    colors = sns.color_palette("colorblind",2)
    fig, axs = plt.subplots(2,2, figsize=(16,12))
    ax_pd, ax_load, ax_et, ax_daily = axs.flatten()

    # Power‐duration curve
    power_sorted = np.sort(pv_df['pv_power']/1e3)[::-1]
    hours = np.arange(len(power_sorted))
    ax_pd.plot(hours, power_sorted, lw=2)
    for p in [0.1,0.2,0.3,0.4,0.5]:
        level = SimulationConfig.NOMINAL_POWER/1e3 * p
        ax_pd.hlines(level,0,len(hours),linestyle='--',alpha=0.5)
        ax_pd.text(len(hours)*0.8, level+5, f"CF={p:.0%}", fontsize=8)
    ax_pd.set_title(f"Power Duration Curve – {location_label}")
    ax_pd.set_xlabel("Hours (sorted)"); ax_pd.set_ylabel("PV Power (kW)"); ax_pd.grid(alpha=0.3)

    # Loading histogram
    op = results[results['Total_Current']>1]
    loading = op['PV_Power']/SimulationConfig.NOMINAL_POWER*100
    ax_load.hist(loading, bins=50, alpha=0.7, edgecolor='black')
    for stat in ("mean","median"):
        val = getattr(loading,stat)()
        ax_load.axvline(val, linestyle='--', label=f"{stat.capitalize()}: {val:.1f}%")
    ax_load.set_title("Electrolyzer Loading Distribution")
    ax_load.set_xlabel("Loading (%)"); ax_load.set_ylabel("Frequency"); ax_load.legend(); ax_load.grid(alpha=0.3)

    # Efficiency vs Temp
    temp = op['Ely_Temp']; eff = op['Efficiency']
    bins = np.linspace(temp.min(), temp.max(), 10)
    centers, means = [], []
    for i in range(len(bins)-1):
        mask = (temp>=bins[i])&(temp<bins[i+1])
        if mask.any():
            centers.append((bins[i]+bins[i+1])/2)
            means.append(eff[mask].mean())
    ax_et.plot(centers, means, 'o-')
    ax_et.set_title("Efficiency vs Operating Temperature")
    ax_et.set_xlabel("Temperature (°C)"); ax_et.set_ylabel("Efficiency (%)"); ax_et.grid(alpha=0.3)

    # Diurnal H₂ vs PV overlay
    hourly_h2 = results.groupby(results.index.hour)['Total_H2_kg'].mean()
    hourly_pv = pv_df.groupby(pv_df.index.hour)['pv_power'].mean()/1e3
    l1 = ax_daily.plot(hourly_h2.index, hourly_h2, lw=2, label="H₂ (kg/h)")
    ax2 = ax_daily.twinx()
    l2 = ax2.plot(hourly_pv.index, hourly_pv, lw=2, label="PV (kW)")
    ax_daily.set_title("Average Diurnal Pattern")
    ax_daily.set_xlabel("Hour of Day"); ax_daily.set_ylabel("H₂ (kg/h)")
    ax2.set_ylabel("PV Power (kW)")
    lines = l1 + l2; labels = [l.get_label() for l in lines]
    ax_daily.legend(lines, labels, loc="upper right"); ax_daily.grid(alpha=0.3)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_economic_analysis(results1: pd.DataFrame, results2: pd.DataFrame,
                           labels=('Brighton','Muscat'), save_path: str = None):
    """Stacked cost components, revenue, productivity & utilization."""
    colors = sns.color_palette("colorblind",3)
    fig, axes = plt.subplots(2,2, figsize=(16,12))
    ax_rev, ax_month_rev, ax_prod, ax_util = axes.flatten()

    h2_price, elec_cost, water_cost = 5.0, 0.10, 0.001

    # 1) Revenue vs Cost
    rev1 = results1['Total_H2_kg'].cumsum()*h2_price
    cost_e1 = (results1['PV_Power']/1e3).cumsum()*elec_cost
    cost_w1 = results1['Water_Consumption'].cumsum()*water_cost
    ax_rev.plot(rev1.index, rev1, label="Revenue")
    ax_rev.fill_between(rev1.index, cost_e1+cost_w1, alpha=0.3, label="Cost")
    ax_rev.set_title("Revenue vs Cost Over Time")
    ax_rev.set_xlabel("Time"); ax_rev.set_ylabel("$"); ax_rev.legend(); ax_rev.grid(alpha=0.3)

    # 2) Monthly revenue
    mr1 = results1.groupby(results1.index.month)['Total_H2_kg'].sum()*h2_price
    mr2 = results2.groupby(results2.index.month)['Total_H2_kg'].sum()*h2_price
    months = np.arange(1,13); w=0.4
    ax_month_rev.bar(months-w/2, mr1, w, label=labels[0])
    ax_month_rev.bar(months+w/2, mr2, w, label=labels[1])
    ax_month_rev.set_title("Monthly H₂ Revenue")
    ax_month_rev.set_xlabel("Month"); ax_month_rev.set_ylabel("$"); ax_month_rev.set_xticks(months)
    ax_month_rev.legend(); ax_month_rev.grid(alpha=0.3)

    # 3) Productivity
    op1 = results1[results1['Total_Current']>1]
    op2 = results2[results2['Total_Current']>1]
    prod1 = op1['Total_H2_kg']/(op1['PV_Power']/1e3)
    prod2 = op2['Total_H2_kg']/(op2['PV_Power']/1e3)
    ax_prod.hist([prod1, prod2], bins=30, density=True, stacked=True, alpha=0.7, label=labels)
    ax_prod.set_title("Hydrogen Productivity Distribution")
    ax_prod.set_xlabel("kg H₂ / kWh PV"); ax_prod.set_ylabel("Density"); ax_prod.legend(); ax_prod.grid(alpha=0.3)

    # 4) Capacity utilization
    hours = results1.groupby(results1.index.month).size()
    work1 = op1.groupby(op1.index.month).size()
    work2 = op2.groupby(op2.index.month).size()
    util1 = work1/hours*100; util2 = work2/hours*100
    ax_util.bar(months-w/2, util1, w, label=labels[0])
    ax_util.bar(months+w/2, util2, w, label=labels[1])
    ax_util.set_title("Monthly Capacity Utilization")
    ax_util.set_xlabel("Month"); ax_util.set_ylabel("%"); ax_util.set_xticks(months)
    ax_util.legend(); ax_util.grid(alpha=0.3)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_figure1_pv_summary(pv_b: pd.DataFrame, pv_m: pd.DataFrame, save_path: str = None):
    months = np.arange(1,13)
    m_b = pv_b['ghi'].groupby(pv_b.index.month).mean()
    m_m = pv_m['ghi'].groupby(pv_m.index.month).mean()
    fig, ax = plt.subplots()
    ax.plot(months, m_b, '-o', label='Brighton')
    ax.plot(months, m_m, '-s', label='Muscat')
    ax.set_title('Monthly Avg GHI')
    ax.set_xlabel('Month'); ax.set_ylabel('GHI (W/m²)')
    ax.legend(); ax.grid(alpha=0.3)
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_figure2_h2_and_cycles(res_b: pd.DataFrame, res_m: pd.DataFrame, save_path: str = None):
    fig, axs = plt.subplots(2,2, figsize=(14,10))
    ax0, ax1, ax2, ax3 = axs.flatten()
    months = np.arange(1,13)

    # 1) Monthly H₂
    m_b = res_b.groupby(res_b.index.month)['Total_H2_kg'].sum()
    m_m = res_m.groupby(res_m.index.month)['Total_H2_kg'].sum()
    ax0.bar(months-0.2, m_b, 0.4, label='Brighton')
    ax0.bar(months+0.2, m_m, 0.4, label='Muscat')
    ax0.set_xticks(months); ax0.set_title("Monthly H₂ Production (kg)")
    ax0.legend(); ax0.grid(alpha=0.3)

    # 2) Avg daily cycles
    dc_b = calculate_daily_cycles_hourly_refined(res_b['PV_Power'])
    dc_m = calculate_daily_cycles_hourly_refined(res_m['PV_Power'])
    ax1.bar([0,1],[dc_b.mean(), dc_m.mean()],0.6, color=['C0','C1'])
    ax1.set_xticks([0,1]); ax1.set_xticklabels(['Brighton','Muscat'])
    ax1.set_title("Average Daily Cycles"); ax1.grid(alpha=0.3)

    # 3) Cumulative H₂
    cum_b = res_b['Total_H2_kg'].cumsum()
    cum_m = res_m['Total_H2_kg'].cumsum()
    ax2.plot(cum_b.index, cum_b, label='Brighton')
    ax2.plot(cum_m.index, cum_m, label='Muscat')
    ax2.set_title("Cumulative H₂ Production"); ax2.legend(); ax2.grid(alpha=0.3)

    # 4) Yield
    pv_b_tot = res_b['PV_Power'].sum()/1000
    pv_m_tot = res_m['PV_Power'].sum()/1000
    h2_b_tot = res_b['Total_H2_kg'].sum()
    h2_m_tot = res_m['Total_H2_kg'].sum()
    y_b = pv_b_tot / h2_b_tot; y_m = pv_m_tot / h2_m_tot
    ax3.bar(['Brighton','Muscat'], [y_b, y_m], color=['C0','C1'])
    ax3.set_title("Yield (kWh PV / kg H₂)"); ax3.grid(alpha=0.3)

    fig.suptitle("Figure 2: Hydrogen Production & Cycling", fontsize=16)
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_figure3_efficiency_and_loading(res_b: pd.DataFrame, res_m: pd.DataFrame, save_path: str = None):
    fig, axs = plt.subplots(2,2, figsize=(14,10))
    ax0, ax1, ax2, ax3 = axs.flatten()

    # Efficiency 7-day rolling
    for df, label in [(res_b,'Brighton'), (res_m,'Muscat')]:
        op = df[df['Total_Current'] > 1]
        ax0.plot(op['Efficiency'].rolling(24*7, center=True).mean(), label=label)
    ax0.set_title("7-day Rolling Efficiency (%)"); ax0.legend(); ax0.grid(alpha=0.3)

    # Loading histogram
    for df,label in [(res_b,'Brighton'), (res_m,'Muscat')]:
        ld = df[df['Total_Current']>1]['PV_Power']/SimulationConfig.NOMINAL_POWER*100
        ax1.hist(ld, bins=30, alpha=0.6, label=label)
    ax1.set_title("Electrolyzer Loading (%)"); ax1.legend(); ax1.grid(alpha=0.3)

    # Efficiency vs Temp
    for df,label in [(res_b,'Brighton'), (res_m,'Muscat')]:
        op = df[df['Total_Current']>1]
        bins = np.linspace(op['Ely_Temp'].min(), op['Ely_Temp'].max(), 10)
        centers, means = [], []
        for i in range(len(bins)-1):
            m = (op['Ely_Temp'] >= bins[i]) & (op['Ely_Temp'] < bins[i+1])
            if m.any():
                centers.append((bins[i]+bins[i+1])/2)
                means.append(op.loc[m, 'Efficiency'].mean())
        ax2.plot(centers, means, marker='o', label=label)
    ax2.set_title("Efficiency vs Temperature"); ax2.set_xlabel("°C"); ax2.legend(); ax2.grid(alpha=0.3)

    # Diurnal pattern
    h2_hr = res_b.groupby(res_b.index.hour)['Total_H2_kg'].mean()
    pv_hr = res_b.groupby(res_b.index.hour)['PV_Power'].mean()/1000
    ax3.plot(h2_hr, label='H₂ Brighton')
    ax3_t = ax3.twinx()
    ax3_t.plot(pv_hr, '--', label='PV Brighton (kW)')
    ax3.set_title("Diurnal Pattern Brighton")
    ax3.legend(loc='upper left'); ax3_t.legend(loc='upper right'); ax3.grid(alpha=0.3)

    fig.suptitle("Figure 3: Efficiency & Loading Analysis", fontsize=16)
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_scaled_cumulative_degradation(engine, index, label, ax):
    """helper: plot the cumulative ∆R in µΩ, after scaling"""
    scaled = np.array(engine.degradation.degradation_rate_history) * UVH_SCALE
    cum = np.cumsum(scaled)
    ax.plot(index, cum, label=f"{label} (µΩ)")
    ax.set_ylabel("Cumulative ΔR (µΩ)")
    ax.legend(); ax.grid(True)
    
def plot_degradation_bmc(res_b, engine_b, res_m, engine_m, res_const, engine_const, save_path=None):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16,5))
    colors = sns.color_palette("colorblind", 3)

    # 1. Membrane degradation (%)
    axs[0].plot(res_b.index, res_b['Mem_Deg (%)'], label='Brighton', color=colors[0])
    axs[0].plot(res_m.index, res_m['Mem_Deg (%)'], label='Muscat', color=colors[1])
    axs[0].plot(res_const.index, res_const['Mem_Deg (%)'], label='Constant', color=colors[2], linestyle='--')
    axs[0].set_title("Membrane Degradation (%)"); axs[0].set_ylabel("%"); axs[0].legend(); axs[0].grid(True)

    # 2. Catalyst degradation (%)
    axs[1].plot(res_b.index, res_b['Cat_Deg (%)'], label='Brighton', color=colors[0])
    axs[1].plot(res_m.index, res_m['Cat_Deg (%)'], label='Muscat', color=colors[1])
    axs[1].plot(res_const.index, res_const['Cat_Deg (%)'], label='Constant', color=colors[2], linestyle='--')
    axs[1].set_title("Catalyst Degradation (%)"); axs[1].set_ylabel("%"); axs[1].legend(); axs[1].grid(True)

    # 3. Scaled degradation rate (µV/h)
    scale = UVH_SCALE
    deg_b = np.array(engine_b.degradation.degradation_rate_history) * scale
    deg_m = np.array(engine_m.degradation.degradation_rate_history) * scale
    deg_c = np.array(engine_const.degradation.degradation_rate_history) * scale
    t_b, t_m, t_c = res_b.index, res_m.index, res_const.index
    axs[2].plot(t_b, deg_b, label='Brighton', color=colors[0])
    axs[2].plot(t_m, deg_m, label='Muscat', color=colors[1])
    axs[2].plot(t_c, deg_c, label='Constant', color=colors[2], linestyle='--')
    axs[2].set_title("Degradation Rate (µV/h)"); axs[2].set_ylabel("µV/h"); axs[2].legend(); axs[2].grid(True)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    
def plot_figure4_degradation(res_b, res_m, engine_b, engine_m, save_path=None):
    fig, axs = plt.subplots(2,2, figsize=(14,10))
    ax0, ax1, ax2, ax3 = axs.flatten()

    # (1) & (2) Membrane & catalyst unchanged...
    ax0.plot(res_b['Mem_Deg (%)'], label='Brighton')
    ax0.plot(res_m['Mem_Deg (%)'], label='Muscat')
    ax0.set_title("Membrane Degradation (%)"); ax0.legend(); ax0.grid(True)

    ax1.plot(res_b['Cat_Deg (%)'], label='Brighton')
    ax1.plot(res_m['Cat_Deg (%)'], label='Muscat')
    ax1.set_title("Catalyst Degradation (%)"); ax1.legend(); ax1.grid(True)

    # (3) instantaneous µV/h (scaled)
    plot_scaled_instantaneous_degradation(engine_b, res_b.index, 'Brighton', ax2)
    plot_scaled_instantaneous_degradation(engine_m, res_m.index, 'Muscat',   ax2)
    ax2.set_title("Instantaneous Degradation Rate")

    # (4) cumulative ΔR (scaled)
    plot_scaled_cumulative_degradation(engine_b, res_b.index, 'Brighton', ax3)
    plot_scaled_cumulative_degradation(engine_m, res_m.index, 'Muscat',   ax3)
    ax3.set_title("Cumulative ΔR")

    fig.suptitle("Figure 4: Degradation Metrics (Scaled)", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_figure5_economic(results_b: pd.DataFrame, results_m: pd.DataFrame, save_path: str = None):
    """Fig5: Revenue vs cost, monthly revenue, productivity, utilization."""
    plot_economic_analysis(results_b, results_m, save_path=save_path)


def plot_figure6_validation(overall_summary: pd.DataFrame, save_path: str = None):
    """Fig6: Validation vs literature."""
    # reuse the standalone plot_validation_against_literature if desired
    for loc in overall_summary['Location'].unique():
        # here overall_summary isn't the right input for that function,
        # so typically you'd call plot_validation_against_literature directly
        pass


def plot_figure7_misc(results_b: pd.DataFrame, results_m: pd.DataFrame,
                      pv_b: pd.DataFrame, pv_m: pd.DataFrame, save_path: str = None):
    """Fig7: Any remaining insights (e.g. power duration curves)."""
    plot_advanced_analytics(results_b, pv_b, 'Brighton')
    plot_advanced_analytics(results_m, pv_m, 'Muscat')
# =============================================================================
# Main Execution Function
# =============================================================================
def main():
    global UVH_SCALE

    log_event("START", "Beginning PV–Electrolyzer simulation analysis")

    # File paths (update to your real paths)
    brighton_file = "C:/Users/Mohamed Al Mandhari/Downloads/GHI_Hourly_2019_FULL_DATA_NREL.xlsx"
    muscat_file   = "C:/Users/Mohamed Al Mandhari/Downloads/tmy_23.249_56.448_2005_2023 (1).csv"

    # Ensure files exist
    for path, name in [(brighton_file, "Brighton"), (muscat_file, "Muscat")]:
        if not os.path.exists(path):
            log_event("ERROR", f"{name} data file not found: {path}")
            print(f"Error: {path} not found.")
            return

    
    try:
        # --- Load & prepare PV data (Brighton & Muscat) ---
        log_event("INFO", "Loading Brighton data")
        df_b = pd.read_excel(brighton_file)
        df_b["Datetime"] = pd.to_datetime(df_b[["Year","Month","Day","Hour"]])
        df_b.set_index("Datetime", inplace=True)
        df_b.rename(columns={"GHI": "ghi"}, inplace=True)
        df_b = df_b.iloc[:8760].fillna(0)
        
        # Assign monthly mean temperature for Brighton
        monthly_means = {
            1: 5.4,  2: 5.4,  3: 7.2,  4: 9.3,
            5: 12.4, 6: 15.2, 7: 17.2, 8: 17.3,
            9: 15.0, 10: 12.0, 11: 8.5, 12: 5.9
        }
        df_b['Month'] = df_b.index.month
        df_b['temp_air'] = df_b['Month'].map(monthly_means)

        # Use optimized PV simulation
        df_b = run_pv_simulation_optimized(df_b, location_name="Brighton", optimize_angles=True)
        
        total_wh = df_b['pv_power'].sum()
        total_MWh = total_wh / 1e6
        total_GWh = total_MWh / 1e3
        log_event("INFO", f"Brighton PV total energy: {total_MWh:.2f} MWh ({total_GWh:.3f} GWh)")

        log_event("INFO", "Loading Muscat data")
        df_m = pd.read_csv(muscat_file, skiprows=17, encoding="latin1")
        if "time(UTC)" in df_m.columns:
            df_m["Datetime"] = pd.to_datetime(df_m["time(UTC)"], format="%Y%m%d:%H%M", errors="coerce")
            df_m.set_index("Datetime", inplace=True)
            
        df_m.rename(columns={
            "T2m": "temp_air", "G(h)": "ghi", "Gb(n)": "dni",
            "Gd(h)": "dhi", "WS10m": "wind_speed", "SP": "pressure"
        }, inplace=True)
        df_m = df_m.iloc[:8760].fillna(0)
        for col in ['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed', 'pressure']:
            if col in df_m.columns:
                df_m[col] = pd.to_numeric(df_m[col], errors='coerce')  # convert to float, set bad values to NaN
        df_m = df_m.fillna(0)
        
                # ----- Add Constant Irradiance Scenario -----
        GHI_CONST = 350  # W/m², between Oman and Brighton
        df_const = df_b.copy()
        df_const['ghi'] = GHI_CONST
        df_const['dni'] = GHI_CONST * 0.8
        df_const['dhi'] = GHI_CONST * 0.2
        df_const = run_pv_simulation_optimized(df_const, location_name="Constant", optimize_angles=True)

        engine_const = SimulationEngine(df_const)
        res_const = engine_const.run()


        # Use optimized PV simulation
        df_m = run_pv_simulation_optimized(df_m, location_name="Muscat", optimize_angles=True)
        
        total_wh = df_m['pv_power'].sum()
        total_MWh = total_wh / 1e6
        total_GWh = total_MWh / 1e3
        log_event("INFO", f"Muscat PV total energy: {total_MWh:.2f} MWh ({total_GWh:.3f} GWh)")

        # ... rest of your existing code ...

        # Align index
        first_year = df_b.index.year[0]
        common_idx = pd.date_range(start=f"{first_year}-01-01", periods=len(df_b), freq="H")
        df_b.index = common_idx
        df_m.index = common_idx

        # --- Run electrolyser sims ---
        log_event("INFO", "Starting electrolyser simulations")
        engine_b, engine_m = SimulationEngine(df_b), SimulationEngine(df_m)
        res_b, res_m = engine_b.run(), engine_m.run()
        log_event("INFO", "Electrolyser simulations completed")

        # --- Apply scaling ---
        if AUTO_SCALE:
            lit_times = np.array([0,1000,5000,10000], dtype=int)
            lit_rates = np.array([2.0,5.0,10.0,14.0])
            hist = np.array(engine_b.degradation.degradation_rate_history)
            valid = lit_times[lit_times < len(hist)]
            nonzero = hist[valid] > 0
            UVH_SCALE = float(np.median(lit_rates[nonzero] / hist[valid][nonzero])) if nonzero.any() else 1.0
            log_event("SCALE", f"Auto‑computed UVH scale factor: {UVH_SCALE:.2f}")
        else:
            log_event("SCALE", f"Using manual UVH scale factor: {UVH_SCALE:.2f}")

        # --- Exports & logs ---
        sum_b = compute_summary_from_engine(engine_b, res_b, "Brighton", df_b)
        sum_m = compute_summary_from_engine(engine_m, res_m, "Muscat", df_m)
        pd.concat([sum_b, sum_m]).to_csv("simulation_summary.csv", index=False)
        res_b.to_csv("brighton_results.csv")
        res_m.to_csv("muscat_results.csv")
        pd.DataFrame({
            'Brighton_dR_uV_h_scaled': np.array(engine_b.degradation.degradation_rate_history) * UVH_SCALE,
            'Muscat_dR_uV_h_scaled':   np.array(engine_m.degradation.degradation_rate_history) * UVH_SCALE
        }, index=res_b.index).to_csv("scaled_degradation_rates.csv")
        pd.DataFrame(log_data).to_csv("simulation_log.csv", index=False)
        
        op_b = res_b[res_b['Total_Current'] > 1]
        print("Brighton operating‐point count:", len(op_b))
        print("Brighton temp range:", op_b['Ely_Temp'].min(), "to", op_b['Ely_Temp'].max())

        # --- Plot definitions ---
        MAIN_FIGS = [
            lambda: plot_combined_monthly_pv(df_b, df_m,
                                             labels=("Brighton","Muscat"),
                                             save_path="fig_main_1_pv.png"),
            lambda: plot_figure2_h2_and_cycles(res_b, res_m,
                                               save_path="fig_main_2_h2_cycles.png"),
            lambda: plot_validation_both_locations(res_b, res_m, engine_b, engine_m,
                               labels=('Brighton', 'Muscat'),
                               save_path='validation_both_locations.png'),
            lambda: plot_figure3_efficiency_and_loading(res_b, res_m,
                                                        save_path="fig_main_3_eff_load.png"),
            lambda: plot_degradation_bmc(res_b, engine_b, res_m, engine_m, res_const, engine_const, save_path="degradation_bmc.png")

        ]

        APPENDIX_FIGS = [
            lambda: plot_detailed_comparison(res_b, res_m, engine_b, engine_m,
                                             labels=('Brighton','Muscat'),
                                             save_path="fig_appA_1_detailed.png"),
            lambda: plot_advanced_analytics(res_b, df_b, "Brighton",
                                            save_path="fig_appA_2_advanced.png"),
            lambda: plot_economic_analysis(res_b, res_m,
                                           labels=("Brighton","Muscat"),
                                           save_path="fig_appA_3_economic.png"),
            lambda: plot_figure4_degradation(res_b, res_m, engine_b, engine_m,
                                             save_path="fig_appA_4_degradation.png"),
            lambda: plot_figure1_pv_summary(df_b, df_m,
                                             save_path="fig_appA_5_pv_summary.png"),
        ]

        # generate main text figures
        for fn in MAIN_FIGS:
            fn()

        # generate appendix figures
        for fn in APPENDIX_FIGS:
            fn()

        log_event("SUCCESS", "All simulations and analyses completed successfully")
        print("All CSVs and plots have been saved.")

    except Exception as e:
        log_event("ERROR", f"Simulation failed with error: {e}")
        print(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    main()

