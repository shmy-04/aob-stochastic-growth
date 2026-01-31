"""
This module simulates Nitrosomonas cell growth and division.
The overall simulation framework was inspired by a previous study
(Witz et al., eLife, 2019), which modeled E. coli cell cycle and size control.
All model assumptions, parameterizations, and analyses
were independently developed based on experimental data obtained in this study.
"""
# Original conceptual inspiration:
#   Witz, G. et al., eLife (2019)
#
# Implementation:
#   Ikeda, S. et al., Journal (2026)


import numpy as np
import pandas as pd
import multiprocessing
import random
from collections import defaultdict
from scipy.stats import truncnorm

def process_cell(cells, current_time):
    """process cell growth and identify dividing cells"""
    divide_cells = []
    new_total_cells_dA = 0
    dt = 1 # hour
    
    for x in range(len(cells)):
        cells['preA'][x] = cells['A'][x]
        cells['A'][x] = min(cells['A'][x] * np.exp(cells['tau'][x] * dt),
                            cells['max_Ad'][x]) 
        dA = cells['A'][x] - cells['preA'][x]
        if dA < 0:
            if dA > -1e-6:
                dA = 0
            else:
                print(f"Warning: Negative growth detected for cell {x} at time {current_time}. dA: {dA}")
        
        # add cell elongation until division
        cells['dA'][x] += dA
        new_total_cells_dA += dA

        # Note: both conditions must be satisfied for division.
        if cells['age'][x] > cells['generation_time'][x] and cells['A'][x] > cells['Ad_sizer'][x]: 
            divide_cells.append(cells['id'][x])
    
    return cells, divide_cells, new_total_cells_dA

def parallel_process(cells, t):
    """parallel processing of cell growth and division identification"""
    num_processes = multiprocessing.cpu_count()  # get available CPU core number
    cells_split = np.array_split(cells, num_processes)  # split cell data by number of cores

    # conduct parallel processing using multiprocessing
    with multiprocessing.Pool() as pool:
        results = pool.starmap(
            process_cell,
            [(cells_chunk, t) for cells_chunk in cells_split]
        )
    
    # get results
    cells = np.concatenate([result[0] for result in results])
    total_cells_dA = np.sum([result[2] for result in results])
    divide_cells = []
    for result in results:
        divide_cells.extend(result[1])

    return cells, divide_cells, total_cells_dA

# mathematical functions
def logistic(x, y_max, r, x0):
    return y_max / (1 + np.exp(- (r * (x - x0) ) ) )

def logistic_3d(x1, x2, y_max, r1, r2, x01, x02):
    return y_max / (1 + np.exp(- (r1 * (x1 - x01) - r2 * (x2 - x02))))

def power_decay_3d(x1, x2, z_max, b1, b2, z_min):
    eps = 1e-12
    return (z_max-z_min) * (x1 + eps)**(-b1) * (x2 + eps)**(-b2) + z_min

def hill_decay(x, y_min, y_max, x_c, k):
    return y_min + (y_max - y_min) / (1 + (x/x_c)**k)

# reservoir sampling for cell history
def reservoir_add(cell_info, T0_value, reservoirs, counts, k=1000):
    counts[T0_value] += 1
    n_seen = counts[T0_value]

    if len(reservoirs[T0_value]) < k:
        reservoirs[T0_value].append(cell_info)
    else:
        j = random.randint(0, n_seen - 1)
        if j < k:
            reservoirs[T0_value][j] = cell_info

def reservoirs_to_dataframe(reservoirs):
    all_cells = []
    for _, cells in reservoirs.items():
        for cell_info in cells:
            all_cells.append(cell_info)
    
    df = pd.DataFrame(all_cells, columns=[
        'id', 'Ab', 'Ad', 'Tb', 'age', 
        'tau', 'Ad_sizer', 'generation_time', 'divR',
        'biomass_production_density_at0' # ∆Vt at birth
    ])
    return df

def r_truncnorm(mu, sigma, lower, upper, size):
    """truncated normal distribution random sampling"""
    a = (lower - mu)/sigma
    b = (upper - mu)/sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size)


def simul_Nitrosomonas(params):
    """main simulation function"""
    dtype = np.dtype([
        ('id', 'uint32'),
        ('A0', 'float64'),
        ('A', 'float64'),
        ('dA', 'float64'),
        ('Ad_sizer', 'float64'),
        ('preA', 'float32'),
        ('max_Ad', 'float64'),
        ('T0', 'float32'),
        ('generation_time', 'float64'),
        ('age', 'float32'),
        ('tau', 'float64'), # elongation rate, alpha
        ('biomass_production_density_at0', 'float64'),
        ('divR', 'float64'),
        ('generation', 'int32')
    ])
        
    # make initial cells
    cells = np.zeros(params['nbstart'], dtype=dtype)
    cells['id'] = np.arange(params['nbstart'], dtype=np.uint32)
    # T0, α0 from KDE
    log_samples = params['kde_log'].resample(params['nbstart'])
    cells['generation_time'], cells['tau'] = np.exp(log_samples)
    # A0 form RF model(QRF)
    X_data = np.vstack((cells['generation_time'], cells['tau'])).T
    all_tree_preds = np.array([tree.predict(X_data) for tree in params["rf"].estimators_])
    quantiles = [5, 25, 50, 75, 95]
    rf_quantiles = np.percentile(all_tree_preds, quantiles, axis=0)
    rf_median = rf_quantiles[2]
    cells['A0'] = rf_median
    if (cells['A0'] < 0).any():
        print("Warning: some predicted A0 values are negative.")
    # initialize max_Ad
    cells['max_Ad'] = float(params['maxAd_max'])
    # Ad_sizer(A_min, div). mu ± 3σ * 1.0136 (= truncnorm 0.001 quantile).
    lower_bound = params["Ad_sizer"] - 3* params["Ad_sizer_sigma"]*1.0136
    upper_bound = params["Ad_sizer"] + 3* params["Ad_sizer_sigma"]*1.0136
    cells['Ad_sizer'] = r_truncnorm(params["Ad_sizer"], params["Ad_sizer_sigma"],
                                    lower_bound, upper_bound,
                                    size=params['nbstart']).astype(np.float64)
    # initialize divR. mu ± 3σ * 1.0136 (= truncnorm 0.001 quantile).
    lower_bound = np.maximum(params['divR_min'], params['divR_mu'] - 3* params['divR_sigma']*1.0136)
    upper_bound = np.minimum(params['divR_max'], params['divR_mu'] + 3* params['divR_sigma']*1.0136)
    cells['divR'] = r_truncnorm(params['divR_mu'], params['divR_sigma'],
                                lower_bound, upper_bound, 
                                size=params['nbstart']).astype(np.float64)
    # Initialize other cell properties
    cells['A'] = cells['A0']
    cells['dA'] = 0
    cells['preA'] = np.nan
    cells['T0'] = 0
    cells['age'] = 0 
    cells['generation'] = 0
    cells['biomass_production_density_at0'] = 0
    
    # initialize tracking variables
    nitrite_production = 0
    divide_cells = []
    total_cells_dA = 0
    cell_count_transition = [[None, None, None, None]]

    # initialize divide_cell_info
    divide_cell_info = []
    reservoirs = defaultdict(list)
    counts = defaultdict(int)

    for t in range(0, params['run_time']):
        if t % 10 == 0:
            print(f"Time {t}: Cells {cell_count_transition[-1][1]}, µm3_Biomass_production_density: {cell_count_transition[-1][2]}, nitrite production: {cell_count_transition[-1][3]}")
        
        # current values
        current_cell_count = len(cells)
        current_nitrite_production = nitrite_production
        
        if current_cell_count < 100000:
            cells, divide_cells, total_cells_dA = process_cell(cells, t)
        else:
            cells, divide_cells, total_cells_dA = parallel_process(cells, t)
        
        # calculate ∆Vt
        current_biomass_production_density = total_cells_dA * 0.75 / params['culture_vol'] # µm3/mL
        # record transition
        cell_count_transition.append([t, current_cell_count, current_biomass_production_density, current_nitrite_production])
        
        # convert cells(numpy array) to pd.DataFrame
        cells_df = pd.DataFrame(cells)
        del cells
        
        # conduct cell division
        divide_cells_df = cells_df[cells_df['id'].isin(divide_cells)]
        if len(divide_cells_df) == 0:
            cells_df['age'] += 1  # update age for non-dividing cells
            
            # convert df to numpy array
            cells = np.zeros(len(cells_df), dtype=dtype)
            cells['id'] = cells_df['id'].astype(np.uint32)
            cells['A0'] = cells_df['A0'].astype(np.float64)
            cells['A'] = cells_df['A'].astype(np.float64)
            cells['dA'] = cells_df['dA'].astype(np.float64)
            cells['preA'] = cells_df['preA'].astype(np.float32)
            cells['T0'] = cells_df['T0'].astype(np.float32)
            cells['age'] = cells_df['age'].astype(np.float32)
            cells['biomass_production_density_at0'] = cells_df['biomass_production_density_at0'].astype(np.float64)
            cells['max_Ad'] = cells_df['max_Ad'].astype(np.float64)
            cells['tau'] = cells_df['tau'].astype(np.float64)
            cells['Ad_sizer'] = cells_df['Ad_sizer'].astype(np.float64)
            cells['generation_time'] = cells_df['generation_time'].astype(np.float64)
            cells['divR'] = cells_df['divR'].astype(np.float64)
            cells['generation'] = cells_df['generation'].astype(int)
            del cells_df
            
        if len(divide_cells_df) > 0:
            # record divide_cell_info
            for _, row in divide_cells_df.iterrows():
                divide_cell_info = [row["id"], row["A0"], row["A"], row["T0"], row["age"],
                                    row["tau"], row['Ad_sizer'], row["generation_time"], row["divR"],
                                    row["biomass_production_density_at0"]]
                # record to reservoir(max k=1000)
                reservoir_add(divide_cell_info, row["T0"], reservoirs, counts, k=1000)
            
            # initialize daughter_cells
            new_daughter_cell_num = len(divide_cells_df) * 2
            daughter_cells = np.zeros(new_daughter_cell_num, dtype=dtype)
            # setting IDs for daughter cells
            max_cell_id = cells_df['id'].max()
            daughter_cells['id'] = np.arange(max_cell_id+1, max_cell_id+1 + new_daughter_cell_num, dtype=np.uint32)
            # setting ∆Vt at birth for daughter cells
            daughter_cells['biomass_production_density_at0'] = current_biomass_production_density
            # setting A0 and A for daughter cells
            daughter_cells['A0'][:len(divide_cells_df)] = divide_cells_df['A'].values * divide_cells_df['divR'].values # divRを使って指定
            daughter_cells['A'][:len(divide_cells_df)] = daughter_cells['A0'][:len(divide_cells_df)]
            daughter_cells['A0'][len(divide_cells_df):] = divide_cells_df['A'].values * (1 - divide_cells_df['divR'].values) # divRを使って指定
            daughter_cells['A'][len(divide_cells_df):] = daughter_cells['A0'][len(divide_cells_df):]
            # max_Adの設定 
            daughter_cells['max_Ad'] = float(params['maxAd_max'])
            # Ad_sizer
            lower_bound = params["Ad_sizer"] - 3* params["Ad_sizer_sigma"]*1.0136
            upper_bound = params["Ad_sizer"] + 3* params["Ad_sizer_sigma"]*1.0136
            daughter_cells['Ad_sizer'] = r_truncnorm(params["Ad_sizer"], params["Ad_sizer_sigma"],
                                                     lower_bound, upper_bound,
                                                     size=new_daughter_cell_num).astype(np.float64)
            # setting generation_time for daughter cells from ∆Vt and A0(regression model)
            mu_Gtime = power_decay_3d(
                daughter_cells['biomass_production_density_at0'],
                daughter_cells["A0"],
                params['Gtime_mu_max'], 
                params['Gtime_r1'], params['Gtime_r2'],
                params['Gtime_mu_min']
                )
            Gtime_sigma = hill_decay(
                daughter_cells['biomass_production_density_at0'],
                params['Gtime_sigma_min'], 
                params['Gtime_sigma_max'], 
                params['Gtime_sigma_x_c'], 
                params['Gtime_sigma_k']
                )
            lower_bound = np.maximum(params["Gtime_min"], mu_Gtime - 3* Gtime_sigma*1.0136)
            upper_bound = np.minimum(np.inf, mu_Gtime + 3* Gtime_sigma*1.0136)
            daughter_cells['generation_time'] = r_truncnorm(mu_Gtime, Gtime_sigma,
                                                            lower_bound, upper_bound, 
                                                            size=new_daughter_cell_num).astype(np.float64)
            # setting tau for daughter cells from ∆Vt and A0 (regression model)
            mu_alpha = (
                logistic_3d(
                    np.log10(daughter_cells['biomass_production_density_at0']),
                    daughter_cells["A0"],
                    params['alpha_mu_max'], 
                    params['alpha_mu_r1'], params['alpha_mu_r2'],
                    params['alpha_mu_x01'], params['alpha_mu_x02'])
                )
            alpha_sigma = (
                logistic(
                    np.log10(daughter_cells['biomass_production_density_at0']),
                    params['alpha_sigma_max'], 
                    params['alpha_sigma_r'],
                    params['alpha_sigma_x0'])
                )
            lower_bound = np.maximum(0.0, mu_alpha - 3* alpha_sigma*1.0136)
            upper_bound = np.minimum(params["alpha_max"], mu_alpha + 3* alpha_sigma*1.0136)
            daughter_cells['tau'] = r_truncnorm(mu_alpha, alpha_sigma,
                                                lower_bound, upper_bound, 
                                                size=new_daughter_cell_num).astype(np.float64)
            # setting divR for daughter cells
            lower_bound = np.maximum(params['divR_min'], params['divR_mu'] - 3* params['divR_sigma']*1.0136)
            upper_bound = np.minimum(params['divR_max'], params['divR_mu'] + 3* params['divR_sigma']*1.0136)
            daughter_cells['divR'] = r_truncnorm(params['divR_mu'], params['divR_sigma'],
                                                 lower_bound, upper_bound, 
                                                 size=new_daughter_cell_num).astype(np.float64)
            # setting other properties for daughter cells
            daughter_cells['dA'] = 0
            daughter_cells['preA'] = np.nan
            daughter_cells['T0'] = t # time of birth
            daughter_cells['age'] = 0
            daughter_cells['generation'] = np.repeat(divide_cells_df['generation'].values, 2) + 1
            
            # update age for non-dividing cells
            del divide_cells_df
            cells_df = cells_df[~cells_df['id'].isin(divide_cells)]
            cells_df['age'] += 1
            
            # add daughter cells to cells_df
            daughter_cells_df = pd.DataFrame(daughter_cells)
            cells_df = pd.concat([cells_df, daughter_cells_df], axis=0, ignore_index=True)
            del daughter_cells_df
            
            # convert df to numpy array
            cells = np.zeros(len(cells_df), dtype=dtype)
            cells['id'] = cells_df['id'].astype(np.uint32)
            cells['A0'] = cells_df['A0'].astype(np.float64)
            cells['A'] = cells_df['A'].astype(np.float64)
            cells['dA'] = cells_df['dA'].astype(np.float64)
            cells['preA'] = cells_df['preA'].astype(np.float32)
            cells['T0'] = cells_df['T0'].astype(np.float32)
            cells['age'] = cells_df['age'].astype(np.float32)
            cells['biomass_production_density_at0'] = cells_df['biomass_production_density_at0'].astype(np.float64)
            cells['max_Ad'] = cells_df['max_Ad'].astype(np.float64)
            cells['tau'] = cells_df['tau'].astype(np.float64)
            cells['Ad_sizer'] = cells_df['Ad_sizer'].astype(np.float64)
            cells['generation_time'] = cells_df['generation_time'].astype(np.float64)
            cells['divR'] = cells_df['divR'].astype(np.float64)
            cells['generation'] = cells_df['generation'].astype(int)
            del cells_df
        
        nitrite_pmol = (len(cells) - params['nbstart']) / 33.5 # pmol, assuming 33.5 cells/pmol nitrite
        nitrite_mmol = nitrite_pmol * 1e-9 # mmol
        nitrite_mM = nitrite_mmol / 1e-3 # mM, assuming 1 mL volume
        nitrite_production = nitrite_mM
        
        if current_biomass_production_density > 3.0*10**6: # µm3/mL
            df = pd.DataFrame(cell_count_transition, columns=['Time', 'CellCount', 'biomass_production_density', 'Nitrite_production'])
            cell_history_df = reservoirs_to_dataframe(reservoirs)
            return df, cell_history_df
    
    df = pd.DataFrame(cell_count_transition, columns=['Time', 'CellCount', 'biomass_production_density', 'Nitrite_production'])
    cell_history_df = reservoirs_to_dataframe(reservoirs)
    
    return df, cell_history_df
