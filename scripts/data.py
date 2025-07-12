# ReadMe
#=========================================================================
"""
data.py
Mert-chan 16 Feb 2025
- Loads training data from a pickle file
- Returns a dictionary necessary to run the main scripts
- Controls the data split
"""
#=========================================================================

# Libraries
#========================================================================
import os
import sys
import pickle
import torch.distributed as dist  # Added to check rank for DDP
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List,Any
from source.myTrainFuns import TECspatiotemporalLoader
from datetime import datetime,timedelta
#========================================================================

# ---- registry so you can plug new experiment loaders later -------------
DATASET_FACTORY = {}

def register_exp(name):
    def dec(fn):
        if name in DATASET_FACTORY:
            raise ValueError(f"{name} already registered")
        DATASET_FACTORY[name] = fn
        return fn
    return dec

# Function to load training data
#========================================================================
def load_training_data(seq_len=12, pred_horz=12, datasplit='stratifiedSplit', verbose=True, features=None, base_path=None) -> Dict:
    """
    Inputs:
        seq_len (int): The input sequence length (default 12).
        pred_horz (int): The prediction horizon (default 12).
        datasplit (str): The type of data split to use 'stratifiedSplit', 'chronologicalSplit' (default 'Stratified').
        verbose (bool): Whether to print output (default True).
        features (list): List of feature names for OMNI to select (default None, selects paper's 17 features).
    Returns:
        dict containing the following keys:
            dates (list): List of dates of dataset period.
            normTEC (np.ndarray): Normalized TEC data. (Min-Max)
            normOMNI (np.ndarray): Normalized OMNI data. (Min-Max)
            OMNI_names (list): List of OMNI column names (auxiliary inputs).
            dataSplit_Dates (dict): Dictionary containing the split dates for subsets.
            stormInfo (dict): Dictionary containing storm information. (Extracted as <-100nT from the original data)
            maxTEC (float): Maximum TEC value. (Use for denormalization)
            minTEC (float): Minimum TEC value. (Use for denormalization)
            NUM_OMNI (int): Number of OMNI columns (auxiliary inputs).
            base_path (str): Base path of the project.
    """
    if base_path is None:
        base_path = Path().resolve().parents[0]                  # ..~/IonoBench
    training_data_path = Path(base_path, "datasets")             # ..~/IonoBench/datasets 
    fun_path = os.path.join(base_path, "source")                 # ..~/IonoBench/source
    
    # Add function path to sys.path if needed
    if fun_path not in sys.path:
        sys.path.append(fun_path)

    # Only print on master process (for DDP)
    if verbose and (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
        print("-" * 20)
        print(
            f"Base Path: {base_path}",
            f"Function Path: {fun_path}",
            f"Training Data Path: {training_data_path}",
            sep="\n"
        )
        print("-" * 20)

    # Load your pickle data (Fix)
    fileName = 'IonoBench_' + datasplit + 'Split.pickle'
    training_data_file = os.path.join(training_data_path, fileName)
    try:
        with open(training_data_file, 'rb') as handle:
            trainingData = pickle.load(handle)
        [dates, normTEC, normOMNI, OMNI_names, dataSplit_Dates, stormInfo, maxTEC, minTEC] = trainingData.values()
        print("Training data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading training data: {e}\n Please ensure the dataset exists at {training_data_file}.")
        return None
    
    # Map feature names to indices
    if features is None:
        # Default to paper's 17 features (keeping original behavior)
        selectedidx = [1,4,5,6,9,10,11,12,13,14,16,17,19,27,28,29,30]
    else:
        # Convert OMNI_names to a list for easier searching
        omni_name_list = OMNI_names.iloc[:, 0].tolist()
        selectedidx = []
        
        for feature in features:
            try:
                idx = omni_name_list.index(feature)
                selectedidx.append(idx)
            except ValueError:
                if verbose and (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"Warning: Feature '{feature}' not found in OMNI data. Skipping.")
        
        if not selectedidx:
            raise ValueError("No valid features found in OMNI data")
    
    selectedOMNI = normOMNI[:, selectedidx]  # shape: [N_samples, len(selectedidx)]
    selectedInput_Names = OMNI_names.iloc[selectedidx, 0].tolist()
    numOMNI = len(selectedidx)
    
    if verbose and (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
        print("-" * 20)
        print(
            "Important Details:\n"
            f"Data Range: {dates[0]} - {dates[-1]}\n"
            f"Shapes=> Dates: {len(dates)}, TEC: {normTEC.shape}, OMNI: {selectedOMNI.shape}\n"
            f"Data Split Sets: {list(dataSplit_Dates.keys())}\n"
            f"Storm Info: {list(stormInfo.keys())}\n"
            f"Max TEC: {maxTEC:.3f}, Min TEC: {minTEC}"
        )
        print("-" * 20)

    # --------------------------------------------------------------------
    # Train/Valid/Test Split 
    # --------------------------------------------------------------------
    dataSplit_Dates_new = {}
    dataSplit_Dates_new['train'] = sum([dataSplit_Dates[key] for key in dataSplit_Dates.keys() if "train" in key], [])
    dataSplit_Dates_new['valid'] = sum([dataSplit_Dates[key] for key in dataSplit_Dates.keys() if "valid" in key], [])
    dataSplit_Dates_new['test'] = sum([dataSplit_Dates[key] for key in dataSplit_Dates.keys() if "test" in key], [])
    dataSplit_Dates_new['train'] = dataSplit_Dates_new['train'][seq_len + pred_horz:]     # Remove first seq_len + pred_horz samples to make sure first samples are usable.
    if verbose and (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
        print(f'Train: {len(dataSplit_Dates_new["train"])}, Valid: {len(dataSplit_Dates_new["valid"])}, Test: {len(dataSplit_Dates_new["test"])}')
        total = len(dataSplit_Dates_new['train']) + len(dataSplit_Dates_new['valid']) + len(dataSplit_Dates_new['test'])
        print(f"Dataset Name: {datasplit} \n Ratio => Train: {len(dataSplit_Dates_new['train'])/total:.2f}, Valid: {len(dataSplit_Dates_new['valid'])/total:.2f}, Test: {len(dataSplit_Dates_new['test'])/total:.2f}")

    return {
        "dates": dates,
        "normTEC": normTEC,
        "normOMNI": selectedOMNI,               
        "OMNI_names": selectedInput_Names,
        "dataSplit_Dates": dataSplit_Dates_new,
        "stormInfo": stormInfo,
        "maxTEC": maxTEC,
        "minTEC": minTEC,
        "NUM_OMNI": numOMNI,
        "base_path": base_path,
        "training_data_path": training_data_path,
        "script_dir": os.path.dirname(base_path)
    }
#========================================================================


# Function to prepare the data following the DataConfig
#========================================================================
def prepare_raw(cfg) -> Dict:
    """Wraps load_training_data & patches cfg with shape/min-max info."""
    d = load_training_data(
            seq_len      = cfg.data.seq_len,
            pred_horz    = cfg.data.pred_horz,
            datasplit    = cfg.data.data_split,
            features  = cfg.data.get('features', None),
            base_path= cfg.paths.base_dir
        )
    if not d:
        raise RuntimeError("load_training_data() returned empty dict")

    C, H, W = d["NUM_OMNI"] + 1, d["normTEC"].shape[1], d["normTEC"].shape[2]
    cfg.data.num_omni, cfg.data.H, cfg.data.W = d["NUM_OMNI"], H, W
    cfg.test.max_tec = float(d["maxTEC"])   # cast numpy → native float
    cfg.test.min_tec = float(d["minTEC"])
    return d
#========================================================================


# Dataloaders for default train/valid/test splits
#========================================================================
@register_exp("default")
def make_default_loaders(cfg, d) -> Dict[str, DataLoader]:
    """train / valid / test – the usual."""
    T = cfg.data.seq_len
    pred_horz = cfg.data.pred_horz
    def get_loader(idxs, split):
        if split == "test":
            B = cfg.test.batch_size
        else:  # for 'train' and 'valid'
            B = cfg.train.batch_size
        shuffle = (split == "train")
        return DataLoader(
            TECspatiotemporalLoader(d["normTEC"], d["normOMNI"], idxs, d["dates"], T, pred_horz),
            batch_size=B,
            shuffle=shuffle
        )

    return {split: get_loader(idxs, split) for split, idxs in d["dataSplit_Dates"].items()}
#========================================================================


# Dataloaders for testing by solar-activity class (Very-Weak / Weak / Moderate / Intense)
#========================================================================
@register_exp("solar")
def make_solar_loaders(cfg, base_path, d) -> Dict[str, DataLoader]:
    """
    Returns one DataLoader per solar-activity class, using the batch-size
    specified in cfg.test.batch_size.

    cfg  : merged config obj (has .data.seq_len, .data.pred_horz, .test.batch_size)
    d    : dict returned by prepare_raw()  (contains dates, normTEC, normOMNI …)
    """
    # 1 - Build the class → index list mapping
    from source.myDataFuns import SI_categorize          # already in your repo
    
    omni_path = os.path.join(base_path, 'datasets', cfg.data.omni_file)
    solar_idx = SI_categorize(
        allDATES      = d["dates"],
        desiredDates     = d["dataSplit_Dates"]["test"],
        OMNI_path      = omni_path,
        verbose        = False
    )  # keys: "Very Weak" … "Intense"  →  list[int]

    key_map = {                          
        "Very Weak": "very_weak",
        "Weak":       "weak",
        "Moderate":   "moderate",
        "Intense":    "intense",
    }
    d["solarClasses"] = solar_idx
    # 2 - Convenience aliases
    T, H, W       = cfg.data.seq_len, d["normTEC"].shape[1], d["normTEC"].shape[2]
    pred_horz     = cfg.data.pred_horz
    B             = cfg.test.batch_size
    normTEC       = d["normTEC"]
    normOMNI      = d["normOMNI"]
    dates         = d["dates"]

    # 3 - Factory that respects the common loader signature
    def _make(idx_list):
        return DataLoader(
            TECspatiotemporalLoader(normTEC, normOMNI, idx_list, dates, T, pred_horz),
            batch_size = B,
            shuffle    = False
        )

    return {key_map[k]: _make(v) for k, v in solar_idx.items()}

# Dataloaders for geomagnetic storm analysis 
#=========================================================================
@register_exp("storm")
def make_storm_loaders(cfg: dict, d: dict) -> dict:
    """
    Identifies geomagnetic storms in the test set and creates a dictionary
    of DataLoader pairs (full period & main phase) for each storm.

    This serves as the data preparation step for the StormAnalysis class.

    Args:
        cfg: The main configuration object.
        d: A dictionary containing raw data like 'stormInfo', 'normTEC', etc.

    Returns:
        A dictionary where each key is a unique storm identifier and each value
        is another dictionary containing loaders and metadata for that storm.
        Example:
        {
            "storm_0_2015-03-17": {
                "full_loader": <DataLoader>,
                "main_loader": <DataLoader>,
                "meta": { ... storm metadata ... }
            },
            ...
        }
    """
    print("Preparing DataLoaders for storm analysis...")
    test_dates = set(d["dataSplit_Dates"]["test"])
    all_storms = [s for intensity in d["stormInfo"].values() for s in intensity]

    # Find and sort storms that are part of the test set
    storms_to_analyze = sorted(
        [s for s in all_storms if set(s['Storm Period']) & test_dates],
        key=lambda s: s['Storm Dst']
    )
    
    print(f"Found {len(storms_to_analyze)} storms intersecting with the test set.")
    
    storm_loaders = {}
    T = cfg.data.seq_len # Input sequence length

    for i, storm_meta in enumerate(storms_to_analyze):
        # --- Define date periods for this storm ---
        full_period_dates = storm_meta['Storm Period'] + [
            storm_meta['Storm Period'][-1] + timedelta(hours=2 * j) for j in range(1, T + 1)
        ]
        start = cfg.storm.main_phase.start_index
        end = cfg.storm.main_phase.end_index
        main_phase_dates = storm_meta['Storm Period'][start:end]

        # --- Create Datasets and DataLoaders ---
        full_set = TECspatiotemporalLoader(
            d['normTEC'], d['normOMNI'], full_period_dates, d['dates'],
            cfg.data.seq_len, cfg.data.pred_horz
        )
        main_set = TECspatiotemporalLoader(
            d['normTEC'], d['normOMNI'], main_phase_dates, d['dates'],
            cfg.data.seq_len, cfg.data.pred_horz
        )

        full_loader = DataLoader(full_set, batch_size=cfg.data.batch_size, shuffle=False)
        main_loader = DataLoader(main_set, batch_size=cfg.data.batch_size, shuffle=False)
        
        # --- Create a unique tag and store everything ---
        tag = f"storm_{i}_{storm_meta['Storm Date'].strftime('%Y-%m-%d')}"
        storm_loaders[tag] = {
            'full_loader': full_loader,
            'main_loader': main_loader,
            'meta': storm_meta
        }

    return storm_loaders
#=========================================================================

# Dataloaders for C1PG comparison periods
#=========================================================================
@register_exp("c1pg_comparison")
def make_c1pg_loaders(cfg: Any, d: Dict) -> Dict[str, DataLoader]:
    # ... (Paste the full make_c1pg_loaders function from Section 1 here)
    """
    Creates DataLoaders for specific C1PG comparison periods, which include
    solar descending, low activity, and high activity phases.

    Args:
        cfg: The main configuration object.
        d: The dictionary returned by prepare_raw() containing the full dataset.

    Returns:
        A dictionary of DataLoaders for each C1PG comparison period.
    """
    is_main_proc = not dist.is_initialized() or dist.get_rank() == 0
    if is_main_proc:
        print("Preparing DataLoaders for C1PG comparison periods...")
    
    def generate_date_range(start, end, step):
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += step
        return dates

    c1pg_periods = {
        "solar_descend": generate_date_range(datetime(2015, 11, 1), datetime(2017, 9, 5, 22), timedelta(hours=2)),
        "low_activity": generate_date_range(datetime(2019, 1, 1), datetime(2020, 9, 28, 22), timedelta(hours=2)),
        "high_activity": generate_date_range(datetime(2023, 9, 30), datetime(2024, 9, 8, 22), timedelta(hours=2)),
    }

    missing_day = datetime(2019, 6, 16).date()
    c1pg_periods["low_activity"] = [dt for dt in c1pg_periods["low_activity"] if dt.date() != missing_day]
    
    T = cfg.data.seq_len
    pred_horz = cfg.data.pred_horz
    B = cfg.test.batch_size
    
    def _make_loader(date_list: List[datetime], name: str):
        available_dates = set(d["dates"])
        valid_dates = [date for date in date_list if date in available_dates]
        
        if not valid_dates and is_main_proc:
            print(f"Warning: No data available in the main dataset for C1PG period '{name}'.")
            return None

        dataset = TECspatiotemporalLoader(d["normTEC"], d["normOMNI"], valid_dates, d["dates"], T, pred_horz)
        return DataLoader(dataset, batch_size=B, shuffle=False)

    loaders = {
        name: loader 
        for name, dates in c1pg_periods.items() 
        if (loader := _make_loader(dates, name)) is not None
    }
    
    if is_main_proc:
        print(f"Created {len(loaders)} C1PG comparison loaders: {list(loaders.keys())}")
    return loaders
#=========================================================================