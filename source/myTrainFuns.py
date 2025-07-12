#####################################################################
"""20 Feb 2025 - Mert"""

# User Defined Functions
#===============================================================================
from myDataFuns import *
#===============================================================================

# Import Libraries
#===============================================================================
import os,glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import datetime
from tqdm import tqdm
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import pytorch_warmup as warmup
import torch.distributed as dist
import logging
import pickle
from filelock import FileLock
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, DistributedSampler
import datetime
#===============================================================================

# Function to seed random functions
#===============================================================================   
def seedME(seed):
    # It seems like calling random functions in different cells might need reseeding. To ensure repeadability, I will define a function to seed whenever I use a random function 
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy
    torch.manual_seed(seed)  # PyTorch CPU operations
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch CUDA (GPU) operations
        torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#===============================================================================
seedME(3)  # Seed everything for reproducibility (Why 3? My fav num :D )

# Function to initialize model weights
#===============================================================================
def initialize_model_weights(m, method = 'kaiming', bias = 'constant'):
    for layer in m.children():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            # Kaiming initialization formula for weights
            
            if method == 'kaiming':
                weights_initializer = torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            if method == 'xavier':
                weights_initializer = torch.nn.init.xavier_normal_(layer.weight, gain=1.0)
            
            # Initialize weights
            layer.weight.data = weights_initializer
            
            # Initialize biases (if applicable)
            if layer.bias is not None:
                if bias == 'zero':
                    layer.bias.data.zero_()
                if bias == 'normal':
                    layer.bias.data.norma_()
                if bias == 'constant':
                    layer.bias.data.fill_(0.01)
#===============================================================================

#==============================================================================
def is_dist_active():
    """Check if torch.distributed is initialized."""
    return dist.is_available() and dist.is_initialized()
#===============================================================================



# If DDP (Multi-GPU) training, remove the 'module.' to use for single GPU testing
#===============================================================================
def DDPtoSingleGPU(state_dict):
    """Remove the 'module.' prefix from state dict keys."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("module."):
            new_key = key[len("module."):]
        new_state_dict[new_key] = value
    return new_state_dict
#===============================================================================

# Learning Rate Range Test
#===============================================================================
def LRrangetest(model, lr_list, train_loader, device, config, plot=True):
    """Perform a quick LR range test over multiple one-epoch runs."""
    # If running in distributed mode, get rank/world_size; else default rank=0, world_size=1
    if config.mixedPrecision:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
        
    if is_dist_active(): rank = dist.get_rank()
    else: rank = 0
    losses = []
    for epoch, lr in enumerate(lr_list):
        seedME(3)                  # Reproducible init
        initialize_model_weights(model)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()

        local_loss_sum = 0.0
        local_samples  = 0

        for inputs_batch, truth_batch in tqdm(train_loader, desc=f"LR={lr}", leave=False):
            inputs_batch = inputs_batch.to(device).float()
            truth_batch  = truth_batch.to(device).float()

            optimizer.zero_grad()
            if config.mixedPrecision:
                with torch.amp.autocast('cuda'):
                    _, loss = model(inputs_batch, truth_batch)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                _, loss = model(inputs_batch, truth_batch)
                loss.backward()
                optimizer.step()

            local_loss_sum += loss.item()  # model outputs sum over batch
            local_samples  += inputs_batch.size(0)

        # If distributed, all-reduce sums
        if is_dist_active():
            loss_tensor   = torch.tensor(local_loss_sum, device=device)
            sample_tensor = torch.tensor(local_samples,  device=device)
            dist.all_reduce(loss_tensor,   op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
            global_loss_sum = loss_tensor.item()
            global_samples  = sample_tensor.item()
        else:
            global_loss_sum = local_loss_sum
            global_samples  = local_samples

        global_avg_loss = global_loss_sum / global_samples if global_samples > 0 else float('inf')

        # Print result (only on rank 0 if DDP)
        if rank == 0:
            print(f"[LR={lr}] Global average loss = {global_avg_loss}")

        losses.append(global_avg_loss)

        # (Optional) Barrier to sync ranks before next LR
        if is_dist_active():
            dist.barrier()
            
    if plot and rank == 0:
        fig = go.Figure(data=go.Scatter(x=lr_list, y=losses))
        fig.update_layout(title=f'LR rang Test - {config.session.name}', 
                        title_x=0.5, 
                        xaxis_title='Learning Rate', 
                        yaxis_title='Loss', template='plotly_dark') 
        fig.update_xaxes(type="log")
        fig.show()
                
    return losses
#===============================================================================


# Dataloader Class for Multi-Channel Spatiotemporal Data
#===============================================================================
class TECspatiotemporalLoader(torch.utils.data.Dataset):
    '''
    ===============================================================================
    Example of 12 in 12 out sequence: Where D₀ (target date) is the last prediction step
                [Input Sequence]                      [Prediction Horizon]
    TEC:   [ D-23 | D-22 | ... | D-12 ]   →   [ D-11 | D-10 | ... | D₀ ] ← Target Date
            ←──────────────┬──────────────→   ←────────┬────────────→
                            |                              |
                    Used for Input                  Used for Prediction Target
    D₀ is picked as last prediction step because it is the most challenging one. So most valuable to balance.
    Notes:
    - seq_len = 12 → D-23 to D-12 (12 maps, each 2 hours apart = 24 hours)
    - pred_horz = 12 → D-11 to D₀ (12 maps, next 24 hours to predict)
    - OMNI inputs align with D-23 to D-12 as well

    Final shapes:
    - input_sample: [channels, 12, H, W]
    - ground_truth: [12, H, W]
    ===============================================================================
    '''

    def __init__(self,tec,omni,selectedDates, allDates, seq_len,pred_horz):   # Initialize the Dataloader class
        self.tec = tec                                                     # TEC Data 
        self.omni = omni                                                   # OMNI Data
        self.selectedDates = selectedDates                                 # Selected Dates	(Train, Valid, Test)
        self.allDates = allDates                                           # All Dates (All Dates in the dataset range)
        self.seq_len = seq_len                                             # Input sequence Length 
        self.pred_horz = pred_horz                                         # Prediction Horizon	
        self.samples = len(selectedDates)                                  # Number of samples of the selected dates
        
    def __getitem__(self, index):           # Get the date sample index 
        d = self.selectedDates[index]       # Find the date sample inside selected dates and get the date as D₀ (target date)
        idx = self.allDates.index(d)        # Find the corresponding index of D₀ in all dates of dataset (OMNI,TEC and Date follows same order and len of samples)                                                                
        ground_truth = self.tec[idx-(self.pred_horz-1):idx+1].astype(np.float32)           # Truth Sequence=>  target idx and previous 11 samples as input. Target date is last sample.
        tec_seq = self.tec[idx-(self.seq_len + self.pred_horz-1):idx-(self.pred_horz-1)]   # TEC Sequence => Past 24 samples (12 maps) up until past 12 samples (start of truth sequence)
        omni2d = readOMNIseq(self.allDates,d,self.seq_len, self.pred_horz,self.omni)       # Similarly OMNI sequence but it is done in defined function while reading and reshaping (H,W)
        input_sample = np.concatenate(( np.expand_dims(tec_seq,axis=0).astype(np.float32) , omni2d.astype(np.float32)), axis=0) # Unite TEC and OMNI sequences
        # Return the input and truth samples as tensors
        return torch.tensor(input_sample), torch.tensor(ground_truth)        # Shapes: [C, 12, H, W], [12, H, W]                                                      
    
    def __len__(self):  
        return self.samples
#===============================================================================

# ModelTester Class
################################################################################  
class IonoTester:
    def __init__(self, model, test_loader, device, config=None, verbose=True):
        seedME(3)
        self.config = config if config is not None else {}
        if is_dist_active():
            self.rank = dist.get_rank()
        else:
            self.rank = 0
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.verbose = verbose
        base = Path(self.config.paths.base_dir)
        self.sessionDir = base / self.config.paths.out_dir / self.config.session.name

    def denormalize(self, data, max_val, min_val):
        """Denormalize the TEC data."""
        return data * (max_val - min_val) + min_val

    def test(self):
        """Main testing loop: calculates metrics and optionally saves raw data."""
        seedME(3)
        if self.rank == 0 and self.verbose:
            print('Testing the model...\n')
        self.model.eval()

        # --- Setup for incremental metrics and raw data saving ---
        num_horizons = self.config.data.pred_horz
        metric_lists = {
            'rmse': [], 'r2': [], 'ssim': [],
            'rmse_hor': [[] for _ in range(num_horizons)],
            'r2_hor': [[] for _ in range(num_horizons)],
            'ssim_hor': [[] for _ in range(num_horizons)],
        }
        
        # should_save_raw = self.config.test.save_raw and (self.config.mode == 'test')
        should_save_raw = self.config.test.save_raw
        temp_batch_dir = self.sessionDir / "temp_raw_batches"
        if self.rank == 0 and should_save_raw:
            if temp_batch_dir.exists(): shutil.rmtree(temp_batch_dir)
            temp_batch_dir.mkdir(parents=True)

        for i, (delayed_tec_batch, truthMaps_batch) in enumerate(tqdm(self.test_loader, disable=(self.rank != 0 or not self.verbose))):
            delayed_tec_batch = delayed_tec_batch.to(self.device, non_blocking=True).float()
            truthMaps_batch = truthMaps_batch.to(self.device, non_blocking=True).float()

            with torch.no_grad():
                output_batch, _ = self.model(delayed_tec_batch, truthMaps_batch)
            
            de_output = self.denormalize(output_batch.cpu().numpy(), self.config.test.max_tec, self.config.test.min_tec)
            de_label = self.denormalize(truthMaps_batch.cpu().numpy(), self.config.test.max_tec, self.config.test.min_tec)

            self._compute_and_append_batch_metrics(de_output, de_label, metric_lists)
            
            if self.rank == 0 and should_save_raw:
                np.savez(temp_batch_dir / f"batch_{i}.npz", preds=de_output, labels=de_label)
            
            del delayed_tec_batch, truthMaps_batch, output_batch, de_output, de_label

        torch.cuda.synchronize()
        if is_dist_active(): torch.distributed.barrier()

        tec_predictions, tec_labels, metrics_all = None, None, None
        if self.rank == 0:
            if self.verbose: print('Testing complete.\n')
            metrics_all = self._aggregate_and_log_metrics(metric_lists, verbose=self.verbose)
            
            if should_save_raw:
                tec_predictions, tec_labels = self._reassemble_raw_data(temp_batch_dir)
                shutil.rmtree(temp_batch_dir) # Clean up

        return {'Predictions': tec_predictions, 'Labels': tec_labels, 'Metrics': metrics_all}
        
    def _compute_and_append_batch_metrics(self, preds_batch, labels_batch, metric_lists):
        """Computes metrics for a single batch and appends them to the master lists."""
        data_range = self.config.test.max_tec - self.config.test.min_tec
        
        # Overall Metrics (per sample)
        metric_lists['rmse'].extend(np.sqrt(np.mean((preds_batch - labels_batch)**2, axis=(1, 2, 3))))
        for i in range(preds_batch.shape[0]):
            true, pred = labels_batch[i], preds_batch[i]
            metric_lists['r2'].append(r2_score(true.flatten(), pred.flatten()))
            ssim_scores = [ssim(true[h], pred[h], data_range=data_range) for h in range(true.shape[0])]
            metric_lists['ssim'].append(np.mean(ssim_scores))

        # Per-Horizon Metrics
        for h in range(preds_batch.shape[1]):
            pred_h, label_h = preds_batch[:, h], labels_batch[:, h]
            metric_lists['rmse_hor'][h].extend(np.sqrt(np.mean((pred_h - label_h)**2, axis=(1, 2))))
            metric_lists['r2_hor'][h].extend([r2_score(t.flatten(), p.flatten()) for t, p in zip(label_h, pred_h)])
            metric_lists['ssim_hor'][h].extend([ssim(t, p, data_range=data_range) for t, p in zip(label_h, pred_h)])

    def _aggregate_and_log_metrics(self, metric_lists, verbose=True):
        """Takes lists of scalar metrics, computes final stats, prints, and returns them."""
        overall = {
            'RMSE(TECU)': np.nanmean(metric_lists['rmse']), 'RMSE_std': np.nanstd(metric_lists['rmse']),
            'R²': np.nanmean(metric_lists['r2']), 'R²_std': np.nanstd(metric_lists['r2']),
            'SSIM': np.nanmean(metric_lists['ssim']), 'SSIM_std': np.nanstd(metric_lists['ssim']),
        }
        per_horizon = {
            'RMSE': [np.nanmean(h) for h in metric_lists['rmse_hor']], 'RMSE_std': [np.nanstd(h) for h in metric_lists['rmse_hor']],
            'R²': [np.nanmean(h) for h in metric_lists['r2_hor']], 'R²_std': [np.nanstd(h) for h in metric_lists['r2_hor']],
            'SSIM': [np.nanmean(h) for h in metric_lists['ssim_hor']], 'SSIM_std': [np.nanstd(h) for h in metric_lists['ssim_hor']],
        }
        results = {'sessionName': self.config.session.name, 'overall': {}, 'perHorizon': {}}
        for k, v in overall.items(): results['overall'][k] = round(v, 4)
        for k, v in per_horizon.items(): results['perHorizon'][k] = [round(float(x), 4) for x in v]

        if verbose: self._print_metrics(results)
        if self.config.test.save_results and (self.config.mode == 'test'): self.save_testing_info(results)
        return results
        
    def _reassemble_raw_data(self, temp_dir):
        """Loads temporary batch files from disk and assembles them into full arrays."""
        if self.verbose:
            print("\nReassembling raw data from disk. This may consume significant RAM.")
        
        batch_files = sorted(temp_dir.glob('batch_*.npz'), key=lambda x: int(x.stem.split('_')[1]))
        if not batch_files: return None, None
        
        # Pre-allocate memory for efficiency
        total_samples = len(self.test_loader.dataset)
        with np.load(batch_files[0]) as data:
            pred_shape = (total_samples,) + data['preds'].shape[1:]
            label_shape = (total_samples,) + data['labels'].shape[1:]
        
        all_preds = np.zeros(pred_shape, dtype=np.float32)
        all_labels = np.zeros(label_shape, dtype=np.float32)
        
        current_pos = 0
        for path in tqdm(batch_files, desc="Assembling raw data",disable=not self.verbose):
            with np.load(path) as data:
                n = data['preds'].shape[0]
                all_preds[current_pos:current_pos + n] = data['preds']
                all_labels[current_pos:current_pos + n] = data['labels']
                current_pos += n
        
        ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        raw_path = self.sessionDir / f"test_raw_{ts}.npz"
        if self.config.mode == 'test':
            np.savez_compressed(raw_path, predictions=all_preds, labels=all_labels)
            if self.verbose: print(f"Saved reassembled raw data -> {raw_path}")
        return all_preds, all_labels

    def _print_metrics(self, results_dict):
        """Helper function to print formatted metrics."""
        overall = results_dict['overall']
        per_horizon = results_dict['perHorizon']
        print("Overall Metrics:")
        print(f"  RMSE(TECU):  {overall['RMSE(TECU)']:.4f} ± {overall['RMSE_std']:.4f}")
        print(f"  R²:          {overall['R²']:.4f} ± {overall['R²_std']:.4f}")
        print(f"  SSIM:        {overall['SSIM']:.4f} ± {overall['SSIM_std']:.4f}")
        print("\nPer-Horizon Metrics:")
        for key in ['RMSE', 'R²', 'SSIM']:
            print(f"  {key} per step:     {per_horizon[key]}")
            print(f"  {key} std per step: {per_horizon[key+'_std']}")

    def save_testing_info(self, testdict):
        """Saves a text log of the testing session results."""
        self.sessionDir.mkdir(parents=True, exist_ok=True)
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        info_file = self.sessionDir / f"test_log_{current_time}.txt"
        
        with open(info_file, 'w') as f:
            f.write(f"Testing Session Information for: {self.config.session.name}\n" + "="*80 + "\n")
            f.write(f"Timestamp: {current_time}\n")
            f.write(f"Inputs: {self.config.data.features}\n\n")
            # Reuse the print helper for logging to file
            import io
            from contextlib import redirect_stdout
            string_io = io.StringIO()
            with redirect_stdout(string_io):
                self._print_metrics(testdict)
            f.write(string_io.getvalue())
            
        if self.verbose: print(f"\nTesting info file created at {info_file}")
###################################################################


# If you wanna log in the same testing file the further analysis use this
#===============================================================================
def get_latest_test_log(session_dir: Path) -> Path:
    files = sorted(session_dir.glob("testing_info*.txt"), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"No testing_info file in {session_dir}")
    return files[-1]
#===============================================================================

# Solar Intensity Analysis
#===============================================================================
class SolarAnalysis:
    _MAP = {
        "very_weak": "Very Weak",
        "weak":      "Weak",
        "moderate":  "Moderate",
        "intense":   "Intense",
    }

    def __init__(self,
                 model,
                 raw_dict: dict,
                 loaders: dict | None = None,  # if None, built from factory
                 device: str = "cpu",
                 cfg: dict | None = None,
                 verbose: bool = True):

        # … distributed housekeeping (unchanged) …
        if is_dist_active():
            self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        else:
            self.rank, self.world_size = 0, 1

        self.m       = model
        self.raw     = raw_dict
        self.cfg     = cfg if cfg is not None else {}
        self.dev     = device
        self.verbose = verbose and self.rank == 0

        # build loaders on-the-fly if they were not supplied
        if loaders is None:
            from scripts.data import DATASET_FACTORY
            loaders = DATASET_FACTORY["solar"](cfg, raw_dict)
        self.loaders = loaders

        # output directory + (possibly) log-file path
        base = Path(cfg.paths.base_dir)
        session_dir = base / cfg.paths.out_dir / cfg.session.name
        
        # Main directory for the log file
        self.sdir: Path = session_dir
        self.sdir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectory for raw .npz files
        self.solar_out_dir: Path = session_dir / "solar"
        if self.rank == 0 and self.cfg.test.save_raw:
            self.solar_out_dir.mkdir(parents=True, exist_ok=True)

        self.log: Path | None = None
        if cfg.test.save_results:
            # use last testing_info…txt if it exists, else create a fresh one
            existing = sorted(self.sdir.glob("test_log*.txt"),
                              key=os.path.getmtime)
            if existing:
                self.log = existing[-1]
            else:
                ts   = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
                self.log = self.sdir / f"test_log{ts}.txt"
                self.log.touch()

    def run(self):
        results, fp = {}, None
        raw_files_were_saved = False
        if self.rank == 0 and self.cfg.test.save_results:
            fp = self.log.open("a", buffering=1)     # line-buffered -> flush

        for tag, loader in self.loaders.items():
            if self.verbose:
                # Print a clean header for the upcoming test
                print("\n" + "="*20)
                print(f"Analyzing Solar Class: {tag}")
            res = IonoTester(self.m, loader, self.dev,
                             config=self.cfg, verbose=self.verbose).test()
            results[tag] = res
            # Update the progress bar with the latest results
            metrics = res["Metrics"]["overall"]
            if self.rank == 0:
                if self.cfg.test.save_results:
                    self._append(fp, tag, res["Metrics"])
                    fp.flush()   
                if self.cfg.test.save_raw and (self.cfg.mode == 'solar'):
                    self._dump_npz(tag, res)
                    raw_files_were_saved = True
            
        if fp is not None:
            fp.close()
            
        if self.verbose:
            print("") # New line after the progress bar
            if self.log:
                print(f"Solar analysis complete. Log appended to: {self.log}")
            if raw_files_were_saved:
                print(f"Raw solar data saved in: {self.solar_out_dir}")
        return results


    @staticmethod
    def _append(fp, tag: str, m: dict):
        """Append nicely formatted metrics of one class to an open file."""
        fp.write(f"\n{tag}:\nOverall Metrics:\n")
        pairs = [("RMSE(TECU)", "RMSE_std"),
                 ("R²",          "R²_std"),
                 ("SSIM",        "SSIM_std")]
        for mean_k, std_k in pairs:
            fp.write(f"  {mean_k}: {m['overall'][mean_k]:.4f} "
                     f"± {m['overall'][std_k]:.4f}\n")

        def _line(lbl, arr, std):
            fp.write(f"  {lbl:<17}: [{', '.join(f'{v:.4f}' for v in arr)}]\n")
            fp.write(f"  {lbl+'_std':<17}: "
                     f"[{', '.join(f'{v:.4f}' for v in std)}]\n")

        ph = m["perHorizon"]
        _line("RMSE per step", ph["RMSE"],  ph["RMSE_std"])
        _line("R² per step",   ph["R²"],    ph["R²_std"])
        _line("SSIM per step", ph["SSIM"],  ph["SSIM_std"])


    def _dump_npz(self, tag: str, res: dict):
        ts   = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        npz  = self.solar_out_dir / f"test_raw_{tag}_{ts}.npz"
        dates = self.raw["solarClasses"][self._MAP[tag]]        # class dates
        np.savez_compressed(
            npz,
            predictions=res["Predictions"],
            labels=res["Labels"],
            dates=np.asarray(dates, dtype="datetime64[s]"),
        )
        if self.verbose:
            print(f"Saved raw arrays → {npz}")      
#===============================================================================
            
# Storm Analysis
#===============================================================================
class StormAnalysis:
    """
    Performs a detailed analysis of the model's performance on specific
    geomagnetic storms, using pre-built DataLoaders.

    Example
    -------
    # from scripts.data import create_storm_loaders
    # storm_loaders = create_storm_loaders(cfg, raw_data)
    # sa = StormAnalysis(model, raw_data, cfg, storm_loaders, device="cuda:0")
    # res = sa.run()
    """
    def __init__(self,
                 model,
                 raw_dict: dict,
                 cfg: dict,
                 loaders: dict,  # <-- Takes pre-built loaders
                 device: str = "cpu",
                 verbose: bool = True):

        if is_dist_active():
            self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        else:
            self.rank, self.world_size = 0, 1

        self.m       = model
        self.raw     = raw_dict
        self.cfg     = cfg
        self.loaders = loaders  # <-- Stores the provided loaders
        self.dev     = device
        self.verbose = verbose and self.rank == 0

        # Output directory and log file setup (same as before)
        base = Path(cfg.paths.base_dir)
        session_dir = base / cfg.paths.out_dir / cfg.session.name
        
        # The main directory for log files
        self.sdir: Path = session_dir
        self.sdir.mkdir(parents=True, exist_ok=True)
        
        # The dedicated subdirectory for raw storm .npz files
        self.storm_out_dir: Path = session_dir / "storms"
        if self.rank == 0 and self.cfg.test.save_raw:
            self.storm_out_dir.mkdir(parents=True, exist_ok=True)

        self.log: Path | None = None
        if self.rank == 0 and cfg.test.save_results:
            # Find the most recent testing_info.txt file
            existing = sorted(self.sdir.glob("test_log_*.txt"), key=os.path.getmtime)
            
            if existing:
                # If a log file exists, use the most recent one
                self.log = existing[-1]
            else:
                # If no log file exists, create a new one with a timestamp
                ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.log = self.sdir / f"test_log_{ts}.txt"
                self.log.touch()
                if self.verbose:
                    print(f"No existing log file found. Created new one: {self.log}")

    def run(self):
        """Executes the storm-by-storm analysis using the pre-built loaders."""
        results, fp = {}, None
        raw_files_were_saved = False
        if self.rank == 0 and self.cfg.test.save_results:
            fp = self.log.open("a", buffering=1)
            self._append_header(fp)

        pbar = tqdm(self.loaders.items(),
                    disable=(not self.verbose),
                    desc="Analyzing Storms")

        for tag, storm_data in pbar:
            storm_meta = storm_data['meta']
            storm_idx = int(tag.split('_')[1])

            if self.verbose:
                pbar.set_postfix_str(f"Date: {storm_meta['Storm Date'].strftime('%Y-%m-%d')}, Dst: {storm_meta['Storm Dst']:.1f} nT")

            # Get loaders from the input dictionary
            full_loader = storm_data['full_loader']
            main_loader = storm_data['main_loader']

            # Run Inference using IonoTester (no changes here)
            tester_kwargs = {'config': self.cfg, 'verbose': False}
            full_res = IonoTester(self.m, full_loader, self.dev, **tester_kwargs).test()
            main_res = IonoTester(self.m, main_loader, self.dev, **tester_kwargs).test()

            # Consolidate and Store Results
            storm_result = {
                'Meta': storm_meta,
                'FullPeriod': full_res,
                'MainPhase': main_res
            }
            results[tag] = storm_result

            # Handle File I/O on rank 0
            if self.rank == 0:
                if self.cfg.test.save_results:
                    self._append_storm_row(fp, storm_idx + 1, storm_result)
                if self.cfg.test.save_raw and self.cfg.mode == 'storm':
                    self._dump_npz(storm_idx + 1, tag, storm_result)
                    raw_files_were_saved = True
        
        # Append detailed summary and close file
        if fp is not None:
            self._append_per_horizon_summary(fp, results)
            fp.close()
            
        if self.verbose:
            print("") # Start on a new line after the progress bar
            if self.log:
                print(f"Storm analysis complete. Log appended to: {self.log}")
            if raw_files_were_saved:
                print(f"Raw storm data saved in: {self.storm_out_dir}")
        
        return results

    def _dump_npz(self, storm_idx: int, tag: str, res: dict):
        """Saves raw predictions and labels for a single storm to a .npz file."""
        # Use the tag for a robust filename, replacing 'storm_X_' prefix
        filename_tag = tag.replace(f"storm_{storm_idx-1}_", "") 
        npz_path = self.storm_out_dir / f"test_raw_storm_{storm_idx}_{filename_tag}.npz"
        
        np.savez_compressed(
            npz_path,
            full_period_predictions=res['FullPeriod']['Predictions'],
            full_period_labels=res['FullPeriod']['Labels'],
            # main_phase_predictions=res['MainPhase']['Predictions'],
            # main_phase_labels=res['MainPhase']['Labels'],
            storm_metadata=res['Meta']
        )
        
    def _append_header(self, fp):
        fp.write("\n\n\nStorm Analysis Results\n")
        header = (
            f"{'='*150}\n"
            f"{'Storm':<7} {'Date':<18} {'Dst(nT)':<8} "
            f"{'Full Period (RMSE / R² / SSIM)':<45} "
            f"{'Main Phase (RMSE / R² / SSIM)'}\n"
            f"{'-'*150}\n"
        )
        fp.write(header)

    def _append_storm_row(self, fp, storm_idx: int, res: dict):
        meta = res['Meta']
        full_m = res['FullPeriod']['Metrics']['overall']
        main_m = res['MainPhase']['Metrics']['overall']
        
        def fmt(m):
            return (f"{m['RMSE(TECU)']:.2f} ± {m['RMSE_std']:.2f} / "
                    f"{m['R²']:.2f} ± {m['R²_std']:.2f} / "
                    f"{m['SSIM']:.2f} ± {m['SSIM_std']:.2f}")

        log_line = (
            f"{storm_idx:<7} {meta['Storm Date'].strftime('%Y-%m-%d %H:%M'):<18} {meta['Storm Dst']:<8.1f} "
            f"{fmt(full_m):<45} "
            f"{fmt(main_m)}\n"
        )
        fp.write(log_line)

    def _append_per_horizon_summary(self, fp, all_results: dict):
        fp.write("\n\n======================== Per-Horizon Breakdown ========================\n")

        def _format_block(ph_dict, label):
            lines = [f"\n  ── {label} Per-Horizon Metrics ──"]
            for key in ['RMSE', 'R²', 'SSIM']:
                lines.append(f"  {key:<9}: [" + ", ".join(f"{v:.4f}" for v in ph_dict[key]) + "]")
                lines.append(f"  {key+'_std':<9}: [" + ", ".join(f"{v:.4f}" for v in ph_dict[key+'_std']) + "]")
            return "\n".join(lines)

        for tag, storm_res in all_results.items():
            storm_idx = int(tag.split('_')[1])
            meta = storm_res['Meta']
            fp.write(f"\n\nStorm {storm_idx+1}: {meta['Storm Date'].strftime('%Y-%m-%d')} (Dst: {meta['Storm Dst']:.1f} nT)")
            fp.write(_format_block(storm_res['FullPeriod']['Metrics']['perHorizon'], "Full Period"))
            fp.write(_format_block(storm_res['MainPhase']['Metrics']['perHorizon'], "Main Phase"))
        
        fp.write("\n" + "="*80 + "\n")
#===============================================================================      

