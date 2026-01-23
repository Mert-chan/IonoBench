#####################################################################
"""
cli.py
@ Mert-chan 
@ 23 January 2026 (Last Modified) 
IonoBench Command-Line Interface
"""
#####################################################################

import argparse
import sys
from pathlib import Path
import torch

from scripts.loadConfigs import load_configs
from scripts.data import load_training_data, prepare_raw, make_default_loaders, make_solar_loaders, make_storm_loaders
from scripts.registry import build_model
from source.myDataFuns import download_dataset, download_model_folder
from source.myTrainFuns import IonoTester, SolarAnalysis, StormAnalysis, DDPtoSingleGPU


def get_parser():
    """Create argument parser with all subcommands"""
    parser = argparse.ArgumentParser(
        description="IonoBench: Ionospheric TEC Prediction & Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data --type stratified
  %(prog)s model --name SimVPv2
  %(prog)s test --model SimVPv2 --checkpoint path/to/checkpoint.pth --session-name test_run
  %(prog)s solar --model SimVPv2 --checkpoint path/to/checkpoint.pth --session-name test_run
  %(prog)s storm --model SimVPv2 --checkpoint path/to/checkpoint.pth --session-name test_run --save-raw
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ===== Data Download =====
    data_cmd = subparsers.add_parser("data", help="Download IonoBench dataset")
    data_cmd.add_argument("--type", choices=["stratified", "chronological"], default="stratified",
                         help="Dataset split type (default: stratified)")
    
    # ===== Model Download =====
    model_cmd = subparsers.add_parser("model", help="Download pre-trained model")
    model_cmd.add_argument("--name", choices=["DCNN", "SimVPv2", "SwinLSTM"], required=True,
                          help="Model name to download")
    
    # ===== Test =====
    test_cmd = subparsers.add_parser("test", help="Test pre-trained model on test set")
    test_cmd.add_argument("--model", default="SimVPv2", choices=["SimVPv2", "DCNN121", "SwinLSTM"],
                         help="Model architecture (default: SimVPv2)")
    test_cmd.add_argument("--checkpoint", required=True, type=str,
                         help="Path to checkpoint file")
    test_cmd.add_argument("--split", choices=["stratified", "chronological"], default="stratified",
                         help="Data split (default: stratified)")
    test_cmd.add_argument("--session-name", required=True, type=str,
                         help="Session name for results directory")
    test_cmd.add_argument("--batch-size", type=int, default=32,
                         help="Batch size (default: 32)")
    test_cmd.add_argument("--save-raw", action="store_true",
                         help="Save raw predictions to npz")
    
    # ===== Solar Analysis =====
    solar_cmd = subparsers.add_parser("solar", help="Solar intensity analysis")
    solar_cmd.add_argument("--model", default="SimVPv2", choices=["SimVPv2", "DCNN121", "SwinLSTM"],
                          help="Model architecture (default: SimVPv2)")
    solar_cmd.add_argument("--checkpoint", required=True, type=str,
                          help="Path to checkpoint file")
    solar_cmd.add_argument("--split", choices=["stratified", "chronological"], default="stratified",
                          help="Data split (default: stratified)")
    solar_cmd.add_argument("--session-name", required=True, type=str,
                          help="Session name for results directory")
    solar_cmd.add_argument("--save-raw", action="store_true",
                          help="Save raw predictions per solar class")
    
    # ===== Storm Analysis =====
    storm_cmd = subparsers.add_parser("storm", help="Storm event analysis")
    storm_cmd.add_argument("--model", default="SimVPv2", choices=["SimVPv2", "DCNN121", "SwinLSTM"],
                          help="Model architecture (default: SimVPv2)")
    storm_cmd.add_argument("--checkpoint", required=True, type=str,
                          help="Path to checkpoint file")
    storm_cmd.add_argument("--split", choices=["stratified", "chronological"], default="stratified",
                          help="Data split (default: stratified)")
    storm_cmd.add_argument("--session-name", required=True, type=str,
                          help="Session name for results directory")
    storm_cmd.add_argument("--save-raw", action="store_true",
                          help="Save raw predictions per storm event")
    
    return parser


def handle_data(args, base_path):
    """Download dataset"""
    print(f"Downloading {args.type} split dataset...")
    download_dataset(dataset_name=args.type, base_path=base_path)
    print(f"✓ Dataset downloaded to {base_path}/datasets/")


def handle_model(args, base_path):
    """Download pre-trained model"""
    print(f"Downloading {args.name} model...")
    download_model_folder(model_name=args.name, base_path=base_path)
    print(f"✓ Model downloaded to {base_path}/training_sessions/{args.name}/")


def handle_test(args, base_path, device):
    """Test pre-trained model on test set"""
    print(f"\n{'='*60}")
    print(f"Testing {args.model} on {args.split} split")
    print(f"{'='*60}\n")
    
    # Load configs
    cfgs = load_configs(model=args.model, mode="test", split=args.split, base_path=base_path)
    cfgs.test.batch_size = args.batch_size
    cfgs.test.save_results = True
    cfgs.test.save_raw = args.save_raw
    cfgs.session.name = args.session_name
    cfgs.paths.base_dir = base_path
    
    # Prepare data
    data = prepare_raw(cfgs)
    cfgs.test.input_names = data['OMNI_names']
    
    # Build model
    cfgs.model.input_shape = (cfgs.data.seq_len, cfgs.data.num_omni + 1, cfgs.data.H, cfgs.data.W)
    model = build_model(cfg=cfgs, base_path=base_path, device=device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, weights_only=True)
    model.load_state_dict(DDPtoSingleGPU(checkpoint["model_state_dict"]))
    print(f"Loaded checkpoint: {args.checkpoint}\n")
    
    # Create loaders and test
    loaders = make_default_loaders(cfg=cfgs, d=data)
    testDict = IonoTester(model, loaders['test'], device=device, config=cfgs).test()
    
    print(f"\nTest complete. Results saved to training_sessions/{args.session_name}/")


def handle_solar(args, base_path, device):
    """Solar intensity analysis"""
    print(f"\n{'='*60}")
    print(f"Solar Analysis: {args.model} on {args.split} split")
    print(f"{'='*60}\n")
    
    # Load configs
    cfgs = load_configs(model=args.model, mode="solar", split=args.split, base_path=base_path)
    cfgs.test.save_results = True
    cfgs.test.save_raw = args.save_raw
    cfgs.session.name = args.session_name
    cfgs.paths.base_dir = base_path
    
    # Prepare data
    data = prepare_raw(cfgs)
    cfgs.test.input_names = data['OMNI_names']
    
    # Build model
    cfgs.model.input_shape = (cfgs.data.seq_len, cfgs.data.num_omni + 1, cfgs.data.H, cfgs.data.W)
    model = build_model(cfg=cfgs, base_path=base_path, device=device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, weights_only=True)
    model.load_state_dict(DDPtoSingleGPU(checkpoint["model_state_dict"]))
    print(f"Loaded checkpoint: {args.checkpoint}\n")
    
    # Create loaders and analyze
    loaders = make_solar_loaders(cfg=cfgs, base_path=base_path, d=data)
    solarDict = SolarAnalysis(model, data, loaders, device=device, cfg=cfgs).run()
    
    print(f"\nSolar analysis complete. Results saved to training_sessions/{args.session_name}/")


def handle_storm(args, base_path, device):
    """Storm event analysis"""
    print(f"\n{'='*60}")
    print(f"Storm Analysis: {args.model} on {args.split} split")
    print(f"{'='*60}\n")
    
    # Load configs
    cfgs = load_configs(model=args.model, mode="storm", split=args.split, base_path=base_path)
    cfgs.test.save_results = True
    cfgs.test.save_raw = args.save_raw
    cfgs.session.name = args.session_name
    cfgs.paths.base_dir = base_path
    
    # Prepare data
    data = prepare_raw(cfgs)
    cfgs.test.input_names = data['OMNI_names']
    
    # Build model
    cfgs.model.input_shape = (cfgs.data.seq_len, cfgs.data.num_omni + 1, cfgs.data.H, cfgs.data.W)
    model = build_model(cfg=cfgs, base_path=base_path, device=device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, weights_only=True)
    model.load_state_dict(DDPtoSingleGPU(checkpoint["model_state_dict"]))
    print(f"Loaded checkpoint: {args.checkpoint}\n")
    
    # Create loaders and analyze
    loaders = make_storm_loaders(cfg=cfgs, d=data)
    testDict = StormAnalysis(model, data, cfgs, loaders, device).run()
    
    print(f"\nStorm analysis complete. Results saved to training_sessions/{args.session_name}/")


def main():
    """Entry point"""
    parser = get_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    base_path = Path(__file__).parent.parent
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}\n")
    
    if args.command == "data":
        handle_data(args, base_path)
    elif args.command == "model":
        handle_model(args, base_path)
    elif args.command == "test":
        handle_test(args, base_path, device)
    elif args.command == "solar":
        handle_solar(args, base_path, device)
    elif args.command == "storm":
        handle_storm(args, base_path, device)


if __name__ == "__main__":
    main()
