#!/usr/bin/env python
"""
Enhanced TensorBoard Launcher Script

This script launches TensorBoard to visualize and compare model training logs.
Usage: python launch_tensorboard.py [--logdir LOGDIR] [--port PORT]
"""

import argparse
import os
import subprocess
import webbrowser
from pathlib import Path
import time
import sys

def main():
    parser = argparse.ArgumentParser(description='Launch TensorBoard to compare experiments and analyze models')
    parser.add_argument('--logdir', default='logs/experiments', help='Directory containing the logs (default: logs/experiments)')
    parser.add_argument('--port', default='6006', help='Port for TensorBoard (default: 6006)')
    parser.add_argument('--show-all', action='store_true', help='Show all logs, including old TensorBoard logs')
    args = parser.parse_args()
    
    # Check if TensorBoard is installed
    try:
        import tensorboard
        print(f"Using TensorBoard version {tensorboard.__version__}")
    except ImportError:
        print("Error: TensorBoard not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])
    
    # Determine log directory
    if args.show_all:
        # Use the root logs dir which should contain both experiments and fit subdirs
        log_dir = Path("logs")
    else:
        log_dir = Path(args.logdir)
    
    # Ensure log directory exists
    if not log_dir.exists():
        print(f"Log directory '{log_dir}' doesn't exist. Creating it...")
        os.makedirs(log_dir, exist_ok=True)
    
    # Check for experiment logs
    experiment_dirs = []
    if log_dir.name == 'experiments':
        experiment_dirs = list(log_dir.glob('*/*'))  # model_name/timestamp
    else:
        experiment_dirs = list(log_dir.glob('experiments/*/*'))  # experiments/model_name/timestamp
    
    # Also count old style logs if we're showing all
    old_logs = []
    if args.show_all:
        old_logs = list(Path('logs/fit').glob('*'))
    
    # Calculate total number of log directories
    total_log_count = len(experiment_dirs) + len(old_logs)
    
    if total_log_count == 0:
        print(f"No log files found in {log_dir}. Have you trained a model yet?")
        print("Run the training script first to generate logs.")
        return
    
    print(f"Found {total_log_count} log directories")
    if experiment_dirs:
        print(f"  - {len(experiment_dirs)} experiment directories")
    if old_logs:
        print(f"  - {len(old_logs)} old-style log directories")
    
    print(f"Launching TensorBoard with log directory: {log_dir}")
    
    # Launch TensorBoard with enhanced settings
    try:
        cmd = [
            'tensorboard',
            '--logdir', str(log_dir),
            '--port', args.port,
            '--bind_all'  # Allow access from other machines on the network
        ]
        process = subprocess.Popen(cmd)
        
        # Open browser after a short delay
        time.sleep(2)
        url = f"http://localhost:{args.port}"
        webbrowser.open(url)
        
        print(f"TensorBoard is running at {url}")
        print("\nAvailable dashboards:")
        print("  • SCALARS - Compare training metrics across runs")
        print("  • HPARAMS - Compare hyperparameters and find optimal settings")
        print("  • GRAPHS - View model architectures")
        print("  • TIME SERIES - Examine metrics over time")
        print("\nTips for comparison:")
        print("  • Use the HPARAMS dashboard to filter models by hyperparameters")
        print("  • In SCALARS, click 'Show data download links' to export CSV data")
        print("  • Use 'regex' in the left sidebar to filter specific models")
        print("\nPress Ctrl+C to exit")
        
        # Keep the script running until interrupted
        process.wait()
    except KeyboardInterrupt:
        print("Shutting down TensorBoard...")
        process.terminate()
    except FileNotFoundError:
        print("Error: TensorBoard executable not found.")
        print("Make sure TensorFlow is installed correctly:")
        print("    pip install tensorflow")
        print("Or try installing just TensorBoard:")
        print("    pip install tensorboard")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()