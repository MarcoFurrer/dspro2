#!/usr/bin/env python
"""
TensorBoard Launcher Script

This script helps launch TensorBoard to visualize model training logs.
Usage: python launch_tensorboard.py [--logdir LOGDIR]
"""

import argparse
import os
import subprocess
import webbrowser
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Launch TensorBoard to view training logs')
    parser.add_argument('--logdir', default='logs/fit', help='Directory containing the logs (default: logs/fit)')
    args = parser.parse_args()
    
    # Ensure log directory exists
    log_dir = Path(args.logdir)
    if not log_dir.exists():
        print(f"Log directory '{log_dir}' doesn't exist. Creating it...")
        os.makedirs(log_dir, exist_ok=True)
    
    # Check if there are any logsj
    log_files = list(log_dir.glob('*'))
    if not log_files:
        print(f"No log files found in {log_dir}. Have you trained a model yet?")
        print("Run the training notebook first to generate logs.")
        return
    
    print(f"Found {len(log_files)} log directories/files")
    print(f"Launching TensorBoard with log directory: {log_dir}")
    
    # Launch TensorBoard
    try:
        process = subprocess.Popen(['tensorboard', '--logdir', str(log_dir)])
        
        # Open browser after a short delay
        import time
        time.sleep(2)
        webbrowser.open('http://localhost:6006')
        
        print("TensorBoard is running at http://localhost:6006")
        print("Press Ctrl+C to exit")
        
        # Keep the script running until interrupted
        process.wait()
    except KeyboardInterrupt:
        print("Shutting down TensorBoard...")
        process.terminate()
    except FileNotFoundError:
        print("Error: TensorBoard not found. Is TensorFlow installed?")
        print("Install with: pip install tensorflow")

if __name__ == "__main__":
    main()