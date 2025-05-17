#!/usr/bin/env python
"""
Model Comparison Utility

This script generates comparison tables and visualizations of trained models
using the logged metrics and configurations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json
from datetime import datetime
from pathlib import Path

def load_model_results():
    """Load all model results from the CSV file"""
    results_file = "logs/all_model_results.csv"
    
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found.")
        print("Have you trained any models yet?")
        return None
    
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} model results")
    return df

def find_best_models(df, metric='correlation', top_n=5):
    """Find the best models based on a specific metric"""
    if df is None or len(df) == 0:
        return None
    
    if metric not in df.columns:
        print(f"Metric '{metric}' not found in results. Available metrics: {df.columns.tolist()}")
        return None
    
    # For most metrics, smaller is better (loss, mae, mse)
    # For correlation, larger is better
    ascending = not (metric in ['correlation', 'era_correlation'])
    
    # Drop rows with NaN values for the chosen metric
    valid_df = df.dropna(subset=[metric])
    
    if len(valid_df) == 0:
        print(f"No models have valid values for metric '{metric}'")
        return None
    
    # Sort and get top N
    top_models = valid_df.sort_values(by=metric, ascending=ascending).head(top_n)
    
    return top_models

def plot_metric_comparison(df, metric='correlation', figsize=(12, 6)):
    """Plot a comparison of models based on a given metric"""
    if df is None or len(df) == 0:
        return
    
    if metric not in df.columns:
        print(f"Metric '{metric}' not found in results. Available metrics: {df.columns.tolist()}")
        return
    
    # Filter out NaN values for the chosen metric
    valid_df = df.dropna(subset=[metric])
    
    if len(valid_df) == 0:
        print(f"No models have valid values for metric '{metric}'")
        return
    
    # Sort by metric
    ascending = not (metric in ['correlation', 'era_correlation'])
    plot_df = valid_df.sort_values(by=metric, ascending=not ascending)
    
    # Create model labels combining model_name and feature_subset
    if 'feature_subset' in plot_df.columns:
        plot_df['model_label'] = plot_df.apply(
            lambda row: f"{row['model_name']}_{row.get('feature_subset', 'unknown')}", 
            axis=1
        )
    else:
        plot_df['model_label'] = plot_df['model_name']
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Use different colors for different feature subsets if available
    if 'feature_subset' in plot_df.columns:
        feature_colors = {'small': 'lightblue', 'medium': 'orange', 'all': 'green', 'unknown': 'gray'}
        bar_colors = [feature_colors.get(fs, 'gray') for fs in plot_df['feature_subset']]
        
        # Create a bar chart
        bars = plt.bar(plot_df['model_label'], plot_df[metric], color=bar_colors)
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label=subset) 
            for subset, color in feature_colors.items()
            if subset in plot_df['feature_subset'].values
        ]
        plt.legend(handles=legend_elements, title='Feature Subset')
    else:
        # Simple bar chart
        plt.bar(plot_df['model_label'], plot_df[metric])
    
    # Set title and labels
    plt.title(f"Model Comparison by {metric}")
    plt.xlabel("Model")
    plt.ylabel(metric)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save and show
    os.makedirs('analysis', exist_ok=True)
    plt.savefig(f"analysis/model_comparison_{metric}.png")
    plt.show()

def analyze_hyperparameter_impact(df, hyperparam, metric='correlation'):
    """Analyze the impact of a hyperparameter on a specific metric"""
    if df is None or len(df) == 0:
        return
    
    if hyperparam not in df.columns:
        print(f"Hyperparameter '{hyperparam}' not found in results.")
        return
    
    if metric not in df.columns:
        print(f"Metric '{metric}' not found in results.")
        return
    
    # Filter out NaN values
    valid_df = df.dropna(subset=[hyperparam, metric])
    
    if len(valid_df) == 0:
        print(f"No models have valid values for both '{hyperparam}' and '{metric}'")
        return
    
    # Convert to numeric if possible
    try:
        valid_df[hyperparam] = pd.to_numeric(valid_df[hyperparam])
        numeric = True
    except:
        numeric = False
    
    # Group by the hyperparameter and compute mean of the metric
    if numeric:
        # For numeric hyperparameters, create a scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=valid_df, x=hyperparam, y=metric, hue='model_name')
        plt.title(f"Impact of {hyperparam} on {metric}")
        plt.tight_layout()
        plt.savefig(f"analysis/hyperparam_impact_{hyperparam}_{metric}.png")
        plt.show()
        
        # Also show correlation
        correlation = valid_df[[hyperparam, metric]].corr().iloc[0, 1]
        print(f"Correlation between {hyperparam} and {metric}: {correlation:.4f}")
    else:
        # For categorical hyperparameters, create a box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=valid_df, x=hyperparam, y=metric)
        plt.title(f"Impact of {hyperparam} on {metric}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"analysis/hyperparam_impact_{hyperparam}_{metric}.png")
        plt.show()
        
        # Show average metric by hyperparameter value
        print(f"Average {metric} by {hyperparam} value:")
        print(valid_df.groupby(hyperparam)[metric].mean().sort_values(ascending=False))

def compare_model_architectures(df, metric='correlation'):
    """Compare different model architectures"""
    if df is None or len(df) == 0:
        return
    
    if 'model_name' not in df.columns or metric not in df.columns:
        print(f"Required columns 'model_name' or '{metric}' not found in results.")
        return
    
    # Filter out NaN values
    valid_df = df.dropna(subset=[metric])
    
    if len(valid_df) == 0:
        print(f"No models have valid values for '{metric}'")
        return
    
    # Group by model_name and compute statistics
    model_stats = valid_df.groupby('model_name').agg({
        metric: ['mean', 'std', 'min', 'max', 'count']
    }).reset_index()
    
    # Flatten the multi-level columns
    model_stats.columns = ['model_name', f'mean_{metric}', f'std_{metric}', 
                          f'min_{metric}', f'max_{metric}', 'count']
    
    # Sort by mean metric
    ascending = not (metric in ['correlation', 'era_correlation'])
    model_stats = model_stats.sort_values(by=f'mean_{metric}', ascending=ascending)
    
    # Print the results
    print("\nModel Architecture Comparison:")
    print("==============================")
    print(model_stats)
    
    # Create a bar plot with error bars
    plt.figure(figsize=(12, 6))
    
    # Create a bar chart
    ax = plt.subplot(111)
    
    # Plot bars
    bars = ax.bar(model_stats['model_name'], model_stats[f'mean_{metric}'], 
           yerr=model_stats[f'std_{metric}'], 
           capsize=5, 
           alpha=0.7)
    
    # Add count as text on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = model_stats['count'].iloc[i]
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'n={count}',
                ha='center', va='bottom', rotation=0, size=8)
    
    # Set title and labels
    plt.title(f"Model Architecture Comparison by {metric}")
    plt.xlabel("Model Architecture")
    plt.ylabel(f"Mean {metric}")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save and show
    plt.savefig(f"analysis/architecture_comparison_{metric}.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare trained models and analyze results')
    
    parser.add_argument('--metric', default='correlation', 
                        help='Metric to use for comparison (default: correlation)')
    
    parser.add_argument('--top', type=int, default=5, 
                        help='Number of top models to show (default: 5)')
    
    parser.add_argument('--plot', action='store_true', 
                        help='Generate visualization plots')
    
    parser.add_argument('--analyze', type=str, default=None,
                        help='Analyze impact of a specific hyperparameter (e.g. learning_rate)')
    
    args = parser.parse_args()
    
    # Load all model results
    results_df = load_model_results()
    
    if results_df is None or len(results_df) == 0:
        print("No model results found.")
        return
    
    # Print summary of available models
    print("\nSummary of available models:")
    print(f"Total models: {len(results_df)}")
    
    if 'model_name' in results_df.columns:
        model_counts = results_df['model_name'].value_counts()
        print("\nModel types:")
        for model, count in model_counts.items():
            print(f"  - {model}: {count}")
    
    # Find and display the best models
    print(f"\nTop {args.top} models by {args.metric}:")
    top_models = find_best_models(results_df, metric=args.metric, top_n=args.top)
    
    if top_models is not None:
        # Select relevant columns for display
        display_columns = ['model_name', args.metric]
        
        # Add other informative columns if they exist
        for col in ['timestamp', 'feature_subset', 'optimizer', 'learning_rate', 'batch_size', 'loss_fn']:
            if col in top_models.columns:
                display_columns.append(col)
        
        print(top_models[display_columns].to_string(index=False))
    
    # Compare model architectures
    compare_model_architectures(results_df, metric=args.metric)
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating comparison plots...")
        plot_metric_comparison(results_df, metric=args.metric)
    
    # Analyze hyperparameter impact if requested
    if args.analyze:
        print(f"\nAnalyzing impact of '{args.analyze}' on {args.metric}...")
        analyze_hyperparameter_impact(results_df, args.analyze, args.metric)

if __name__ == "__main__":
    main()
