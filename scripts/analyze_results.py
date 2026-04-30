"""
FYP Results Analysis Script
Run after all experiments are complete to generate comparison tables and figures.

Usage: python analyze_results.py --results-dir D:\WoundCare\experiments

This will:
1. Load all experiment summaries
2. Create comparison tables (CSV + formatted text)
3. Generate comparison figures
4. Create a LaTeX-ready table (optional)
"""

import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def get_args():
    p = argparse.ArgumentParser(description="Analyze FYP experiment results")
    p.add_argument("--results-dir", required=True, help="Directory containing all experiment folders")
    p.add_argument("--output-dir", default=None, help="Output directory (default: results-dir/analysis)")
    return p.parse_args()


def load_experiment_results(results_dir):
    """Load all experiment summaries"""
    results_dir = Path(results_dir)
    
    classification_results = []
    segmentation_results = []
    
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        summary_path = exp_dir / "summary.json"
        if not summary_path.exists():
            print(f"Warning: No summary.json in {exp_dir.name}")
            continue
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        summary['exp_dir'] = str(exp_dir)
        
        # Determine if classification or segmentation based on metrics
        if 'best_accuracy' in summary:
            classification_results.append(summary)
        elif 'best_dice' in summary:
            segmentation_results.append(summary)
    
    return classification_results, segmentation_results


def create_classification_comparison_table(results, output_dir):
    """Create comparison table for classification experiments"""
    if not results:
        print("No classification results found")
        return
    
    # Sort by experiment name
    results = sorted(results, key=lambda x: x['experiment'])
    
    # Create table data
    headers = ['Experiment', 'Backbone', 'Img Size', 'LR', 'MixUp', 'Best Acc (%)', 'ECE', 'Epochs', 'Time (min)']
    rows = []
    
    for r in results:
        config = r.get('config', {})
        rows.append([
            r['experiment'],
            r.get('model', 'N/A'),
            config.get('img_size', 'N/A'),
            config.get('lr', 'N/A'),
            config.get('mixup_alpha', 0),
            f"{r['best_accuracy']*100:.2f}",
            f"{r.get('ece', 'N/A'):.4f}" if isinstance(r.get('ece'), float) else 'N/A',
            r['epochs_trained'],
            f"{r['training_time_min']:.1f}"
        ])
    
    # Print formatted table
    print("\n" + "="*100)
    print("CLASSIFICATION EXPERIMENTS COMPARISON")
    print("="*100)
    
    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]
    
    # Print header
    header_line = "|".join(str(headers[i]).center(col_widths[i]) for i in range(len(headers)))
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows
    for row in rows:
        row_line = "|".join(str(row[i]).center(col_widths[i]) for i in range(len(row)))
        print(row_line)
    
    print("="*100)
    
    # Save to CSV
    csv_path = output_dir / "classification_comparison.csv"
    with open(csv_path, 'w') as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"Saved: {csv_path}")
    
    # Find best experiment
    best_idx = np.argmax([r['best_accuracy'] for r in results])
    print(f"\n Best Classification: {results[best_idx]['experiment']} with {results[best_idx]['best_accuracy']*100:.2f}% accuracy")
    
    return results


def create_segmentation_comparison_table(results, output_dir):
    """Create comparison table for segmentation experiments"""
    if not results:
        print("No segmentation results found")
        return
    
    results = sorted(results, key=lambda x: x['experiment'])
    
    headers = ['Experiment', 'Img Size', 'LR', 'Batch', 'Best Dice', 'Best IoU', 'Epochs', 'Time (min)']
    rows = []
    
    for r in results:
        config = r.get('config', {})
        rows.append([
            r['experiment'],
            config.get('img_size', 'N/A'),
            config.get('lr', 'N/A'),
            config.get('batch', 'N/A'),
            f"{r['best_dice']:.4f}",
            f"{r.get('best_iou', 'N/A'):.4f}" if isinstance(r.get('best_iou'), float) else 'N/A',
            r['epochs_trained'],
            f"{r['training_time_min']:.1f}"
        ])
    
    print("\n" + "="*100)
    print("SEGMENTATION EXPERIMENTS COMPARISON")
    print("="*100)
    
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]
    
    header_line = "|".join(str(headers[i]).center(col_widths[i]) for i in range(len(headers)))
    print(header_line)
    print("-" * len(header_line))
    
    for row in rows:
        row_line = "|".join(str(row[i]).center(col_widths[i]) for i in range(len(row)))
        print(row_line)
    
    print("="*100)
    
    csv_path = output_dir / "segmentation_comparison.csv"
    with open(csv_path, 'w') as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"Saved: {csv_path}")
    
    best_idx = np.argmax([r['best_dice'] for r in results])
    print(f"\n Best Segmentation: {results[best_idx]['experiment']} with Dice={results[best_idx]['best_dice']:.4f}")
    
    return results


def plot_classification_comparison(results, output_dir):
    """Create bar chart comparing classification experiments"""
    if not results:
        return
    
    results = sorted(results, key=lambda x: x['experiment'])
    
    names = [r['experiment'] for r in results]
    accuracies = [r['best_accuracy'] * 100 for r in results]
    eces = [r.get('ece', 0) for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy bar chart
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))
    bars = axes[0].bar(names, accuracies, color=colors, edgecolor='black')
    axes[0].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[0].set_xlabel('Experiment', fontsize=12)
    axes[0].set_title('Classification Accuracy Comparison', fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Highlight best
    best_idx = np.argmax(accuracies)
    bars[best_idx].set_color('green')
    bars[best_idx].set_edgecolor('darkgreen')
    bars[best_idx].set_linewidth(2)
    
    # ECE bar chart
    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(names)))
    bars2 = axes[1].bar(names, eces, color=colors, edgecolor='black')
    axes[1].set_ylabel('Expected Calibration Error (ECE)', fontsize=12)
    axes[1].set_xlabel('Experiment', fontsize=12)
    axes[1].set_title('Model Calibration Comparison (Lower is Better)', fontsize=14)
    axes[1].tick_params(axis='x', rotation=45)
    
    for bar, ece in zip(bars2, eces):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f'{ece:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Highlight best (lowest ECE)
    best_ece_idx = np.argmin(eces)
    bars2[best_ece_idx].set_color('green')
    
    plt.tight_layout()
    save_path = output_dir / "classification_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_segmentation_comparison(results, output_dir):
    """Create bar chart comparing segmentation experiments"""
    if not results:
        return
    
    results = sorted(results, key=lambda x: x['experiment'])
    
    names = [r['experiment'] for r in results]
    dices = [r['best_dice'] for r in results]
    ious = [r.get('best_iou', 0) for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, dices, width, label='Dice Score', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, ious, width, label='IoU Score', color='darkorange', edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_title('Segmentation Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, dices):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, ious):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Highlight best Dice
    best_idx = np.argmax(dices)
    bars1[best_idx].set_edgecolor('green')
    bars1[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    save_path = output_dir / "segmentation_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_combined_training_curves(results, output_dir, task='classification'):
    """Plot training curves from all experiments on same graph"""
    if not results:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, r in enumerate(sorted(results, key=lambda x: x['experiment'])):
        exp_dir = Path(r['exp_dir'])
        log_path = exp_dir / "training_log.csv"
        
        if not log_path.exists():
            continue
        
        # Read CSV
        epochs, train_loss, val_loss, metric = [], [], [], []
        with open(log_path, 'r') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(',')
                epochs.append(int(parts[0]))
                train_loss.append(float(parts[1]))
                
                if task == 'classification':
                    val_loss.append(float(parts[2]))
                    metric.append(float(parts[3]) * 100)  # accuracy
                else:
                    val_loss.append(float(parts[3]))
                    metric.append(float(parts[4]))  # dice
        
        label = r['experiment']
        axes[0].plot(epochs, val_loss, color=colors[i], label=label, linewidth=1.5)
        axes[1].plot(epochs, metric, color=colors[i], label=label, linewidth=1.5)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('Validation Loss Comparison')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(alpha=0.3)
    
    if task == 'classification':
        axes[1].set_ylabel('Validation Accuracy (%)')
        axes[1].set_title('Validation Accuracy Comparison')
    else:
        axes[1].set_ylabel('Validation Dice Score')
        axes[1].set_title('Validation Dice Comparison')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(loc='lower right', fontsize=8)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / f"{task}_curves_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_latex_table(cls_results, seg_results, output_dir):
    """Generate LaTeX-formatted tables for report"""
    latex_content = []
    
    # Classification table
    if cls_results:
        latex_content.append("% Classification Results Table")
        latex_content.append("\\begin{table}[h]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Classification Experiment Results}")
        latex_content.append("\\begin{tabular}{lcccccc}")
        latex_content.append("\\hline")
        latex_content.append("Experiment & Backbone & Img Size & LR & Accuracy (\\%) & ECE \\\\")
        latex_content.append("\\hline")
        
        for r in sorted(cls_results, key=lambda x: x['experiment']):
            config = r.get('config', {})
            latex_content.append(f"{r['experiment']} & {r.get('model', 'N/A')} & {config.get('img_size', 'N/A')} & "
                                f"{config.get('lr', 'N/A')} & {r['best_accuracy']*100:.2f} & {r.get('ece', 0):.4f} \\\\")
        
        latex_content.append("\\hline")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        latex_content.append("")
    
    # Segmentation table
    if seg_results:
        latex_content.append("% Segmentation Results Table")
        latex_content.append("\\begin{table}[h]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Segmentation Experiment Results}")
        latex_content.append("\\begin{tabular}{lccccc}")
        latex_content.append("\\hline")
        latex_content.append("Experiment & Img Size & LR & Batch & Dice & IoU \\\\")
        latex_content.append("\\hline")
        
        for r in sorted(seg_results, key=lambda x: x['experiment']):
            config = r.get('config', {})
            latex_content.append(f"{r['experiment']} & {config.get('img_size', 'N/A')} & {config.get('lr', 'N/A')} & "
                                f"{config.get('batch', 'N/A')} & {r['best_dice']:.4f} & {r.get('best_iou', 0):.4f} \\\\")
        
        latex_content.append("\\hline")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
    
    # Save LaTeX
    latex_path = output_dir / "results_tables.tex"
    with open(latex_path, 'w') as f:
        f.write("\n".join(latex_content))
    print(f"Saved: {latex_path}")


def main():
    args = get_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("FYP RESULTS ANALYSIS")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print("="*70)
    
    # Load results
    cls_results, seg_results = load_experiment_results(results_dir)
    print(f"\nFound {len(cls_results)} classification experiments")
    print(f"Found {len(seg_results)} segmentation experiments")
    
    # Generate comparison tables
    create_classification_comparison_table(cls_results, output_dir)
    create_segmentation_comparison_table(seg_results, output_dir)
    
    # Generate comparison figures
    print("\nGenerating comparison figures...")
    plot_classification_comparison(cls_results, output_dir)
    plot_segmentation_comparison(seg_results, output_dir)
    
    # Plot combined training curves
    plot_combined_training_curves(cls_results, output_dir, 'classification')
    plot_combined_training_curves(seg_results, output_dir, 'segmentation')
    
    # Generate LaTeX tables
    generate_latex_table(cls_results, seg_results, output_dir)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print(f"All outputs saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
