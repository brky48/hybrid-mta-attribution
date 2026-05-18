#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Comparison Report
========================

This module aggregates the outputs of all attribution models and
generates the comparative performance analysis reported in Chapter 4
of the thesis. The eight attribution models are evaluated against the
ground truth distribution using multiple complementary metrics:
Mean Absolute Error, Root Mean Squared Error, Spearman rank
correlation, and ranking accuracy.

Implementation features:
    1. The hybrid model employs the optimal alpha value identified
       through grid search (read from optimal_alpha.csv), rather than
       a default fixed value.
    2. Absolute-value normalization protects against negative removal
       effects produced by Laplace smoothing artifacts.

Models compared:
    - Last-Click           (classical baseline)
    - First-Click          (classical baseline)
    - Linear               (classical baseline)
    - Time-Decay           (classical baseline)
    - Position-Based       (classical baseline)
    - Markov Chain         (proposed - removal effect)
    - Shapley Value        (proposed - fractional credit)
    - Hybrid (Markov+SHAP) (proposed - optimal alpha)

Part of the thesis:
    "A Hybrid Markov Chain and Shapley Value Approach to Multi-Touch
    Attribution and Budget Optimization in Digital Marketing"
    Istanbul University, Management Information Systems, 2026.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os


# =============================================================================
# 1. LOAD ALL MODEL OUTPUTS
# =============================================================================

def load_all_results() -> tuple:
    """Aggregate ground truth and all model attribution outputs into
    a single comparison DataFrame.

    Returns:
        tuple: (merged_dataframe, optimal_alpha_value)
    """

    # Ground Truth
    gt = pd.read_csv('ground_truth.csv')
    gt = gt.rename(columns={'ground_truth_weight': 'Ground_Truth'})

    # Markov (absolute value + normalization)
    markov = pd.read_csv('markov_results.csv')
    markov['markov_abs'] = markov['markov_removal_effect'].abs()
    m_sum = markov['markov_abs'].sum()
    markov['Markov'] = markov['markov_abs'] / m_sum if m_sum > 0 else 0
    markov = markov[['channel', 'Markov']]

    # Shapley (absolute value + normalization)
    shapley = pd.read_csv('shapley_results.csv')
    shapley['shapley_abs'] = shapley['shapley_value'].abs()
    s_sum = shapley['shapley_abs'].sum()
    shapley['Shapley'] = shapley['shapley_abs'] / s_sum if s_sum > 0 else 0
    shapley = shapley[['channel', 'Shapley']]

    # Baseline models (convert percentages to fractions)
    baseline = pd.read_csv('baseline_results_pct.csv')
    for col in baseline.columns:
        if col != 'channel':
            baseline[col] = baseline[col] / 100

    # Merge all results
    df = gt.merge(markov,   on='channel')
    df = df.merge(shapley,  on='channel')
    df = df.merge(baseline, on='channel')

    # Read optimal alpha if available
    optimal_alpha = 0.5
    if os.path.exists('optimal_alpha.csv'):
        alpha_df = pd.read_csv('optimal_alpha.csv')
        optimal_alpha = alpha_df['optimal_alpha'].values[0]

    # Hybrid model with optimal alpha
    df['Hybrid_M_S'] = optimal_alpha * df['Markov'] + (1 - optimal_alpha) * df['Shapley']

    return df, optimal_alpha


# =============================================================================
# 2. PERFORMANCE METRICS
# =============================================================================

def compute_metrics(df: pd.DataFrame, model_columns: list) -> pd.DataFrame:
    """Compute MAE, RMSE, Spearman correlation, and ranking accuracy
    for each attribution model against the ground truth."""
    gt = df['Ground_Truth'].values
    gt_rank = pd.Series(gt).rank(ascending=False).values

    metrics = []
    for model in model_columns:
        pred = df[model].values

        mae  = np.mean(np.abs(pred - gt))
        rmse = np.sqrt(np.mean((pred - gt) ** 2))
        spearman_corr, _ = spearmanr(pred, gt)

        pred_rank = pd.Series(pred).rank(ascending=False).values
        ranking_acc = np.mean(pred_rank == gt_rank)

        metrics.append({
            'Model'           : model,
            'MAE'             : round(mae, 4),
            'RMSE'            : round(rmse, 4),
            'Spearman_Corr'   : round(spearman_corr, 3),
            'Ranking_Accuracy': f"{ranking_acc * 100:.0f}%"
        })

    return pd.DataFrame(metrics).sort_values('MAE')


# =============================================================================
# 3. VISUALIZATIONS
# =============================================================================

def plot_attribution_comparison(df, model_columns, output):
    """Visualize attribution distributions across all models against
    the ground truth."""
    channels = df['channel'].tolist()
    n_models = len(model_columns) + 1
    x = np.arange(len(channels))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(15, 7))

    bars_gt = ax.bar(x, df['Ground_Truth'] * 100, width, label='Ground Truth',
                     color='#1F2937', edgecolor='white', linewidth=1.5, zorder=3)
    ax.bar_label(bars_gt, fmt='%.1f%%', padding=2, fontsize=7, fontweight='bold')

    colors = ['#DC2626', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6',
              '#EC4899', '#06B6D4', '#F97316']
    for i, model in enumerate(model_columns):
        offset = width * (i + 1)
        ax.bar(x + offset, df[model] * 100, width, label=model,
               color=colors[i % len(colors)], alpha=0.85,
               edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Channel', fontsize=12)
    ax.set_ylabel('Attribution Share (%)', fontsize=12)
    ax.set_title('Attribution Model Comparison Against Ground Truth',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x + width * len(model_columns) / 2)
    ax.set_xticklabels(channels, rotation=15, ha='right')
    ax.legend(loc='upper right', ncol=2, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.show()


def plot_error_metrics(metrics_df, output):
    """Visualize MAE and RMSE for all models side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Performance - Error Metrics (Lower = Better)',
                 fontsize=14, fontweight='bold', y=1.02)

    def get_color(model):
        if model in ['Markov', 'Shapley', 'Hybrid_M_S']:
            return '#DC2626'
        return '#9CA3AF'

    colors = [get_color(m) for m in metrics_df['Model']]

    bars1 = axes[0].barh(metrics_df['Model'], metrics_df['MAE'], color=colors,
                         edgecolor='white', height=0.6)
    axes[0].bar_label(bars1, fmt='%.4f', padding=3, fontsize=9, fontweight='bold')
    axes[0].set_xlabel('MAE')
    axes[0].set_title('Mean Absolute Error', fontsize=11, fontweight='bold')
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].invert_yaxis()

    bars2 = axes[1].barh(metrics_df['Model'], metrics_df['RMSE'], color=colors,
                         edgecolor='white', height=0.6)
    axes[1].bar_label(bars2, fmt='%.4f', padding=3, fontsize=9, fontweight='bold')
    axes[1].set_xlabel('RMSE')
    axes[1].set_title('Root Mean Squared Error', fontsize=11, fontweight='bold')
    axes[1].spines[['top', 'right']].set_visible(False)
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(df, model_columns, output):
    """Visualize Spearman rank correlation matrix across all models
    and the ground truth."""
    cols = ['Ground_Truth'] + model_columns
    corr_matrix = df[cols].corr(method='spearman')

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr_matrix.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.set_yticklabels(cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr_matrix.values[i, j]
            text_color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=text_color, fontweight='bold', fontsize=9)

    plt.colorbar(im, ax=ax, label='Spearman Correlation')
    ax.set_title('Model Ranking Correlation (Spearman)',
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# 4. MAIN ROUTINE
# =============================================================================

if __name__ == '__main__':

    print("="*70)
    print("  FINAL COMPARISON REPORT")
    print("  Attribution Models vs Ground Truth")
    print("="*70)

    # --- Load ---
    print("\n[1/4] Loading all model outputs...")
    df, optimal_alpha = load_all_results()
    model_columns = [c for c in df.columns if c not in ['channel', 'Ground_Truth']]
    print(f"  Loaded models    : {model_columns}")
    print(f"  Hybrid alpha     : {optimal_alpha:.2f}")

    # --- Comparison table ---
    print("\n[2/4] Attribution Comparison Table (%)")
    display_df = df.copy()
    for col in df.columns:
        if col != 'channel':
            display_df[col] = (df[col] * 100).round(2)
    print("\n" + display_df.to_string(index=False))

    # --- Metrics ---
    print("\n[3/4] Performance Metrics")
    metrics_df = compute_metrics(df, model_columns)
    print("\n" + "="*70)
    print("  MODEL PERFORMANCE RANKING (sorted by MAE - Lower = Better)")
    print("="*70)
    print(metrics_df.to_string(index=False))

    winner = metrics_df.iloc[0]['Model']
    print(f"\n  Best Model: {winner}")
    print(f"     MAE      : {metrics_df.iloc[0]['MAE']}")
    print(f"     RMSE     : {metrics_df.iloc[0]['RMSE']}")
    print(f"     Spearman : {metrics_df.iloc[0]['Spearman_Corr']}")

    # Comparison against Last-Click
    proposed = ['Markov', 'Shapley', 'Hybrid_M_S']
    print("\n  --- Proposed Models vs Last-Click Performance ---")
    lc_mae = metrics_df[metrics_df['Model'] == 'Last_Click']['MAE'].values[0]
    for model in proposed:
        if model in metrics_df['Model'].values:
            model_mae = metrics_df[metrics_df['Model'] == model]['MAE'].values[0]
            improvement = (lc_mae - model_mae) / lc_mae * 100
            sign = 'improvement' if improvement > 0 else 'degradation'
            print(f"    {model:<12} MAE change: {abs(improvement):.1f}% ({sign})")

    # --- Visualizations ---
    print("\n[4/4] Generating visualizations...")
    plot_attribution_comparison(df, model_columns, 'comparison_attribution.png')
    plot_error_metrics(metrics_df, 'comparison_errors.png')
    plot_correlation_heatmap(df, model_columns, 'comparison_correlation.png')

    display_df.to_csv('final_comparison.csv', index=False)
    metrics_df.to_csv('final_metrics.csv', index=False)

    print("\n" + "="*70)
    print("  Final report complete.")
    print("="*70)
