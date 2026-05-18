#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revenue Simulation Module
==========================

This module implements the revenue simulation framework described in
Section 3.8.4 of the thesis. It addresses a methodologically distinct
question from attribution accuracy: given that each attribution model
produces a budget allocation based on its own (potentially erroneous)
estimates of channel effectiveness, which model's allocation produces
the highest expected revenue when evaluated under the true (ground
truth) channel structure?

The simulation framework operates on the principle that attribution
models do not know the ground truth; each model believes its own
estimates and allocates budget accordingly. The resulting allocations
are then evaluated against the true diminishing-returns response
curves, yielding an objective measure of each model's practical
performance.

Implementation features:
    1. Absolute USD-based saturation parameters (consistent behavior
       across 10K-500K USD budget range).
    2. Relaxed per-channel constraints (min_pct=0.05, max_pct=0.80)
       to permit meaningful model differentiation.
    3. Differential Evolution optimizer for robust global
       optimization on non-convex response surfaces.

Question addressed:
    "Which attribution model, applied to the same budget, generates
    the most conversions under the true channel effect structure?"

Part of the thesis:
    "A Hybrid Markov Chain and Shapley Value Approach to Multi-Touch
    Attribution and Budget Optimization in Digital Marketing"
    Istanbul University, Management Information Systems, 2026.
"""

import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 0. PARAMETERS (consistent with budget_optimization.py)
# =============================================================================

SATURATION_PARAMS = {
    'Google_Ads'    : {'k': 1.5, 'half_saturation': 40_000},
    'Email'         : {'k': 3.0, 'half_saturation': 8_000},
    'Social_Media'  : {'k': 2.0, 'half_saturation': 20_000},
    'Direct'        : {'k': 2.5, 'half_saturation': 5_000},
    'Organic_Search': {'k': 1.0, 'half_saturation': 30_000},
}

PAID_CHANNELS = ['Social_Media', 'Google_Ads', 'Email']


# =============================================================================
# 1. RESPONSE FUNCTION (Ground Truth-Based)
# =============================================================================

def true_response(spend: float, true_effect: float, k: float, half_saturation: float) -> float:
    """Compute the ground truth-based response for a given channel
    expenditure level."""
    if spend <= 0:
        return 0.0
    return true_effect * (1 - np.exp(-k * spend / half_saturation))


def total_expected_response(
    allocations: dict, true_effects: dict, channels: list = None
) -> float:
    """Compute the total expected response of a budget allocation
    under the ground truth channel structure.

    Args:
        allocations: Channel-level budget allocations.
        true_effects: Ground truth channel effect parameters.
        channels: Optional channel filter.

    Returns:
        Total expected response across all channels.
    """
    if channels is None:
        channels = list(allocations.keys())
    total = 0.0
    for ch in channels:
        if ch in allocations and ch in true_effects and ch in SATURATION_PARAMS:
            params = SATURATION_PARAMS[ch]
            total += true_response(
                allocations[ch],
                true_effects[ch],
                params['k'],
                params['half_saturation']
            )
    return total


# =============================================================================
# 2. PER-MODEL BUDGET ALLOCATION
# =============================================================================

def get_model_allocations(
    model_weights: pd.Series,
    total_budget: float,
    paid_channels: list = PAID_CHANNELS,
    min_pct: float = 0.05,
    max_pct: float = 0.80,
    seed: int = 42
) -> dict:
    """
    Compute the budget allocation that an attribution model would
    produce given its own attribution weights.

    Methodological note: each model maximizes the response function
    using its OWN weights (it does not have access to the ground
    truth). This simulates the real-world scenario in which a model
    must allocate budget based on its own estimates of channel
    effectiveness.

    Args:
        model_weights: Series of attribution weights from a model.
        total_budget: Total budget to allocate.
        paid_channels: List of channels eligible for paid spend.
        min_pct: Minimum allocation per channel (fraction of total).
        max_pct: Maximum allocation per channel (fraction of total).
        seed: Random seed for reproducibility.

    Returns:
        dict: {channel: allocated_budget}
    """
    weights = model_weights.copy()
    weights = weights[weights.index.isin(paid_channels)]

    if weights.sum() == 0:
        return {ch: total_budget / len(paid_channels) for ch in paid_channels}

    # Score each channel according to the model's own beliefs
    weights = weights / weights.sum()
    scores = np.array([weights.get(ch, 0) for ch in paid_channels])

    k_vals  = np.array([SATURATION_PARAMS[ch]['k'] for ch in paid_channels])
    hs_vals = np.array([SATURATION_PARAMS[ch]['half_saturation'] for ch in paid_channels])

    n = len(paid_channels)

    # The model attempts to maximize its own perceived response
    def neg_score_response(x):
        total = 0.0
        for i in range(n):
            total += true_response(x[i], scores[i], k_vals[i], hs_vals[i])
        budget_violation = abs(np.sum(x) - total_budget) / total_budget
        return -total + 100 * budget_violation

    bounds = [(total_budget * min_pct, total_budget * max_pct)] * n

    result = differential_evolution(
        neg_score_response, bounds,
        seed=seed, tol=1e-8, maxiter=300, popsize=30, polish=True
    )

    x_final = result.x
    x_final = x_final * (total_budget / np.sum(x_final))
    return {paid_channels[i]: x_final[i] for i in range(n)}


# =============================================================================
# 3. MODEL COMPARISON
# =============================================================================

def compare_all_models(
    total_budget: float = 100_000,
    final_comparison_csv: str = 'final_comparison.csv',
    ground_truth_csv: str = 'ground_truth.csv'
) -> pd.DataFrame:
    """Compare all attribution models in terms of revenue performance
    by evaluating each model's allocation under the ground truth
    response structure."""
    comp_df = pd.read_csv(final_comparison_csv)
    gt_df = pd.read_csv(ground_truth_csv)
    true_effects = dict(zip(gt_df['channel'], gt_df['ground_truth_weight']))

    model_columns = [c for c in comp_df.columns
                     if c not in ['channel', 'Ground_Truth']]

    results = []
    for model in model_columns:
        weights = pd.Series(
            comp_df[model].values / 100,
            index=comp_df['channel']
        )
        allocations = get_model_allocations(weights, total_budget)
        # Compute actual response under ground truth
        expected_revenue = total_expected_response(allocations, true_effects, PAID_CHANNELS)
        results.append({
            'Model'            : model,
            **{f'Spend_{ch}': allocations[ch] for ch in PAID_CHANNELS},
            'Expected_Response': expected_revenue
        })

    # Ground Truth Optimal (theoretical upper bound)
    gt_weights = pd.Series(
        [true_effects.get(ch, 0) for ch in comp_df['channel']],
        index=comp_df['channel']
    )
    gt_allocations = get_model_allocations(gt_weights, total_budget)
    gt_revenue = total_expected_response(gt_allocations, true_effects, PAID_CHANNELS)
    results.append({
        'Model'            : 'Ground_Truth_Optimal',
        **{f'Spend_{ch}': gt_allocations[ch] for ch in PAID_CHANNELS},
        'Expected_Response': gt_revenue
    })

    df = pd.DataFrame(results).sort_values('Expected_Response', ascending=False)

    if 'Hybrid_M_S' in df['Model'].values:
        hybrid_revenue = df.loc[df['Model'] == 'Hybrid_M_S', 'Expected_Response'].values[0]
        df['Vs_Hybrid_%'] = ((df['Expected_Response'] - hybrid_revenue) / hybrid_revenue * 100).round(3)

    if 'Last_Click' in df['Model'].values:
        lc_revenue = df.loc[df['Model'] == 'Last_Click', 'Expected_Response'].values[0]
        df['Vs_LastClick_%'] = ((df['Expected_Response'] - lc_revenue) / lc_revenue * 100).round(3)

    return df


# =============================================================================
# 4. MULTI-BUDGET ANALYSIS
# =============================================================================

def multi_budget_revenue_test(budgets: list, **kwargs) -> pd.DataFrame:
    """Run revenue simulation across multiple total budget levels to
    analyze the budget-dependence of model performance."""
    all_results = []
    for budget in budgets:
        df = compare_all_models(total_budget=budget, **kwargs)
        df['Total_Budget'] = budget
        all_results.append(df)
    return pd.concat(all_results, ignore_index=True)


# =============================================================================
# 5. VISUALIZATIONS
# =============================================================================

def plot_revenue_comparison(df: pd.DataFrame, total_budget: float):
    """Visualize the revenue performance ranking of all models at a
    given total budget level."""
    df = df.sort_values('Expected_Response', ascending=True).reset_index(drop=True)

    def get_color(model):
        if model == 'Ground_Truth_Optimal':
            return '#1F2937'
        elif model in ['Markov', 'Shapley', 'Hybrid_M_S']:
            return '#DC2626'
        return '#9CA3AF'

    colors = [get_color(m) for m in df['Model']]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(df['Model'], df['Expected_Response'], color=colors,
                   edgecolor='white', height=0.65)
    ax.bar_label(bars, fmt='%.4f', padding=5, fontsize=9, fontweight='bold')
    ax.set_xlabel('Expected Total Response (Ground Truth-Based)', fontsize=12)
    ax.set_title(f'Expected Conversion Performance by Model\n'
                 f'(Budget: {total_budget:,.0f} USD  |  Higher = Better)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1F2937', label='Theoretical Upper Bound'),
        Patch(facecolor='#DC2626', label='Proposed Models'),
        Patch(facecolor='#9CA3AF', label='Classical Baselines'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig('revenue_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_multi_budget_revenue(multi_df: pd.DataFrame):
    """Visualize multi-budget revenue simulation results: absolute
    response curves (left) and Hybrid-vs-Last-Click improvement
    pattern (right)."""
    budgets = sorted(multi_df['Total_Budget'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: absolute response
    ax1 = axes[0]
    models_to_show = ['Hybrid_M_S', 'Last_Click', 'Shapley', 'Markov', 'Ground_Truth_Optimal']
    colors_map = {
        'Ground_Truth_Optimal': '#1F2937',
        'Hybrid_M_S'          : '#DC2626',
        'Markov'              : '#F59E0B',
        'Shapley'             : '#10B981',
        'Last_Click'          : '#6B7280',
    }
    markers_map = {
        'Ground_Truth_Optimal': 's',
        'Hybrid_M_S'          : 'o',
        'Markov'              : '^',
        'Shapley'             : 'D',
        'Last_Click'          : 'v',
    }

    for model in models_to_show:
        sub = multi_df[multi_df['Model'] == model].sort_values('Total_Budget')
        if len(sub) > 0:
            ax1.plot(sub['Total_Budget'], sub['Expected_Response'],
                     marker=markers_map[model], linewidth=2.5, markersize=10,
                     label=model, color=colors_map.get(model, '#888'))

    ax1.set_xscale('log')
    ax1.set_xlabel('Total Budget (USD, log scale)', fontsize=11)
    ax1.set_ylabel('Expected Response', fontsize=11)
    ax1.set_title('Model Performance Across Budget Levels', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

    # Right: Hybrid vs Last-Click improvement
    ax2 = axes[1]
    improvements = []
    for budget in budgets:
        sub = multi_df[multi_df['Total_Budget'] == budget]
        h = sub.loc[sub['Model'] == 'Hybrid_M_S', 'Expected_Response'].values
        lc = sub.loc[sub['Model'] == 'Last_Click', 'Expected_Response'].values
        if len(h) > 0 and len(lc) > 0:
            imp = (h[0] - lc[0]) / lc[0] * 100
            improvements.append({'budget': budget, 'improvement': imp})

    imp_df = pd.DataFrame(improvements)
    bars = ax2.bar([f'{b/1000:.0f}K' for b in imp_df['budget']],
                   imp_df['improvement'],
                   color='#DC2626', edgecolor='white', width=0.6)
    ax2.bar_label(bars, fmt='%+.2f%%', padding=4, fontsize=10, fontweight='bold')
    ax2.axhline(y=0, color='gray', linewidth=0.8)
    ax2.set_xlabel('Total Budget (USD)', fontsize=11)
    ax2.set_ylabel('Hybrid vs Last-Click Improvement (%)', fontsize=11)
    ax2.set_title('Hybrid Model Advantage Over Last-Click', fontsize=12, fontweight='bold')
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('Revenue Simulation - Multi-Budget Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('multi_budget_revenue.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# 6. MAIN ROUTINE
# =============================================================================

if __name__ == '__main__':
    PRIMARY_BUDGET = 100_000
    TEST_BUDGETS   = [10_000, 50_000, 100_000, 500_000]

    print("="*70)
    print("  REVENUE SIMULATION - Model Comparison")
    print("="*70)

    print(f"\n[1/3] Primary comparison (Budget: {PRIMARY_BUDGET:,.0f} USD)...")
    df = compare_all_models(total_budget=PRIMARY_BUDGET)

    print("\n--- MODEL PERFORMANCE RANKING ---")
    print(f"{'Model':<25} {'Response':>10} {'vs Hybrid':>11} {'vs LastClick':>13}")
    print("-"*65)
    for _, row in df.iterrows():
        vs_h  = f"{row.get('Vs_Hybrid_%', 0):+.2f}%"
        vs_lc = f"{row.get('Vs_LastClick_%', 0):+.2f}%"
        print(f"{row['Model']:<25} {row['Expected_Response']:>10.4f}  "
              f"{vs_h:>10}  {vs_lc:>12}")

    winner = df.iloc[0]
    print(f"\n  Best Model: {winner['Model']} (Response: {winner['Expected_Response']:.4f})")

    if 'Hybrid_M_S' in df['Model'].values:
        h_row = df[df['Model'] == 'Hybrid_M_S'].iloc[0]
        lc_imp = h_row.get('Vs_LastClick_%', 0)
        print(f"\n  Hybrid generates {lc_imp:+.2f}% more conversions than Last-Click.")

    plot_revenue_comparison(df, PRIMARY_BUDGET)
    df.to_csv('revenue_comparison_results.csv', index=False)

    print(f"\n[2/3] Multi-budget test ({TEST_BUDGETS} USD)...")
    multi_df = multi_budget_revenue_test(budgets=TEST_BUDGETS)
    multi_df.to_csv('multi_budget_revenue_results.csv', index=False)

    print("\n--- HYBRID vs LAST-CLICK (Across Budget Levels) ---")
    for budget in TEST_BUDGETS:
        sub = multi_df[multi_df['Total_Budget'] == budget]
        h = sub.loc[sub['Model'] == 'Hybrid_M_S', 'Expected_Response'].values
        lc = sub.loc[sub['Model'] == 'Last_Click', 'Expected_Response'].values
        if len(h) > 0 and len(lc) > 0:
            imp = (h[0] - lc[0]) / lc[0] * 100
            print(f"  {budget:>8,} USD: Hybrid {h[0]:.4f}  |  "
                  f"Last-Click {lc[0]:.4f}  |  Improvement: {imp:+.2f}%")

    plot_multi_budget_revenue(multi_df)

    print("\n[3/3] Results saved successfully.")
