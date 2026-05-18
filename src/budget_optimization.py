#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Budget Optimization Model
=================================

This module implements the hybrid Markov-Shapley attribution model
and the non-linear budget optimization layer described in Chapter 3
of the thesis. The implementation integrates three methodological
components:

    1. Alpha grid search for empirical determination of the optimal
       hybrid mixing parameter against the ground truth.

    2. Channel-level diminishing-returns response curves based on the
       exponential-saturation formulation drawn from the Marketing
       Mix Modeling literature.

    3. Constrained non-linear budget optimization via the
       Differential Evolution algorithm.

Implementation refinements:
    1. Saturation parameters are specified in absolute USD rather
       than as budget proportions, ensuring consistent behavior
       across the full range of budget levels considered in the
       analysis (10K-500K USD).

    2. Per-channel allocation constraints are relaxed (min_pct=0.05,
       max_pct=0.80) to permit the optimizer to express genuine
       model preferences rather than converging mechanically to
       boundary values.

    3. The Differential Evolution algorithm is employed in place of
       gradient-based methods (e.g., SLSQP), providing robust
       convergence on non-convex response surfaces.

Additional features:
    - Alpha grid search for optimal mixing parameter selection
    - Absolute-value normalization for numerical stability
    - Diminishing-returns response curve modeling

References:
    Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017).
        Bayesian methods for media mix modeling with carryover and
        shape effects. Google Research.
    Chan, D., & Perry, M. (2017). Challenges and opportunities in
        media mix modeling. Google Research.
    Storn, R., & Price, K. (1997). Differential evolution - A simple
        and efficient heuristic for global optimization over
        continuous spaces. Journal of Global Optimization, 11,
        341-359.

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
# 0. DIMINISHING RETURNS PARAMETERS (Absolute USD-Based)
# =============================================================================

# Per-channel saturation parameters (in absolute USD)
# k                : saturation curvature
# half_saturation  : USD expenditure at which the channel reaches
#                    half of its maximum response
#
# Reference: MMM literature (Jin et al. 2017, Chan & Perry 2017)
# Industry benchmarks:
# - Email: Low CPC, rapid saturation (fixed subscriber list)
# - Social_Media: Moderate saturation (reach vs. irritation tradeoff)
# - Google_Ads: Slow saturation (continuous flow of new search queries)
SATURATION_PARAMS = {
    'Google_Ads'    : {'k': 1.5, 'half_saturation': 40_000},  # Half-saturated at 40K USD
    'Email'         : {'k': 3.0, 'half_saturation': 8_000},   # Half-saturated at 8K USD
    'Social_Media'  : {'k': 2.0, 'half_saturation': 20_000},  # Half-saturated at 20K USD
    'Direct'        : {'k': 2.5, 'half_saturation': 5_000},
    'Organic_Search': {'k': 1.0, 'half_saturation': 30_000},
}


# =============================================================================
# 1. LOAD MODEL OUTPUTS
# =============================================================================

def load_model_outputs(
    markov_file : str = 'markov_results.csv',
    shapley_file: str = 'shapley_results.csv',
    paid_only   : bool = True
) -> pd.DataFrame:
    """Load Markov and Shapley attribution outputs and merge them
    into a single DataFrame. Optionally restrict to paid channels for
    budget optimization purposes."""
    markov_df  = pd.read_csv(markov_file)
    shapley_df = pd.read_csv(shapley_file)
    merged = pd.merge(markov_df, shapley_df, on='channel')
    merged['markov_removal_effect'] = merged['markov_removal_effect'].abs()
    merged['shapley_value']         = merged['shapley_value'].abs()

    if paid_only:
        paid_channels = ['Social_Media', 'Google_Ads', 'Email']
        merged = merged[merged['channel'].isin(paid_channels)].reset_index(drop=True)
    return merged


# =============================================================================
# 2. ALPHA GRID SEARCH
# =============================================================================

def find_optimal_alpha(
    markov_full_csv: str = 'markov_results.csv',
    shapley_full_csv: str = 'shapley_results.csv',
    ground_truth_csv: str = 'ground_truth.csv',
    n_grid: int = 101
) -> tuple:
    """Identify the optimal hybrid mixing parameter alpha through
    one-dimensional grid search over [0, 1], minimizing the Mean
    Absolute Error between hybrid attribution weights and the ground
    truth distribution.

    Args:
        markov_full_csv: Path to Markov attribution results.
        shapley_full_csv: Path to Shapley attribution results.
        ground_truth_csv: Path to ground truth distribution.
        n_grid: Number of grid points (default 101, resolution 0.01).

    Returns:
        tuple: (best_alpha, best_mae, alpha_grid, mae_list)
    """
    markov_df  = pd.read_csv(markov_full_csv)
    shapley_df = pd.read_csv(shapley_full_csv)
    gt_df      = pd.read_csv(ground_truth_csv)

    df = markov_df.merge(shapley_df, on='channel').merge(gt_df, on='channel')
    df['m_abs'] = df['markov_removal_effect'].abs()
    df['s_abs'] = df['shapley_value'].abs()
    m_norm = df['m_abs'] / df['m_abs'].sum() if df['m_abs'].sum() > 0 else df['m_abs']
    s_norm = df['s_abs'] / df['s_abs'].sum() if df['s_abs'].sum() > 0 else df['s_abs']
    gt     = df['ground_truth_weight']

    alphas    = np.linspace(0, 1, n_grid)
    mae_list  = []
    for a in alphas:
        hybrid = a * m_norm + (1 - a) * s_norm
        mae_list.append(np.mean(np.abs(hybrid - gt)))

    best_idx   = int(np.argmin(mae_list))
    return alphas[best_idx], mae_list[best_idx], alphas, mae_list


# =============================================================================
# 3. HYBRID SCORE COMPUTATION
# =============================================================================

def compute_hybrid_scores(df: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    """Compute hybrid attribution scores as a convex linear
    combination of normalized Markov and Shapley weights:

        hybrid = alpha * markov_norm + (1 - alpha) * shapley_norm
    """
    df = df.copy()
    m_sum = df['markov_removal_effect'].sum()
    s_sum = df['shapley_value'].sum()
    df['markov_norm']  = df['markov_removal_effect'] / m_sum if m_sum > 0 else 0
    df['shapley_norm'] = df['shapley_value']         / s_sum if s_sum > 0 else 0
    df['hybrid_score'] = alpha * df['markov_norm'] + (1 - alpha) * df['shapley_norm']
    return df


# =============================================================================
# 4. RESPONSE FUNCTION (Absolute USD-Based)
# =============================================================================

def response_function(
    spend: float,
    score: float,
    k: float,
    half_saturation: float
) -> float:
    """
    Diminishing-returns response curve (Hill-like saturation).

        response(x) = score * (1 - exp(-k * x / half_saturation))

    Note: half_saturation is specified in absolute USD (e.g., 10K,
    40K), ensuring consistent and meaningful behavior across the
    full range of budget levels from 10K to 500K USD.

    Args:
        spend: Channel expenditure in USD.
        score: Channel-level attribution score.
        k: Saturation curvature parameter.
        half_saturation: USD expenditure at half-maximum response.

    Returns:
        Expected response at the given expenditure level.
    """
    if spend <= 0:
        return 0.0
    return score * (1 - np.exp(-k * spend / half_saturation))


# =============================================================================
# 5. BUDGET OPTIMIZATION (Differential Evolution)
# =============================================================================

def optimize_budget_with_dr(
    df: pd.DataFrame,
    total_budget: float = 10_000,
    min_pct: float = 0.05,
    max_pct: float = 0.80,
    seed: int = 42
) -> pd.DataFrame:
    """
    Optimize channel budget allocation under diminishing-returns
    response curves using the Differential Evolution algorithm.

    The algorithm provides global optimization on non-linear,
    multi-modal problems with greater robustness than gradient-based
    methods such as SLSQP, at a modest cost in computational time.

    Args:
        df: DataFrame containing channel hybrid scores.
        total_budget: Total budget to allocate (USD).
        min_pct: Minimum allocation per channel (fraction of total).
        max_pct: Maximum allocation per channel (fraction of total).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with channel-level allocations and expected
        responses.
    """
    n = len(df)
    channels = df['channel'].tolist()
    scores   = df['hybrid_score'].values

    k_vals  = np.array([SATURATION_PARAMS[ch]['k'] for ch in channels])
    hs_vals = np.array([SATURATION_PARAMS[ch]['half_saturation'] for ch in channels])

    def neg_response_with_penalty(x):
        # Sum of channel responses
        total = 0.0
        for i in range(n):
            total += response_function(x[i], scores[i], k_vals[i], hs_vals[i])
        # Budget conservation penalty
        budget_violation = abs(np.sum(x) - total_budget) / total_budget
        return -total + 100 * budget_violation

    bounds = [(total_budget * min_pct, total_budget * max_pct)] * n

    result = differential_evolution(
        neg_response_with_penalty, bounds,
        seed=seed, tol=1e-8, maxiter=300, popsize=30,
        polish=True
    )

    # Renormalize to exact budget conservation
    # (penalty term may leave a small residual deviation)
    x_final = result.x
    x_final = x_final * (total_budget / np.sum(x_final))

    df = df.copy()
    df['allocated_budget']  = np.round(x_final, 2)
    df['expected_response'] = [
        response_function(x_final[i], scores[i], k_vals[i], hs_vals[i])
        for i in range(n)
    ]
    return df


# =============================================================================
# 6. MULTI-BUDGET TEST
# =============================================================================

def multi_budget_test(
    df: pd.DataFrame,
    budgets: list,
    min_pct: float = 0.05,
    max_pct: float = 0.80
) -> pd.DataFrame:
    """Run budget optimization across multiple total budget levels
    to analyze the budget-dependence of optimal allocations."""
    all_results = []
    for budget in budgets:
        df_test = optimize_budget_with_dr(df, total_budget=budget,
                                          min_pct=min_pct, max_pct=max_pct)
        for _, row in df_test.iterrows():
            all_results.append({
                'total_budget'     : budget,
                'channel'          : row['channel'],
                'allocated_budget' : row['allocated_budget'],
                'allocated_pct'    : row['allocated_budget'] / budget * 100,
                'expected_response': row['expected_response']
            })
    return pd.DataFrame(all_results)


# =============================================================================
# 7. VISUALIZATIONS
# =============================================================================

def plot_alpha_search(alphas, mae_list, best_alpha, best_mae):
    """Visualize the alpha grid search results."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(alphas, mae_list, color='#2563EB', linewidth=2)
    ax.axvline(best_alpha, color='#DC2626', linestyle='--',
               label=f'Optimal $\\alpha$ = {best_alpha:.2f}\nMAE = {best_mae:.4f}')
    ax.fill_between(alphas, mae_list, alpha=0.15, color='#2563EB')
    ax.scatter([best_alpha], [best_mae], color='#DC2626', s=100, zorder=5)
    ax.set_xlabel(r'$\alpha$ (Markov weight)', fontsize=12)
    ax.set_ylabel('MAE (against Ground Truth)', fontsize=12)
    ax.set_title('Hybrid Model Alpha Grid Search', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('alpha_grid_search.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_response_curves(df: pd.DataFrame, max_spend: float = 100_000):
    """Visualize diminishing-returns response curves for paid
    channels, with half-saturation points marked."""
    fig, ax = plt.subplots(figsize=(11, 6))
    spend_range = np.linspace(0, max_spend, 200)
    colors = ['#2563EB', '#16A34A', '#DC2626']

    for i, (_, row) in enumerate(df.iterrows()):
        ch = row['channel']
        score = row['hybrid_score']
        k = SATURATION_PARAMS[ch]['k']
        hs = SATURATION_PARAMS[ch]['half_saturation']
        responses = [response_function(x, score, k, hs) for x in spend_range]
        ax.plot(spend_range, responses,
                label=f"{ch} (k={k}, half_sat={hs/1000:.0f}K USD)",
                color=colors[i % len(colors)], linewidth=2.5)
        # Mark the half-saturation point
        half_resp = response_function(hs, score, k, hs)
        ax.scatter([hs], [half_resp], color=colors[i % len(colors)],
                   s=80, zorder=5, edgecolor='black')

    ax.set_xlabel('Channel Spend (USD)', fontsize=12)
    ax.set_ylabel('Expected Response', fontsize=12)
    ax.set_title('Diminishing Returns Response Curves\n'
                 '(Points: half-saturation levels)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    plt.tight_layout()
    plt.savefig('response_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_multi_budget_comparison(multi_df: pd.DataFrame):
    """Visualize optimal channel allocation patterns across multiple
    budget levels."""
    budgets   = sorted(multi_df['total_budget'].unique())
    channels  = multi_df['channel'].unique().tolist()
    colors    = ['#2563EB', '#16A34A', '#DC2626'][:len(channels)]

    fig, ax = plt.subplots(figsize=(12, 6))
    x        = np.arange(len(budgets))
    width    = 0.25
    offsets  = np.linspace(-width, width, len(channels))

    for i, ch in enumerate(channels):
        ch_data = multi_df[multi_df['channel'] == ch].sort_values('total_budget')
        bars = ax.bar(x + offsets[i], ch_data['allocated_pct'], width,
                      label=ch, color=colors[i], alpha=0.85, edgecolor='white')
        ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{b/1000:.0f}K' for b in budgets])
    ax.set_xlabel('Total Budget (USD)', fontsize=12)
    ax.set_ylabel('Channel Share (%)', fontsize=12)
    ax.set_title('Optimal Allocation Across Budget Levels\n'
                 '(Diminishing Returns Effect)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('multi_budget_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# 8. MAIN ROUTINE
# =============================================================================

if __name__ == '__main__':
    PRIMARY_BUDGET = 100_000
    TEST_BUDGETS   = [10_000, 50_000, 100_000, 500_000]
    MIN_PCT        = 0.05
    MAX_PCT        = 0.80

    print("="*65)
    print("  ALPHA GRID SEARCH")
    print("="*65)
    best_alpha, best_mae, alphas, mae_list = find_optimal_alpha()
    print(f"  Optimal alpha : {best_alpha:.2f}  |  MAE: {best_mae:.4f}")
    plot_alpha_search(alphas, mae_list, best_alpha, best_mae)

    print("\n" + "="*65)
    print("  BUDGET OPTIMIZATION (Diminishing Returns)")
    print("="*65)
    df = load_model_outputs(paid_only=True)
    df = compute_hybrid_scores(df, alpha=best_alpha)
    print("\n  Hybrid scores:")
    for _, row in df.iterrows():
        print(f"    {row['channel']:<15}: {row['hybrid_score']*100:.1f}%")

    plot_response_curves(df, max_spend=100_000)

    df_primary = optimize_budget_with_dr(
        df, total_budget=PRIMARY_BUDGET, min_pct=MIN_PCT, max_pct=MAX_PCT
    )

    print(f"\n  Primary Budget: {PRIMARY_BUDGET:,.0f} USD  |  alpha={best_alpha:.2f}")
    print("-"*65)
    print(f"  {'Channel':<15} {'Hybrid':>8} {'Budget':>12} {'Share (%)':>10} {'Response':>10}")
    print("-"*65)
    for _, row in df_primary.iterrows():
        pct = row['allocated_budget'] / PRIMARY_BUDGET * 100
        print(f"  {row['channel']:<15} "
              f"{row['hybrid_score']*100:>7.1f}%  "
              f"{row['allocated_budget']:>11,.0f}  "
              f"{pct:>9.1f}%  "
              f"{row['expected_response']:>10.4f}")
    print("-"*65)
    print(f"  TOTAL Response: {df_primary['expected_response'].sum():.4f}")

    print("\n" + "="*65)
    print("  MULTI-BUDGET TEST")
    print("="*65)
    multi_df = multi_budget_test(df, TEST_BUDGETS, MIN_PCT, MAX_PCT)
    pivot = multi_df.pivot_table(
        index='channel', columns='total_budget',
        values='allocated_pct', aggfunc='first'
    ).round(2)
    pivot.columns = [f'{b/1000:.0f}K USD' for b in pivot.columns]
    print("\n  Channel Share (%) Across Budget Levels:")
    print(pivot.to_string())
    plot_multi_budget_comparison(multi_df)

    df_primary.to_csv('budget_optimization_results.csv', index=False)
    multi_df.to_csv('multi_budget_results.csv', index=False)
    pd.DataFrame({'optimal_alpha': [best_alpha], 'optimal_mae': [best_mae]}).to_csv(
        'optimal_alpha.csv', index=False
    )

    print("\nResults saved successfully.")
