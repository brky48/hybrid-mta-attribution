#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shapley Value Attribution Model
================================

This module implements a Shapley value attribution model based on
cooperative game theory. Channels are treated as players in a
cooperative game, and the marginal contribution of each channel
across all possible coalitions is computed to produce the final
attribution distribution.

Methodological refinement: the characteristic function v(S) is
defined using the fractional credit formulation introduced by
Shao & Li (2011).

Naive definition (problematic):
    v(S) = sum over users of [1 if S intersects path, else 0] * conversion
    Issue: frequently occurring channels (e.g., Social_Media) appear in
    every coalition, artificially inflating their attribution.

Fractional credit definition:
    v(S) = sum over users of (|S intersects path| / |path|) * conversion
    Rationale: a user whose journey contains 4 channels, of which 2 are
    in the coalition, contributes with weight 0.5 (not 1).

Advantages of the fractional credit formulation:
    - Preserves the efficiency axiom of the Shapley framework
    - Prevents artificial inflation of frequently occurring channels
    - Accounts for channel density within each customer journey

Reference: Shao, X., & Li, L. (2011). "Data-driven multi-touch
    attribution models." Proceedings of the 17th ACM SIGKDD
    International Conference on Knowledge Discovery and Data Mining.

Part of the thesis:
    "A Hybrid Markov Chain and Shapley Value Approach to Multi-Touch
    Attribution and Budget Optimization in Digital Marketing"
    Istanbul University, Management Information Systems, 2026.
"""

import pandas as pd
import numpy as np
import itertools
import math
from collections import defaultdict
import matplotlib.pyplot as plt


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_and_build_paths(filepath: str) -> pd.DataFrame:
    """Load CSV and construct customer journey paths with cached
    path-set and path-length representations for efficient coalition
    overlap computation."""
    df = pd.read_csv(filepath)
    df = df.sort_values(['cookie_id', 'timestamp'])

    path_df = (
        df.groupby('cookie_id')['channel']
        .apply(list)
        .reset_index(name='path')
    )

    conversions = df.groupby('cookie_id')['conversion'].max().reset_index()
    path_df = pd.merge(path_df, conversions, on='cookie_id')

    # Cache path-set and path-length for efficient overlap computation
    path_df['path_set'] = path_df['path'].apply(set)
    path_df['path_len'] = path_df['path'].apply(len)
    return path_df


# =============================================================================
# 2. CHARACTERISTIC FUNCTION (with Fractional Credit)
# =============================================================================

def calculate_v_s(path_df: pd.DataFrame, channels: list) -> dict:
    """
    Compute the characteristic function v(S) for all coalitions using
    the fractional credit formulation.

    v(S) = sum over converted users of (|path intersects S| / |path|)

    Methodological rationale:
        Each converted journey contributes to a coalition's value in
        proportion to the fraction of its channels that belong to the
        coalition. This prevents the artificial inflation of channels
        that appear frequently across journeys.

    Args:
        path_df: DataFrame containing customer journey paths.
        channels: List of marketing channels.

    Returns:
        dict: {coalition_tuple: characteristic_function_value}
    """
    # Use only converted journeys (conversion = 1)
    converted_df = path_df[path_df['conversion'] == 1].copy()

    if len(converted_df) == 0:
        subsets = []
        for r in range(1, len(channels) + 1):
            subsets.extend(itertools.combinations(channels, r))
        return {tuple(sorted(s)): 0.0 for s in subsets}

    # Enumerate all non-empty coalitions
    subsets = []
    for r in range(1, len(channels) + 1):
        subsets.extend(itertools.combinations(channels, r))

    v_s = {}
    for subset in subsets:
        subset_set = set(subset)

        # For each converted user: |path intersects S| / |path|
        # This ratio reflects the coalition's contribution share in
        # the corresponding journey.
        fractions = converted_df['path_set'].apply(
            lambda x: len(x & subset_set) / len(x) if len(x) > 0 else 0
        )

        v_s[tuple(sorted(subset))] = float(fractions.sum())

    return v_s


# =============================================================================
# 3. SHAPLEY VALUE COMPUTATION
# =============================================================================

def compute_shapley(v_s: dict, channels: list) -> dict:
    """
    Compute Shapley values using the standard formula:

        phi_i = sum over S subset N\{i} of
                [|S|! * (n - |S| - 1)! / n!] *
                [v(S union {i}) - v(S)]

    Args:
        v_s: Dictionary mapping coalitions to characteristic function
            values.
        channels: List of marketing channels (players in the game).

    Returns:
        dict: {channel: shapley_value}
    """
    n = len(channels)
    shapley_values = defaultdict(float)

    for i in channels:
        phi_i = 0.0

        # Empty coalition contribution: v({i}) - v(empty) = v({i})
        singleton = tuple(sorted([i]))
        v_singleton = v_s.get(singleton, 0)
        weight_empty = math.factorial(0) * math.factorial(n - 1) / math.factorial(n)
        phi_i += weight_empty * v_singleton

        # Non-empty coalitions excluding i
        other_channels = [c for c in channels if c != i]
        for r in range(1, len(other_channels) + 1):
            for s in itertools.combinations(other_channels, r):
                s_key       = tuple(sorted(s))
                s_union_i   = tuple(sorted(s + (i,)))

                v_s_val     = v_s.get(s_key, 0)
                v_s_union_i = v_s.get(s_union_i, 0)

                marginal = v_s_union_i - v_s_val
                weight   = (math.factorial(r) * math.factorial(n - r - 1)) / math.factorial(n)

                phi_i += weight * marginal

        shapley_values[i] = phi_i

    return dict(shapley_values)


# =============================================================================
# 4. VISUALIZATION
# =============================================================================

def plot_shapley(shapley_values: dict, channels: list):
    """Visualize Shapley value results with bar chart and pie chart."""
    vals   = np.array([shapley_values[c] for c in channels])
    # Take absolute values (Shapley values may occasionally be negative)
    vals   = np.abs(vals)
    norm   = vals / vals.sum() * 100 if vals.sum() > 0 else vals
    colors = ['#2563EB', '#16A34A', '#DC2626', '#D97706', '#7C3AED'][:len(channels)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bars = axes[0].bar(channels, vals, color=colors, edgecolor='white', width=0.55)
    axes[0].bar_label(bars, fmt='%.1f', padding=4, fontsize=10, fontweight='bold')
    axes[0].set_title('Shapley Value (Fractional Credit)',
                      fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Attributed Conversions (Fractional)')
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].tick_params(axis='x', rotation=15)

    axes[1].pie(norm, labels=channels, autopct='%1.1f%%', colors=colors,
                startangle=140, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Shapley Value (Percentage Distribution)',
                      fontsize=12, fontweight='bold')

    plt.suptitle('Shapley Value Attribution (Fractional Credit Formulation)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('shapley_results.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# 5. MAIN ROUTINE
# =============================================================================

if __name__ == '__main__':

    FILEPATH = 'customer_journey_data.csv'
    CHANNELS = ['Google_Ads', 'Social_Media', 'Email', 'Organic_Search', 'Direct']

    # --- Step 1: Data ---
    print("Loading data...")
    path_df = load_and_build_paths(FILEPATH)
    print(f"  Total users       : {len(path_df):,}")
    print(f"  Total conversions : {path_df['conversion'].sum():,}")

    # --- Step 2: v(S) with fractional credit ---
    print("\nComputing coalition values (fractional credit)...")
    v_s = calculate_v_s(path_df, CHANNELS)
    print(f"  Number of coalitions computed: {len(v_s)}")

    # --- Step 3: Shapley values ---
    print("\nComputing Shapley values...")
    shapley_vals = compute_shapley(v_s, CHANNELS)

    total_conv = path_df['conversion'].sum()
    total_shapley = sum(shapley_vals.values())

    print("\n--- Shapley Value Results (Fractional Credit) ---")
    for ch, val in sorted(shapley_vals.items(), key=lambda x: -x[1]):
        pct = val / total_shapley * 100 if total_shapley > 0 else 0
        print(f"  {ch:<20}: {val:>7.2f}  ({pct:.1f}%)")

    print(f"\n  Check - Total Shapley    : {total_shapley:.2f}")
    print(f"  Actual conversion count  : {total_conv}")
    print(f"  Efficiency ratio         : {total_shapley/total_conv*100:.1f}% "
          "(should be ~100% under fractional credit)")

    # --- Step 4: Visualization ---
    plot_shapley(shapley_vals, CHANNELS)

    # --- Step 5: Save ---
    results_df = pd.DataFrame({
        'channel': list(shapley_vals.keys()),
        'shapley_value': list(shapley_vals.values())
    })
    results_df.to_csv('shapley_results.csv', index=False)
    print("\nResults saved to 'shapley_results.csv'.")
