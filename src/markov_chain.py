#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markov Chain Attribution Model
===============================

This module implements an absorbing Markov chain attribution model
following the methodology of Anderl et al. (2016). Channel-level
attribution weights are derived from removal-effect analysis, in which
the relative reduction in overall conversion probability resulting from
the hypothetical elimination of each channel is computed.

Methodological refinements:
    1. Laplace smoothing (alpha=1.0) is applied to transition counts,
       ensuring numerical stability in the removal-effect computation
       when sparse transitions are present.
       Reference: Anderl et al. (2016), "Mapping the customer journey:
       Lessons learned from graph-based online attribution modeling."

    2. The fundamental matrix is computed via np.linalg.solve rather
       than explicit inversion through np.linalg.inv, providing
       improved numerical stability and computational efficiency.

    3. Negative removal effects, which may arise as numerical artifacts
       of the Laplace smoothing operation, are floored at zero in
       accordance with the convention established by Anderl et al.
       (2016).

Part of the thesis:
    "A Hybrid Markov Chain and Shapley Value Approach to Multi-Touch
    Attribution and Budget Optimization in Digital Marketing"
    Istanbul University, Management Information Systems, 2026.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# =============================================================================
# 1. DATA LOADING AND PATH CONSTRUCTION
# =============================================================================

def load_and_build_paths(filepath: str) -> pd.DataFrame:
    """Load CSV and construct customer journey paths."""
    df = pd.read_csv(filepath)
    df = df.sort_values(['cookie_id', 'timestamp'])

    path_df = (
        df.groupby('cookie_id')['channel']
        .apply(list)
        .reset_index(name='path')
    )

    conversions = df.groupby('cookie_id')['conversion'].max().reset_index()
    path_df = pd.merge(path_df, conversions, on='cookie_id')
    return path_df


# =============================================================================
# 2. TRANSITION MATRIX (with Laplace Smoothing)
# =============================================================================

def build_transition_matrix(
    path_df: pd.DataFrame,
    channels: list,
    alpha: float = 1.0
) -> pd.DataFrame:
    """
    Compute the Markov chain transition matrix from observed customer
    journeys, with Laplace smoothing applied to all valid transitions.

    Laplace smoothing:
        A small positive constant (alpha) is added to all transition
        counts prior to normalization. This ensures:
        - No transition probability is exactly zero
        - Numerical stability in the removal-effect computation
        - Robust behavior even when the data are sparse

    Args:
        path_df: DataFrame containing customer journey paths.
        channels: List of marketing channels.
        alpha: Smoothing parameter (default=1.0, classical Laplace
            add-one).

    Returns:
        DataFrame: Transition probability matrix.
    """
    states = ['Start'] + channels + ['Conversion', 'Non-Conversion']

    # Initialize with zeros
    counts = pd.DataFrame(0.0, index=states, columns=states)

    # Accumulate empirical transition counts
    for _, row in path_df.iterrows():
        path = row['path']
        conv = row['conversion']

        # Start -> first channel
        first = path[0] if path[0] in channels else None
        if first:
            counts.loc['Start', first] += 1

        # Channel -> Channel
        for i in range(len(path) - 1):
            src = path[i]
            dst = path[i + 1]
            if src in channels and dst in channels:
                counts.loc[src, dst] += 1

        # Last channel -> absorbing state
        last = path[-1] if path[-1] in channels else None
        if last:
            if conv == 1:
                counts.loc[last, 'Conversion'] += 1
            else:
                counts.loc[last, 'Non-Conversion'] += 1

    # === LAPLACE SMOOTHING ===
    # Add +alpha to all logically possible transitions:
    # - Start to channels
    # - Channel to channel (excluding self-loops)
    # - Channel to Conversion/Non-Conversion
    # No transitions out of absorbing states; no transitions back to Start
    for ch in channels:
        # Start -> channel
        counts.loc['Start', ch] += alpha
        # Channel -> channel (excluding self)
        for ch2 in channels:
            if ch != ch2:
                counts.loc[ch, ch2] += alpha
        # Channel -> absorbing
        counts.loc[ch, 'Conversion']     += alpha
        counts.loc[ch, 'Non-Conversion'] += alpha

    # Normalize rows to probability distributions
    row_sums = counts.sum(axis=1)
    transition_matrix = counts.div(row_sums.replace(0, np.nan), axis=0).fillna(0)

    # Absorbing states transition to themselves with probability 1
    transition_matrix.loc['Conversion',     'Conversion']     = 1.0
    transition_matrix.loc['Non-Conversion', 'Non-Conversion'] = 1.0

    return transition_matrix


# =============================================================================
# 3. CONVERSION PROBABILITY (Absorbing Markov Chain)
# =============================================================================

def conversion_probability(transition_matrix: pd.DataFrame, channels: list) -> dict:
    """
    Compute the probability of reaching the 'Conversion' absorbing
    state from each transient state.

    Fundamental matrix method:
        N = (I - Q)^{-1}
        B = N @ R   ->  Absorption probability from each transient
                        state to each absorbing state.

    Implementation note: np.linalg.solve is used rather than
    np.linalg.inv for improved numerical stability.
    """
    transient_states = ['Start'] + channels

    Q = transition_matrix.loc[transient_states, transient_states].values.astype(float)
    R = transition_matrix.loc[transient_states, ['Conversion', 'Non-Conversion']].values.astype(float)

    I = np.eye(len(transient_states))

    # solve(I-Q, R) is more stable than explicit np.linalg.inv
    try:
        B = np.linalg.solve(I - Q, R)
    except np.linalg.LinAlgError:
        B = np.linalg.pinv(I - Q) @ R

    prob_dict = {state: B[i, 0] for i, state in enumerate(transient_states)}
    return prob_dict


# =============================================================================
# 4. REMOVAL EFFECT COMPUTATION
# =============================================================================

def compute_removal_effects(
    transition_matrix: pd.DataFrame,
    channels: list,
    epsilon: float = 1e-6
) -> dict:
    """
    Compute the removal effect for each channel.

    The removal effect of a channel is defined as the relative
    reduction in overall conversion probability that would result from
    the hypothetical elimination of that channel from the system.

    Implementation notes:
        1. Small floating-point noise (< epsilon) is treated as zero.
        2. Genuinely small effects (e.g., 0.01) are preserved.
    """
    base_probs  = conversion_probability(transition_matrix, channels)
    base_conv_p = base_probs['Start']

    if base_conv_p < epsilon:
        return {ch: 0.0 for ch in channels}

    removal_effects = {}

    for ch in channels:
        reduced_channels = [c for c in channels if c != ch]

        # Remove the channel from the transition matrix
        reduced_tm = transition_matrix.drop(index=ch, columns=ch).copy()

        # Renormalize remaining rows
        transient = ['Start'] + reduced_channels
        for state in transient:
            row_sum = reduced_tm.loc[state, transient + ['Conversion', 'Non-Conversion']].sum()
            if row_sum > 0:
                reduced_tm.loc[state] = reduced_tm.loc[state] / row_sum

        # Restore absorbing state self-transitions
        reduced_tm.loc['Conversion',     'Conversion']     = 1.0
        reduced_tm.loc['Non-Conversion', 'Non-Conversion'] = 1.0

        # Compute counterfactual conversion probability
        reduced_probs  = conversion_probability(reduced_tm, reduced_channels)
        reduced_conv_p = reduced_probs['Start']

        re = (base_conv_p - reduced_conv_p) / base_conv_p

        # INTERPRETIVE NOTE:
        # Negative removal effects indicate that removing the channel
        # "increased" conversions. This may signal cannibalization, or
        # may be a Laplace-smoothing artifact. Following the convention
        # of Anderl et al. (2016), negative values are floored at zero,
        # since only non-negative removal effects represent meaningful
        # contributions for budget allocation.
        re = max(re, 0.0)

        # Clean up floating-point noise
        if re < epsilon:
            re = 0.0

        removal_effects[ch] = re

    return removal_effects


# =============================================================================
# 5. VISUALIZATION
# =============================================================================

def plot_removal_effects(removal_effects: dict, channels: list):
    """Visualize channel-level removal effects as a horizontal bar chart."""
    effects_arr = np.array([removal_effects[c] for c in channels])
    colors = ['#2563EB', '#16A34A', '#DC2626', '#D97706', '#7C3AED'][:len(channels)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(channels, effects_arr * 100, color=colors, edgecolor='white', height=0.55)

    ax.bar_label(bars, fmt='%.2f%%', padding=5, fontsize=11, fontweight='bold')
    ax.set_xlabel('Removal Effect (%)', fontsize=12)
    ax.set_title('Markov Chain Removal Effects (Smoothed)\n'
                 '(Relative conversion reduction when channel is removed)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('markov_removal_effects.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# 6. MAIN ROUTINE
# =============================================================================

if __name__ == '__main__':

    FILEPATH = 'customer_journey_data.csv'
    CHANNELS = ['Google_Ads', 'Social_Media', 'Email', 'Organic_Search', 'Direct']

    # --- Step 1: Data ---
    print("Loading data...")
    path_df = load_and_build_paths(FILEPATH)
    print(f"  Total users       : {len(path_df):,}")
    print(f"  Total conversions : {path_df['conversion'].sum():,}")

    # --- Step 2: Transition matrix (with Laplace smoothing) ---
    print("\nBuilding transition matrix (Laplace alpha=1.0)...")
    tm = build_transition_matrix(path_df, CHANNELS, alpha=1.0)
    print("\n--- Transition Matrix ---")
    print(tm.round(4).to_string())

    # --- Step 3: Removal effects ---
    print("\nComputing removal effects...")
    removal_effects = compute_removal_effects(tm, CHANNELS)

    print("\n--- Markov Chain Removal Effect Results ---")
    for ch, re in sorted(removal_effects.items(), key=lambda x: -x[1]):
        print(f"  {ch:<20}: {re*100:.3f}%")

    # Normalized percentages
    total_effect = sum(abs(v) for v in removal_effects.values())
    if total_effect > 0:
        print("\n--- Normalized (total=100%) ---")
        for ch in CHANNELS:
            pct = abs(removal_effects[ch]) / total_effect * 100
            print(f"  {ch:<20}: {pct:.2f}%")

    # --- Step 4: Visualization ---
    plot_removal_effects(removal_effects, CHANNELS)

    # --- Step 5: Save ---
    results_df = pd.DataFrame({
        'channel': list(removal_effects.keys()),
        'markov_removal_effect': list(removal_effects.values())
    })
    results_df.to_csv('markov_results.csv', index=False)
    print("\nResults saved to 'markov_results.csv'.")
