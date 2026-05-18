#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical (Heuristic) Attribution Models
=========================================

This module implements five classical heuristic attribution models
that serve as baselines for comparison against the data-driven
Markov chain, Shapley value, and hybrid attribution models:

    1. Last-Click Attribution
       - All credit assigned to the terminal touchpoint
       - The default model in Google Analytics until recent versions
       - The most widely used yet most criticized approach

    2. First-Click Attribution
       - All credit assigned to the initial touchpoint
       - Used in awareness-stage and acquisition-focused analyses

    3. Linear Attribution
       - Credit distributed equally across all touchpoints
       - Simple but assumes equal touchpoint importance (often invalid)

    4. Time-Decay Attribution
       - Exponentially increasing weight toward recent touchpoints
       - Reflects recency bias in conversion attribution

    5. Position-Based (U-Shaped) Attribution
       - 40% credit to first and last touchpoints, 20% to middle
       - Industry-standard compromise between first- and last-click

Part of the thesis:
    "A Hybrid Markov Chain and Shapley Value Approach to Multi-Touch
    Attribution and Budget Optimization in Digital Marketing"
    Istanbul University, Management Information Systems, 2026.
"""

import pandas as pd
import numpy as np
from collections import defaultdict


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_and_build_paths(filepath: str) -> pd.DataFrame:
    """Load CSV and construct customer journey paths with conversion
    information."""
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
# 2. ATTRIBUTION MODELS
# =============================================================================

def last_click_attribution(path_df: pd.DataFrame, channels: list) -> dict:
    """
    Last-Click attribution: assign full credit to the terminal channel.

    Only converted journeys are considered. Each converting journey
    contributes 1.0 unit of credit to its last channel.
    """
    attribution = defaultdict(float)
    converted   = path_df[path_df['conversion'] == 1]

    for _, row in converted.iterrows():
        last_channel = row['path'][-1]
        if last_channel in channels:
            attribution[last_channel] += 1.0

    return {ch: attribution[ch] for ch in channels}


def first_click_attribution(path_df: pd.DataFrame, channels: list) -> dict:
    """
    First-Click attribution: assign full credit to the initial channel.

    Each converting journey contributes 1.0 unit of credit to its
    first channel.
    """
    attribution = defaultdict(float)
    converted   = path_df[path_df['conversion'] == 1]

    for _, row in converted.iterrows():
        first_channel = row['path'][0]
        if first_channel in channels:
            attribution[first_channel] += 1.0

    return {ch: attribution[ch] for ch in channels}


def linear_attribution(path_df: pd.DataFrame, channels: list) -> dict:
    """
    Linear attribution: distribute credit equally across all
    touchpoints.

    For a converting journey with N touchpoints, each touchpoint
    receives 1/N units of credit.
    """
    attribution = defaultdict(float)
    converted   = path_df[path_df['conversion'] == 1]

    for _, row in converted.iterrows():
        path = row['path']
        if not path:
            continue
        credit_per_touch = 1.0 / len(path)
        for ch in path:
            if ch in channels:
                attribution[ch] += credit_per_touch

    return {ch: attribution[ch] for ch in channels}


def time_decay_attribution(
    path_df: pd.DataFrame,
    channels: list,
    half_life: float = 1.0
) -> dict:
    """
    Time-Decay attribution: weight touchpoints by their temporal
    proximity to the conversion event, with exponentially increasing
    weights toward recent touchpoints.

    Weight formula:
        w_i = 2 ** ((i - n + 1) / half_life)

    Where:
        i = touchpoint index
        n = total number of touchpoints in the journey
        half_life = number of steps for weight to halve

    The terminal touchpoint receives w = 2^0 = 1; the preceding
    touchpoint receives w = 2^(-1/half_life). Weights are normalized
    so that each journey contributes 1.0 unit of total credit.
    """
    attribution = defaultdict(float)
    converted   = path_df[path_df['conversion'] == 1]

    for _, row in converted.iterrows():
        path = row['path']
        if not path:
            continue

        n = len(path)
        # Compute weight for each touchpoint
        weights = np.array([2 ** ((i - n + 1) / half_life) for i in range(n)])
        # Normalize (sum = 1)
        weights = weights / weights.sum()

        for idx, ch in enumerate(path):
            if ch in channels:
                attribution[ch] += weights[idx]

    return {ch: attribution[ch] for ch in channels}


def position_based_attribution(
    path_df: pd.DataFrame,
    channels: list,
    first_weight : float = 0.40,
    last_weight  : float = 0.40
) -> dict:
    """
    Position-Based (U-Shaped) attribution: assign 40% credit to the
    first and last touchpoints, distributing the remaining 20%
    equally across intermediate touchpoints.

    This is one of the default models in Google Analytics 4 and
    reflects an industry-standard compromise between first-click and
    last-click attribution.

    Special cases:
        - Single-touchpoint journey: full credit to the only touchpoint
        - Two-touchpoint journey: 50% to each
        - Three or more touchpoints: 40% / 20% middle / 40% U-shaped
    """
    attribution = defaultdict(float)
    converted   = path_df[path_df['conversion'] == 1]
    middle_weight = 1.0 - first_weight - last_weight

    for _, row in converted.iterrows():
        path = row['path']
        if not path:
            continue

        if len(path) == 1:
            if path[0] in channels:
                attribution[path[0]] += 1.0
            continue

        if len(path) == 2:
            if path[0]  in channels: attribution[path[0]]  += 0.5
            if path[-1] in channels: attribution[path[-1]] += 0.5
            continue

        # 3+ touchpoints: U-shaped
        if path[0]  in channels: attribution[path[0]]  += first_weight
        if path[-1] in channels: attribution[path[-1]] += last_weight

        middle = path[1:-1]
        if middle:
            per_middle = middle_weight / len(middle)
            for ch in middle:
                if ch in channels:
                    attribution[ch] += per_middle

    return {ch: attribution[ch] for ch in channels}


# =============================================================================
# 3. MAIN ROUTINE
# =============================================================================

if __name__ == '__main__':

    FILEPATH = 'customer_journey_data.csv'
    CHANNELS = ['Google_Ads', 'Social_Media', 'Email', 'Organic_Search', 'Direct']

    # --- Step 1: Data ---
    print("Loading data...")
    path_df = load_and_build_paths(FILEPATH)
    total_conv = path_df['conversion'].sum()
    print(f"  Total users       : {len(path_df):,}")
    print(f"  Total conversions : {total_conv:,}")

    # --- Step 2: Run all baseline models ---
    print("\nComputing classical attribution models...")
    results = {
        'Last_Click'    : last_click_attribution(path_df, CHANNELS),
        'First_Click'   : first_click_attribution(path_df, CHANNELS),
        'Linear'        : linear_attribution(path_df, CHANNELS),
        'Time_Decay'    : time_decay_attribution(path_df, CHANNELS, half_life=2.0),
        'Position_Based': position_based_attribution(path_df, CHANNELS),
    }

    # --- Step 3: Tabulate results ---
    results_df = pd.DataFrame(results).round(2)
    results_df.index.name = 'channel'
    results_df = results_df.reset_index()

    # Raw attributed conversions
    print("\n" + "="*75)
    print("  BASELINE MODEL RESULTS - Raw Attributed Conversions")
    print("="*75)
    print(results_df.to_string(index=False))

    # Normalized percentages
    pct_df = results_df.copy()
    for model in results.keys():
        col_sum = pct_df[model].sum()
        if col_sum > 0:
            pct_df[model] = (pct_df[model] / col_sum * 100).round(2)

    print("\n" + "="*75)
    print("  BASELINE MODEL RESULTS - Normalized (%)")
    print("="*75)
    print(pct_df.to_string(index=False))

    # --- Step 4: Save ---
    results_df.to_csv('baseline_results.csv', index=False)
    pct_df.to_csv('baseline_results_pct.csv', index=False)
    print("\nResults saved to 'baseline_results.csv' and 'baseline_results_pct.csv'.")
