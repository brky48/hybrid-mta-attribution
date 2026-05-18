#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Journey Data Generator
================================

This module generates synthetic customer journey data for evaluating
multi-touch attribution models. The data is grounded in industry
conversion rate benchmarks and incorporates channel-specific effects,
pairwise interaction effects, and recency weighting.

The ground truth attribution distribution is established through Monte
Carlo simulation, providing an objective benchmark against which the
outputs of competing attribution models can be evaluated.

References:
    WordStream (2025). Google Ads Industry Benchmarks.
    Lucky Orange (2025). Conversion Rate Benchmarks by Industry.

Part of the thesis:
    "A Hybrid Markov Chain and Shapley Value Approach to Multi-Touch
    Attribution and Budget Optimization in Digital Marketing"
    Istanbul University, Management Information Systems, 2026.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# =============================================================================
# 1. GROUND TRUTH PARAMETERS (Industry Benchmark-Based)
# =============================================================================

# Marketing channels considered in the analysis
CHANNELS = ['Google_Ads', 'Social_Media', 'Email', 'Organic_Search', 'Direct']

# Individual channel effect strengths (ground truth)
# Values derived from industry conversion rate benchmarks
CHANNEL_EFFECTS = {
    'Google_Ads'    : 0.35,  # Highest intent (paid search)
    'Email'         : 0.25,  # High conversion (1-5%, segmented 10%+)
    'Direct'        : 0.20,  # Decision-stage users (3-6%)
    'Organic_Search': 0.15,  # Exploration/research stage (2-4%)
    'Social_Media'  : 0.10,  # Awareness stage, low direct conversion
}

# Channel visit probabilities (relative frequency in customer journeys)
# Reflects the reach/cost balance across channels
CHANNEL_VISIT_PROB = {
    'Google_Ads'    : 0.25,
    'Social_Media'  : 0.25,
    'Organic_Search': 0.20,
    'Email'         : 0.15,
    'Direct'        : 0.15,
}

# Channel interaction effects (pairwise synergy multipliers)
# These interactions represent cross-channel dynamics that single-touch
# attribution models cannot capture.
CHANNEL_INTERACTIONS = {
    ('Social_Media',   'Google_Ads'    ): 1.15,   # +15%: social to search
    ('Email',          'Direct'        ): 1.20,   # +20%: email to direct visit
    ('Organic_Search', 'Email'         ): 1.10,   # +10%: research to subscription
    ('Google_Ads',     'Email'         ): 1.12,   # +12%: search to retargeting
    ('Social_Media',   'Organic_Search'): 1.08,   # +8%:  social to organic search
}

# Journey length distribution
PATH_LENGTH_DIST  = [1, 2, 3, 4, 5, 6, 7, 8]
PATH_LENGTH_PROBS = [0.10, 0.20, 0.25, 0.20, 0.12, 0.07, 0.04, 0.02]

# Base conversion rate (without path-specific effects)
# Real-world e-commerce benchmark: 2-4%
BASE_CONVERSION_RATE = 0.025

# Channel costs (per click) - WordStream CPC references
CHANNEL_COSTS = {
    'Google_Ads'    : 1.20,   # High CPC
    'Social_Media'  : 0.80,   # Medium CPC
    'Email'         : 0.20,   # Very low (delivery cost)
    'Organic_Search': 0.00,   # Free (SEO investment not included)
    'Direct'        : 0.00,   # Free
}


# =============================================================================
# 2. CONVERSION PROBABILITY (Ground Truth Function)
# =============================================================================

def calculate_conversion_probability(path: list) -> float:
    """
    Calculate the true conversion probability for a given customer
    journey.

    Formula:
        P(conv) = BASE_RATE
                * sum(channel_effects)
                * product(pairwise_interaction_multipliers)
                * recency_bonus
                / length_normalizer

    Args:
        path: List of channels representing the customer journey.

    Returns:
        Conversion probability in the [0, 1] interval.
    """
    if not path:
        return 0.0

    # 1. Sum of individual channel effects (unique channels only)
    unique_channels = set(path)
    individual_effect = sum(CHANNEL_EFFECTS[ch] for ch in unique_channels)

    # 2. Pairwise interaction multipliers (consecutive channel pairs)
    interaction_multiplier = 1.0
    for i in range(len(path) - 1):
        pair = (path[i], path[i + 1])
        if pair in CHANNEL_INTERACTIONS:
            interaction_multiplier *= CHANNEL_INTERACTIONS[pair]

    # 3. Recency bonus: the terminal channel receives amplified weight
    #    (reflecting the empirically observed last-touch signal strength)
    last_channel_bonus = 1.0 + (CHANNEL_EFFECTS[path[-1]] * 0.3)

    # 4. Path length normalization (prevents unbounded growth with length)
    length_normalizer = 1.0 / (1.0 + 0.1 * len(path))

    # 5. Penalty for journeys consisting solely of Social_Media
    #    (single-channel social journeys exhibit low conversion in practice)
    if unique_channels == {'Social_Media'}:
        interaction_multiplier *= 0.6

    # Final probability
    prob = (BASE_CONVERSION_RATE
            * individual_effect
            * interaction_multiplier
            * last_channel_bonus
            * length_normalizer)

    # Clip to [0, 1]
    return min(max(prob, 0.0), 1.0)


# =============================================================================
# 3. JOURNEY SIMULATION
# =============================================================================

def generate_user_path(rng: np.random.Generator) -> list:
    """
    Generate a customer journey path for a single user.

    Journey length is sampled from PATH_LENGTH_DIST; at each step the
    channel is selected according to CHANNEL_VISIT_PROB.
    """
    length   = rng.choice(PATH_LENGTH_DIST, p=PATH_LENGTH_PROBS)

    channels = list(CHANNEL_VISIT_PROB.keys())
    probs    = list(CHANNEL_VISIT_PROB.values())

    path = list(rng.choice(channels, size=length, p=probs))
    return path


def simulate_customer_journey(
    n_users      : int = 30_000,
    start_date   : str = '2026-02-01',
    seed         : int = 42,
    verbose      : bool = True
) -> pd.DataFrame:
    """
    Generate customer journey data for n_users users.

    Returns:
        DataFrame with columns: cookie_id | timestamp | channel |
        conversion | cost
    """
    rng = np.random.default_rng(seed)
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')

    records  = []
    n_conversions = 0

    for user_idx in range(n_users):
        cookie_id = f"user_{user_idx:05d}"

        # 1. Generate the journey
        path = generate_user_path(rng)

        # 2. Compute the true conversion probability
        true_prob = calculate_conversion_probability(path)

        # 3. Bernoulli trial: did this user convert?
        converted = rng.random() < true_prob

        # 4. Generate timestamps (touchpoints 1-6 hours apart)
        user_start_offset = rng.integers(0, 21)   # within a 3-week window
        user_start = start_dt + timedelta(days=int(user_start_offset))

        for step, channel in enumerate(path):
            timestamp = user_start + timedelta(hours=step * 2)

            # Conversion is recorded at the terminal touchpoint
            is_conv = 1 if (converted and step == len(path) - 1) else 0

            records.append({
                'cookie_id' : cookie_id,
                'timestamp' : timestamp,
                'channel'   : channel,
                'conversion': is_conv,
                'cost'      : CHANNEL_COSTS[channel]
            })

        if converted:
            n_conversions += 1

        # Progress reporting
        if verbose and (user_idx + 1) % 5000 == 0:
            print(f"  {user_idx + 1:>6} / {n_users} users processed...")

    df = pd.DataFrame(records)

    if verbose:
        print(f"\n  Total users       : {n_users:,}")
        print(f"  Total conversions : {n_conversions:,}")
        print(f"  Conversion rate   : {n_conversions / n_users * 100:.2f}%")
        print(f"  Total rows        : {len(df):,}")

    return df


# =============================================================================
# 4. GROUND TRUTH ATTRIBUTION COMPUTATION
# =============================================================================

def compute_ground_truth_attribution(n_simulations: int = 100_000, seed: int = 123) -> dict:
    """
    Compute the ground truth attribution distribution through Monte
    Carlo simulation.

    For each channel, the average marginal contribution is computed as
    the difference between the conversion probability of journeys
    containing the channel and the counterfactual probability of
    journeys with the channel removed. Final values are normalized to
    sum to unity.

    Returns:
        dict: {channel: normalized_ground_truth_weight}
    """
    rng = np.random.default_rng(seed)

    # For each channel: avg P(conv | channel present) - avg P(conv | absent)
    contributions = {ch: 0.0 for ch in CHANNELS}
    counts        = {ch: 0   for ch in CHANNELS}

    baseline_total = 0.0
    n_paths = 0

    for _ in range(n_simulations):
        path = generate_user_path(rng)
        prob = calculate_conversion_probability(path)
        baseline_total += prob
        n_paths += 1

        # Compute marginal contribution for each channel in the path
        for ch in set(path):
            # Remove the channel from the path; what is the new probability?
            path_without = [p for p in path if p != ch]
            prob_without = calculate_conversion_probability(path_without) if path_without else 0.0

            contributions[ch] += (prob - prob_without)
            counts[ch] += 1

    # Compute average contributions and normalize
    avg_contributions = {
        ch: contributions[ch] / counts[ch] if counts[ch] > 0 else 0.0
        for ch in CHANNELS
    }

    total = sum(avg_contributions.values())
    normalized = {ch: v / total for ch, v in avg_contributions.items()} if total > 0 else avg_contributions

    return normalized


# =============================================================================
# 5. MAIN ROUTINE
# =============================================================================

if __name__ == '__main__':

    print("="*65)
    print("  CUSTOMER JOURNEY DATA GENERATOR")
    print("  Ground Truth-Based Synthetic Data Generation")
    print("="*65)

    # --- Parameters ---
    N_USERS    = 30_000
    SEED       = 42
    OUTPUT_CSV = 'customer_journey_data.csv'

    # --- Step 1: Compute ground truth ---
    print("\n[1/3] Computing ground truth attribution weights...")
    print("      (Monte Carlo simulation with 100K journeys)")
    ground_truth = compute_ground_truth_attribution(n_simulations=100_000, seed=123)

    print("\n  --- GROUND TRUTH CHANNEL ATTRIBUTION ---")
    print(f"  {'Channel':<20} {'True Weight (%)':<15}")
    print("  " + "-"*40)
    for ch, val in sorted(ground_truth.items(), key=lambda x: -x[1]):
        print(f"  {ch:<20} {val*100:>6.2f}%")

    # Save ground truth to CSV for downstream comparison
    gt_df = pd.DataFrame({
        'channel'              : list(ground_truth.keys()),
        'ground_truth_weight'  : list(ground_truth.values())
    })
    gt_df.to_csv('ground_truth.csv', index=False)
    print("\n  Ground truth saved to 'ground_truth.csv'.")

    # --- Step 2: Generate the dataset ---
    print(f"\n[2/3] Generating dataset with {N_USERS:,} users...")
    df = simulate_customer_journey(n_users=N_USERS, seed=SEED, verbose=True)

    # --- Step 3: Save to CSV ---
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[3/3] Data saved to '{OUTPUT_CSV}'.")

    # --- Summary statistics ---
    print("\n" + "="*65)
    print("  DATASET SUMMARY")
    print("="*65)
    print(f"  Total touchpoints     : {len(df):,}")
    print(f"  Total unique users    : {df['cookie_id'].nunique():,}")
    print(f"  Total conversions     : {df['conversion'].sum():,}")
    print(f"  Conversion rate       : {df['conversion'].sum() / df['cookie_id'].nunique() * 100:.2f}%")
    print(f"  Average path length   : {df.groupby('cookie_id').size().mean():.2f} steps")
    print(f"  Total cost            : {df['cost'].sum():,.2f}")

    print("\n  --- Channel-Level Distribution ---")
    channel_stats = df.groupby('channel').agg(
        total_touchpoints=('channel', 'count'),
        total_conversions=('conversion', 'sum'),
        total_cost=('cost', 'sum')
    ).round(2)
    print(channel_stats.to_string())

    print("\n" + "="*65)
    print("  Data generation complete. Run modules in this order:")
    print("    1) python markov_chain.py")
    print("    2) python shapley_value.py")
    print("    3) python budget_optimization.py")
    print("    4) python baseline_models.py")
    print("    5) python comparison_report.py")
    print("="*65)
