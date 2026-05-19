"""
Computational Models Module
============================

Streamlit-friendly wrappers around the core attribution and budget
optimization logic. These functions are designed for fast, repeated
calls with parameter variations (e.g., when sliders move).

For full reference implementations, see the modules under src/.
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import differential_evolution

from utils.data_loader import (
    load_all_attributions_normalized,
    load_ground_truth,
    PAID_CHANNELS,
)


# ============================================================================
# SATURATION PARAMETERS (consistent with budget_optimization.py)
# ============================================================================

SATURATION_PARAMS = {
    'Google_Ads'    : {'k': 1.5, 'half_saturation': 40_000},
    'Email'         : {'k': 3.0, 'half_saturation': 8_000},
    'Social_Media'  : {'k': 2.0, 'half_saturation': 20_000},
    'Direct'        : {'k': 2.5, 'half_saturation': 5_000},
    'Organic_Search': {'k': 1.0, 'half_saturation': 30_000},
}


# ============================================================================
# HYBRID ATTRIBUTION (LIVE COMPUTATION)
# ============================================================================

@st.cache_data
def compute_hybrid_for_alpha(alpha: float) -> pd.DataFrame:
    """
    Compute the hybrid attribution distribution for a given alpha.

    hybrid = alpha * markov_normalized + (1 - alpha) * shapley_normalized

    Args:
        alpha: Mixing parameter in [0, 1].

    Returns:
        DataFrame with columns: channel, Hybrid_M_S
    """
    df = load_all_attributions_normalized()
    hybrid = alpha * df['Markov'] + (1 - alpha) * df['Shapley']
    return pd.DataFrame({
        'channel'  : df['channel'],
        'Hybrid_M_S': hybrid.values
    })


@st.cache_data
def compute_mae_for_alpha(alpha: float) -> float:
    """Compute MAE between hybrid attribution and ground truth at
    a given alpha."""
    hybrid_df = compute_hybrid_for_alpha(alpha)
    gt = load_ground_truth()
    gt = gt.rename(columns={'ground_truth_weight': 'gt'})

    merged = hybrid_df.merge(gt, on='channel')
    mae = float(np.mean(np.abs(merged['Hybrid_M_S'] - merged['gt'])))
    return mae


@st.cache_data
def compute_alpha_grid(n_points: int = 101) -> pd.DataFrame:
    """
    Pre-compute MAE values across the alpha grid for visualization.

    Returns:
        DataFrame with columns: alpha, mae
    """
    alphas = np.linspace(0, 1, n_points)
    maes = [compute_mae_for_alpha(a) for a in alphas]
    return pd.DataFrame({'alpha': alphas, 'mae': maes})


# ============================================================================
# RESPONSE FUNCTION
# ============================================================================

def response_function(spend: float, score: float, k: float,
                      half_saturation: float) -> float:
    """
    Diminishing returns response curve (exponential saturation).

        response(x) = score * (1 - exp(-k * x / half_saturation))
    """
    if spend <= 0:
        return 0.0
    return score * (1 - np.exp(-k * spend / half_saturation))


# ============================================================================
# BUDGET OPTIMIZATION (LIVE)
# ============================================================================

@st.cache_data
def optimize_budget(
    alpha: float,
    total_budget: float,
    min_pct: float = 0.05,
    max_pct: float = 0.80,
    paid_channels: tuple = ('Google_Ads', 'Email', 'Social_Media'),
    seed: int = 42
) -> pd.DataFrame:
    """
    Optimize channel budget allocation using Differential Evolution
    over diminishing-returns response curves.

    Args:
        alpha: Hybrid mixing parameter.
        total_budget: Total budget in USD.
        min_pct: Minimum allocation per channel (fraction).
        max_pct: Maximum allocation per channel (fraction).
        paid_channels: Tuple of channels eligible for paid spend.
        seed: Random seed.

    Returns:
        DataFrame with columns: channel, score, allocated_budget,
        expected_response, allocated_pct
    """
    # Compute hybrid scores for paid channels only
    hybrid_df = compute_hybrid_for_alpha(alpha)
    hybrid_paid = hybrid_df[hybrid_df['channel'].isin(paid_channels)].copy()

    # Normalize within paid channels
    hybrid_sum = hybrid_paid['Hybrid_M_S'].sum()
    if hybrid_sum > 0:
        hybrid_paid['score'] = hybrid_paid['Hybrid_M_S'] / hybrid_sum
    else:
        hybrid_paid['score'] = 1.0 / len(paid_channels)

    channels = hybrid_paid['channel'].tolist()
    scores   = hybrid_paid['score'].values

    k_vals  = np.array([SATURATION_PARAMS[ch]['k'] for ch in channels])
    hs_vals = np.array([SATURATION_PARAMS[ch]['half_saturation'] for ch in channels])

    n = len(channels)

    def neg_response_with_penalty(x):
        total = 0.0
        for i in range(n):
            total += response_function(x[i], scores[i], k_vals[i], hs_vals[i])
        budget_violation = abs(np.sum(x) - total_budget) / total_budget
        return -total + 100 * budget_violation

    bounds = [(total_budget * min_pct, total_budget * max_pct)] * n

    result = differential_evolution(
        neg_response_with_penalty, bounds,
        seed=seed, tol=1e-8, maxiter=300, popsize=30, polish=True
    )

    # Renormalize to exact budget
    x_final = result.x
    x_final = x_final * (total_budget / np.sum(x_final))

    return pd.DataFrame({
        'channel'           : channels,
        'score'             : scores,
        'allocated_budget'  : x_final.round(2),
        'allocated_pct'     : (x_final / total_budget * 100).round(2),
        'expected_response' : [
            response_function(x_final[i], scores[i], k_vals[i], hs_vals[i])
            for i in range(n)
        ]
    })


@st.cache_data
def compute_total_response(
    alpha: float,
    total_budget: float,
    min_pct: float = 0.05,
    max_pct: float = 0.80
) -> float:
    """Compute the total expected response under an optimized
    allocation."""
    df = optimize_budget(alpha, total_budget, min_pct, max_pct)
    return float(df['expected_response'].sum())


# ============================================================================
# CUSTOMER JOURNEY ANALYSIS (PATH EXPLORER)
# ============================================================================

# Ground truth parameters from data_generator.py
CHANNEL_EFFECTS = {
    'Google_Ads'    : 0.35,
    'Email'         : 0.25,
    'Direct'        : 0.20,
    'Organic_Search': 0.15,
    'Social_Media'  : 0.10,
}

CHANNEL_INTERACTIONS = {
    ('Social_Media',   'Google_Ads'    ): 1.15,
    ('Email',          'Direct'        ): 1.20,
    ('Organic_Search', 'Email'         ): 1.10,
    ('Google_Ads',     'Email'         ): 1.12,
    ('Social_Media',   'Organic_Search'): 1.08,
}

BASE_CONVERSION_RATE = 0.025


def calculate_conversion_probability(path: list) -> dict:
    """
    Compute the ground truth conversion probability for a given path,
    along with the contribution of each component.

    Args:
        path: List of channel names in journey order.

    Returns:
        dict with keys: probability, individual_effect,
        interaction_multiplier, recency_bonus, length_normalizer
    """
    if not path:
        return {
            'probability'           : 0.0,
            'individual_effect'     : 0.0,
            'interaction_multiplier': 1.0,
            'recency_bonus'         : 1.0,
            'length_normalizer'     : 1.0,
        }

    # Individual effects (unique channels)
    unique_channels = set(path)
    individual_effect = sum(CHANNEL_EFFECTS[ch] for ch in unique_channels)

    # Pairwise interactions (consecutive pairs)
    interaction_multiplier = 1.0
    for i in range(len(path) - 1):
        pair = (path[i], path[i + 1])
        if pair in CHANNEL_INTERACTIONS:
            interaction_multiplier *= CHANNEL_INTERACTIONS[pair]

    # Recency bonus
    last_channel_bonus = 1.0 + (CHANNEL_EFFECTS[path[-1]] * 0.3)

    # Length normalizer
    length_normalizer = 1.0 / (1.0 + 0.1 * len(path))

    # Single-channel social penalty
    if unique_channels == {'Social_Media'}:
        interaction_multiplier *= 0.6

    prob = (BASE_CONVERSION_RATE
            * individual_effect
            * interaction_multiplier
            * last_channel_bonus
            * length_normalizer)

    prob = min(max(prob, 0.0), 1.0)

    return {
        'probability'           : prob,
        'individual_effect'     : individual_effect,
        'interaction_multiplier': interaction_multiplier,
        'recency_bonus'         : last_channel_bonus,
        'length_normalizer'     : length_normalizer,
    }


def compute_path_attributions(path: list) -> pd.DataFrame:
    """
    Compute attribution credits for each channel in the path
    under different attribution models.

    Args:
        path: List of channel names.

    Returns:
        DataFrame with columns: channel, Last_Click, First_Click,
        Linear, Time_Decay, Position_Based
    """
    if not path:
        from utils.data_loader import CHANNELS
        return pd.DataFrame({
            'channel'       : CHANNELS,
            'Last_Click'    : [0.0] * 5,
            'First_Click'   : [0.0] * 5,
            'Linear'        : [0.0] * 5,
            'Time_Decay'    : [0.0] * 5,
            'Position_Based': [0.0] * 5,
        })

    from utils.data_loader import CHANNELS
    n = len(path)

    # Initialize
    credits = {ch: {'Last_Click': 0.0, 'First_Click': 0.0,
                    'Linear': 0.0, 'Time_Decay': 0.0,
                    'Position_Based': 0.0}
               for ch in CHANNELS}

    # Last_Click
    credits[path[-1]]['Last_Click'] = 1.0

    # First_Click
    credits[path[0]]['First_Click'] = 1.0

    # Linear
    for ch in path:
        credits[ch]['Linear'] += 1.0 / n

    # Time_Decay (half_life = 2)
    half_life = 2.0
    weights = np.array([2 ** ((i - n + 1) / half_life) for i in range(n)])
    weights = weights / weights.sum()
    for i, ch in enumerate(path):
        credits[ch]['Time_Decay'] += weights[i]

    # Position_Based (40-20-40)
    if n == 1:
        credits[path[0]]['Position_Based'] = 1.0
    elif n == 2:
        credits[path[0]]['Position_Based']  += 0.5
        credits[path[-1]]['Position_Based'] += 0.5
    else:
        credits[path[0]]['Position_Based']  += 0.40
        credits[path[-1]]['Position_Based'] += 0.40
        middle = path[1:-1]
        if middle:
            per_middle = 0.20 / len(middle)
            for ch in middle:
                credits[ch]['Position_Based'] += per_middle

    # Build DataFrame
    df = pd.DataFrame([{
        'channel'       : ch,
        'Last_Click'    : credits[ch]['Last_Click'],
        'First_Click'   : credits[ch]['First_Click'],
        'Linear'        : credits[ch]['Linear'],
        'Time_Decay'    : credits[ch]['Time_Decay'],
        'Position_Based': credits[ch]['Position_Based'],
    } for ch in CHANNELS])

    return df
