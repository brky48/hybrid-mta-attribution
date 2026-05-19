"""
Data Loader Module
==================

Cached loaders for all CSV files in the results/ directory.
Uses Streamlit's @st.cache_data to avoid redundant disk reads
across pages.

All paths are relative to the repository root.
"""

import os
import pandas as pd
import streamlit as st


# ============================================================================
# PATHS
# ============================================================================

# Repository root (two levels up from app/utils/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
DATA_DIR    = os.path.join(REPO_ROOT, 'data')


# ============================================================================
# CONSTANTS
# ============================================================================

CHANNELS = ['Google_Ads', 'Email', 'Direct', 'Organic_Search', 'Social_Media']

PAID_CHANNELS = ['Google_Ads', 'Email', 'Social_Media']

# Channel display names (for UI)
CHANNEL_DISPLAY = {
    'Google_Ads'    : 'Google Ads',
    'Email'         : 'Email',
    'Direct'        : 'Direct',
    'Organic_Search': 'Organic Search',
    'Social_Media'  : 'Social Media',
}

# Channel colors (consistent across all pages)
CHANNEL_COLORS = {
    'Google_Ads'    : '#4285F4',   # Google blue
    'Email'         : '#EA4335',   # Email red
    'Direct'        : '#34A853',   # Direct green
    'Organic_Search': '#FBBC04',   # Organic yellow
    'Social_Media'  : '#9C27B0',   # Social purple
}

# Model display info
MODEL_INFO = {
    'Ground_Truth'  : {'name': 'Ground Truth',     'color': '#1F2937', 'category': 'reference'},
    'Hybrid_M_S'    : {'name': 'Hybrid (M+S)',     'color': '#DC2626', 'category': 'proposed'},
    'Markov'        : {'name': 'Markov Chain',     'color': '#F59E0B', 'category': 'proposed'},
    'Shapley'       : {'name': 'Shapley Value',    'color': '#10B981', 'category': 'proposed'},
    'Last_Click'    : {'name': 'Last-Click',       'color': '#6B7280', 'category': 'baseline'},
    'First_Click'   : {'name': 'First-Click',      'color': '#9CA3AF', 'category': 'baseline'},
    'Linear'        : {'name': 'Linear',           'color': '#D1D5DB', 'category': 'baseline'},
    'Time_Decay'    : {'name': 'Time-Decay',       'color': '#A78BFA', 'category': 'baseline'},
    'Position_Based': {'name': 'Position-Based',   'color': '#60A5FA', 'category': 'baseline'},
}


# ============================================================================
# CSV LOADERS (CACHED)
# ============================================================================

@st.cache_data
def load_ground_truth() -> pd.DataFrame:
    """Load the ground truth attribution distribution."""
    path = os.path.join(RESULTS_DIR, 'ground_truth.csv')
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_markov_results() -> pd.DataFrame:
    """Load Markov chain removal effects."""
    path = os.path.join(RESULTS_DIR, 'markov_results.csv')
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_shapley_results() -> pd.DataFrame:
    """Load Shapley value attributions."""
    path = os.path.join(RESULTS_DIR, 'shapley_results.csv')
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_baseline_results() -> pd.DataFrame:
    """Load all 5 baseline model attributions (raw counts)."""
    path = os.path.join(RESULTS_DIR, 'baseline_results.csv')
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_baseline_results_pct() -> pd.DataFrame:
    """Load all 5 baseline model attributions (normalized %)."""
    path = os.path.join(RESULTS_DIR, 'baseline_results_pct.csv')
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_final_comparison() -> pd.DataFrame:
    """Load the final comparison table with all 8 models."""
    path = os.path.join(RESULTS_DIR, 'final_comparison.csv')
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_final_metrics() -> pd.DataFrame:
    """Load performance metrics (MAE, RMSE, Spearman) for all models."""
    path = os.path.join(RESULTS_DIR, 'final_metrics.csv')
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_optimal_alpha() -> float:
    """Load the optimal alpha value from grid search."""
    path = os.path.join(RESULTS_DIR, 'optimal_alpha.csv')
    df = pd.read_csv(path)
    return float(df['optimal_alpha'].values[0])


@st.cache_data
def load_budget_optimization_results() -> pd.DataFrame:
    """Load primary budget optimization results (100K USD)."""
    path = os.path.join(RESULTS_DIR, 'budget_optimization_results.csv')
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_multi_budget_results() -> pd.DataFrame:
    """Load multi-budget optimization results across 4 budget levels."""
    path = os.path.join(RESULTS_DIR, 'multi_budget_results.csv')
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_revenue_comparison() -> pd.DataFrame:
    """Load revenue simulation comparison across all models."""
    path = os.path.join(RESULTS_DIR, 'revenue_comparison_results.csv')
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_multi_budget_revenue() -> pd.DataFrame:
    """Load multi-budget revenue simulation results."""
    path = os.path.join(RESULTS_DIR, 'multi_budget_revenue_results.csv')
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_transition_matrix() -> pd.DataFrame:
    """Load the Markov transition matrix (8x8)."""
    path = os.path.join(RESULTS_DIR, 'markov_transition_matrix.csv')
    df = pd.read_csv(path, index_col=0)
    return df


@st.cache_data
def load_interaction_matrix() -> pd.DataFrame:
    """Load the channel interaction matrix (5x5)."""
    path = os.path.join(RESULTS_DIR, 'channel_interaction_matrix.csv')
    df = pd.read_csv(path, index_col=0)
    return df


@st.cache_data
def load_customer_journeys() -> pd.DataFrame:
    """Load the synthetic customer journey dataset (large file)."""
    path = os.path.join(DATA_DIR, 'customer_journey_data.csv')
    df = pd.read_csv(path)
    return df


# ============================================================================
# AGGREGATED LOADERS
# ============================================================================

@st.cache_data
def load_all_attributions_normalized() -> pd.DataFrame:
    """
    Load and combine attributions from all 8 models, normalized to
    sum to 1.0 per model.

    Returns:
        DataFrame with columns:
            channel, Ground_Truth, Markov, Shapley, Last_Click,
            First_Click, Linear, Time_Decay, Position_Based
    """
    # Ground truth
    gt = load_ground_truth()
    gt = gt.rename(columns={'ground_truth_weight': 'Ground_Truth'})

    # Markov (normalize)
    markov = load_markov_results().copy()
    markov['markov_abs'] = markov['markov_removal_effect'].abs()
    m_sum = markov['markov_abs'].sum()
    markov['Markov'] = markov['markov_abs'] / m_sum if m_sum > 0 else 0
    markov = markov[['channel', 'Markov']]

    # Shapley (normalize)
    shapley = load_shapley_results().copy()
    shapley['shapley_abs'] = shapley['shapley_value'].abs()
    s_sum = shapley['shapley_abs'].sum()
    shapley['Shapley'] = shapley['shapley_abs'] / s_sum if s_sum > 0 else 0
    shapley = shapley[['channel', 'Shapley']]

    # Baselines (convert from pct to fraction)
    baseline = load_baseline_results_pct().copy()
    for col in baseline.columns:
        if col != 'channel':
            baseline[col] = baseline[col] / 100

    # Merge all
    df = gt.merge(markov, on='channel')
    df = df.merge(shapley, on='channel')
    df = df.merge(baseline, on='channel')

    return df
