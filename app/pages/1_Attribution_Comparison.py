"""
Page 1: Attribution Comparison
===============================

Interactive comparison of 8 attribution models against the ground
truth. Users adjust the hybrid mixing parameter α via slider and
observe how the hybrid attribution distribution shifts in real time.
"""

import sys
import os

# Add app/ to path so utils can be imported
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import (
    load_all_attributions_normalized,
    load_final_metrics,
    load_optimal_alpha,
    CHANNELS,
    CHANNEL_DISPLAY,
    MODEL_INFO,
)
from utils.models import (
    compute_hybrid_for_alpha,
    compute_mae_for_alpha,
    compute_alpha_grid,
)
from utils.plotting import (
    plot_attribution_bars,
    plot_alpha_curve,
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Attribution Comparison",
    page_icon="📊",
    layout="wide",
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .page-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #60A5FA;
        margin-bottom: 0.5rem;
    }
    .page-subtitle {
        font-size: 1.1rem;
        color: #9CA3AF;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .insight-box {
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #60A5FA;
        margin: 1rem 0;
        color: #E5E7EB;
    }
    .insight-box-optimal {
        border-left-color: #10B981;
    }
    .insight-box-warning {
        border-left-color: #F59E0B;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HEADER
# ============================================================================

st.markdown('<h1 class="page-title">📊 Attribution Comparison</h1>',
            unsafe_allow_html=True)

st.markdown('<p class="page-subtitle">Adjust the hybrid mixing parameter α and observe attribution shifts in real time</p>',
            unsafe_allow_html=True)


# ============================================================================
# LOAD CONSTANTS
# ============================================================================

# Get optimal alpha from CSV
optimal_alpha = load_optimal_alpha()


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

# Initialize alpha slider state ONCE (only on first page load)
if 'alpha_slider' not in st.session_state:
    st.session_state.alpha_slider = optimal_alpha


# ============================================================================
# BUTTON CALLBACKS (set state BEFORE slider is rendered)
# ============================================================================

def set_alpha_optimal():
    """Set alpha to the optimal value."""
    st.session_state.alpha_slider = optimal_alpha


def set_alpha_reset():
    """Reset alpha to 0.5 (neutral middle)."""
    st.session_state.alpha_slider = 0.5


# ============================================================================
# SIDEBAR — CONTROLS
# ============================================================================

with st.sidebar:
    st.markdown("### 🎛️ Controls")

    st.markdown("**Hybrid Mixing Parameter (α)**")
    st.caption("Controls the balance between Markov and Shapley:")
    st.caption("- α = 0: Pure Shapley")
    st.caption("- α = 1: Pure Markov")

    # Slider — uses session_state.alpha_slider automatically via key
    alpha = st.slider(
        "α value",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key='alpha_slider',
        help="The hybrid attribution weight = α × Markov + (1-α) × Shapley"
    )

    # Quick action buttons with callbacks
    col_a, col_b = st.columns(2)
    with col_a:
        st.button(
            "🎯 Optimal",
            use_container_width=True,
            on_click=set_alpha_optimal,
            help=f"Set α to optimal value ({optimal_alpha:.2f})"
        )
    with col_b:
        st.button(
            "🔄 Reset",
            use_container_width=True,
            on_click=set_alpha_reset,
            help="Reset to α = 0.5"
        )

    st.divider()

    # Display options
    st.markdown("### 📋 Display Options")
    show_baselines = st.checkbox(
        "Show baseline models",
        value=True,
        help="Include Last-Click, First-Click, Linear, Time-Decay, Position-Based"
    )

    show_proposed = st.checkbox(
        "Show proposed models",
        value=True,
        help="Include Markov, Shapley, and Hybrid"
    )

    st.divider()

    # Info section
    with st.expander("ℹ️ About this page"):
        st.markdown("""
        This page compares **8 attribution models** against the
        Monte Carlo ground truth distribution.

        **Metrics used:**
        - **MAE** (Mean Absolute Error): Average magnitude error
        - **RMSE**: Root mean squared error
        - **Spearman ρ**: Rank correlation

        **The Hybrid model** combines Markov (good at ranking) with
        Shapley (good at magnitude) to achieve the best balance.
        """)


# ============================================================================
# COMPUTE CURRENT METRICS
# ============================================================================

# Pre-computed grid for alpha curve
alpha_grid = compute_alpha_grid(n_points=101)

# Current alpha's MAE
current_mae = compute_mae_for_alpha(alpha)

# Hybrid attribution at current alpha
hybrid_df = compute_hybrid_for_alpha(alpha)

# All attributions
all_attr = load_all_attributions_normalized()

# Static metrics from CSV
final_metrics = load_final_metrics()
last_click_mae = float(final_metrics[final_metrics['Model'] == 'Last_Click']['MAE'].values[0])

# Compute improvement
improvement = (last_click_mae - current_mae) / last_click_mae * 100


# ============================================================================
# KEY METRICS ROW
# ============================================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Current Hybrid MAE",
        value=f"{current_mae:.4f}",
        help="Mean Absolute Error against ground truth at the selected α"
    )

with col2:
    delta_label = f"{improvement:+.1f}% vs Last-Click"
    st.metric(
        label="Improvement",
        value=f"{abs(improvement):.1f}%",
        delta=delta_label,
        delta_color="inverse" if improvement > 0 else "off",
        help="How much better (lower MAE) than Last-Click baseline"
    )

with col3:
    st.metric(
        label="Selected α",
        value=f"{alpha:.2f}",
        delta=f"Optimal: {optimal_alpha:.2f}",
        delta_color="off",
        help="Current mixing parameter value"
    )


# ============================================================================
# DYNAMIC INSIGHT MESSAGE
# ============================================================================

if abs(alpha - optimal_alpha) < 0.03:
    st.markdown("""
    <div class="insight-box insight-box-optimal">
    🎯 <strong>You're at the optimal α!</strong>
    This combination minimizes MAE against the ground truth.
    Hybrid achieves the best balance between magnitude accuracy
    (Shapley's strength) and rank identification (Markov's strength).
    </div>
    """, unsafe_allow_html=True)
elif alpha < 0.1:
    st.markdown("""
    <div class="insight-box insight-box-warning">
    📐 <strong>Pure Shapley territory.</strong>
    Magnitudes are well-calibrated, but the ranking is compressed.
    Notice the channels' attribution percentages are close together.
    </div>
    """, unsafe_allow_html=True)
elif alpha > 0.9:
    st.markdown("""
    <div class="insight-box insight-box-warning">
    📈 <strong>Pure Markov territory.</strong>
    Excellent rank identification, but magnitudes are exaggerated.
    The top channel gets too much credit, the bottom too little.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="insight-box">
    ⚖️ <strong>Mixed regime (α = {alpha:.2f}).</strong>
    This combination blends both approaches. Try α = {optimal_alpha:.2f}
    to see the empirically optimal balance.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN BAR CHART
# ============================================================================

st.markdown("## 🎨 Attribution Distributions")

# Build the comparison DataFrame with current hybrid
display_df = all_attr.copy()
display_df['Hybrid_M_S'] = hybrid_df['Hybrid_M_S'].values

# Determine which models to show
models_to_show = ['Ground_Truth']
if show_proposed:
    models_to_show += ['Hybrid_M_S', 'Markov', 'Shapley']
if show_baselines:
    models_to_show += ['Last_Click', 'First_Click', 'Linear',
                       'Time_Decay', 'Position_Based']

fig_bars = plot_attribution_bars(
    display_df,
    models_to_show=models_to_show,
    title=f'Attribution Comparison (α = {alpha:.2f})'
)
st.plotly_chart(fig_bars, use_container_width=True)


# ============================================================================
# ALPHA CURVE
# ============================================================================

st.markdown("## 📈 Alpha Sensitivity Analysis")

st.markdown("""
The curve below shows how MAE changes as α varies from 0 to 1.
The green dashed line marks the empirically optimal α.
The red dot is your current selection.
""")

fig_alpha = plot_alpha_curve(alpha_grid, alpha, current_mae)
st.plotly_chart(fig_alpha, use_container_width=True)


# ============================================================================
# METRICS TABLE
# ============================================================================

st.markdown("## 📋 Performance Metrics — All Models")

st.markdown("""
The table below ranks all 8 attribution models by MAE (lower is better).
The Hybrid row updates dynamically based on your α selection.
""")

# Build dynamic metrics table
metrics_display = final_metrics.copy()

# Override Hybrid row with current alpha's MAE
hybrid_mask = metrics_display['Model'] == 'Hybrid_M_S'
if hybrid_mask.any():
    metrics_display.loc[hybrid_mask, 'MAE'] = round(current_mae, 4)

# Sort by MAE
metrics_display = metrics_display.sort_values('MAE').reset_index(drop=True)
metrics_display.index = metrics_display.index + 1  # Start from 1

# Add display names
metrics_display['Model'] = metrics_display['Model'].map(
    lambda m: MODEL_INFO.get(m, {'name': m})['name']
)

# Rename columns for display
metrics_display = metrics_display.rename(columns={
    'Spearman_Corr'   : 'Spearman ρ',
    'Ranking_Accuracy': 'Rank Acc',
})

# Style the table
def highlight_hybrid(row):
    if 'Hybrid' in str(row['Model']):
        return ['background-color: rgba(220, 38, 38, 0.2); font-weight: 700;'] * len(row)
    return [''] * len(row)

styled = metrics_display.style.apply(highlight_hybrid, axis=1)

st.dataframe(
    styled,
    use_container_width=True,
    column_config={
        'Model'    : st.column_config.TextColumn('Model', width='medium'),
        'MAE'      : st.column_config.NumberColumn('MAE', format='%.4f'),
        'RMSE'     : st.column_config.NumberColumn('RMSE', format='%.4f'),
        'Spearman ρ': st.column_config.NumberColumn('Spearman ρ', format='%.3f'),
        'Rank Acc' : st.column_config.TextColumn('Rank Acc'),
    },
)


# ============================================================================
# EDUCATIONAL FOOTER
# ============================================================================

st.divider()

with st.expander("📚 How to interpret this page"):
    st.markdown("""
    ### The Markov-Shapley Trade-off

    Each attribution paradigm has fundamental strengths and weaknesses:

    **Markov Chain (Removal Effects)**
    - ✅ Excellent at identifying which channels matter most (highest Spearman ρ = 0.894)
    - ❌ Exaggerates magnitudes due to chain-effect propagation (MAE = 0.1466)

    **Shapley Value (Cooperative Game Theory)**
    - ✅ Mathematically balanced magnitudes via axioms of efficiency and symmetry
    - ❌ Compresses rankings; top channel doesn't stand out enough

    **Hybrid (α × Markov + (1-α) × Shapley)**
    - 🎯 At α = 0.25, achieves both: balanced magnitudes AND clear ranking
    - The optimal α was determined via grid search against ground truth

    ### Why This Matters in Practice

    Real marketing teams need:
    1. **Correct ranking** to prioritize budget allocation decisions
    2. **Correct magnitudes** to know *how much* to invest in each channel

    Classical models like Last-Click fail at both. The hybrid framework
    achieves both with mathematical justification.

    > *"Markov knows which channels matter most but exaggerates how much;
    > Shapley approximates how much each matters but obscures which
    > matters most."*
    """)
