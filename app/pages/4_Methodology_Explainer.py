"""
Page 4: Methodology Explainer
==============================

Educational deep-dive into the four methodological pillars of the
thesis: Markov chain attribution, Shapley value attribution, the
Markov-Shapley trade-off, and diminishing returns in budget
optimization.

Structured as a four-tab page for progressive disclosure.
"""

import sys
import os

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import itertools
import math
import plotly.graph_objects as go

from utils.data_loader import (
    load_transition_matrix,
    load_interaction_matrix,
    load_ground_truth,
    load_markov_results,
    load_shapley_results,
    load_final_metrics,
    CHANNELS,
    CHANNEL_DISPLAY,
    CHANNEL_COLORS,
    MODEL_INFO,
)
from utils.models import (
    response_function,
    SATURATION_PARAMS,
    CHANNEL_EFFECTS,
)
from utils.plotting import apply_dark_theme


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Methodology Explainer",
    page_icon="📚",
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
        color: #F59E0B;
        margin-bottom: 0.5rem;
    }
    .page-subtitle {
        font-size: 1.1rem;
        color: #9CA3AF;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .concept-box {
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
        color: #E5E7EB;
    }
    .concept-box-blue {
        border-left-color: #60A5FA;
    }
    .concept-box-green {
        border-left-color: #10B981;
    }
    .concept-box-red {
        border-left-color: #DC2626;
    }
    .formula-large {
        background-color: #111827;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        margin: 1.5rem 0;
        border: 1px solid #374151;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HEADER
# ============================================================================

st.markdown('<h1 class="page-title">📚 Methodology Explainer</h1>',
            unsafe_allow_html=True)

st.markdown('<p class="page-subtitle">An interactive deep-dive into the mathematical foundations of the hybrid attribution framework</p>',
            unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### 📖 Navigation")
    st.markdown("""
    This page is organized into four tabs:

    1. **🔄 Markov Chain** — Transition matrices and removal effects
    2. **🎲 Shapley Value** — Cooperative game theory for attribution
    3. **⚖️ The Trade-off** — Why neither method alone is sufficient
    4. **📈 Diminishing Returns** — The budget optimization layer
    """)

    st.divider()

    st.markdown("### 🎓 Audience")
    st.markdown("""
    Designed to be accessible to:
    - **Marketing practitioners** with basic statistical background
    - **Students** in marketing analytics or data science
    - **Researchers** seeking a quick refresher on attribution methods
    """)

    st.divider()

    with st.expander("📚 Recommended Reading"):
        st.markdown("""
        **Markov attribution:**
        - Anderl et al. (2016)

        **Shapley value:**
        - Shapley (1953)
        - Shao & Li (2011)

        **Marketing Mix Modeling:**
        - Jin et al. (2017)
        - Chan & Perry (2017)

        **Differential Evolution:**
        - Storn & Price (1997)
        """)


# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔄 Markov Chain",
    "🎲 Shapley Value",
    "⚖️ The Trade-off",
    "📈 Diminishing Returns"
])


# ============================================================================
# TAB 1: MARKOV CHAIN
# ============================================================================

with tab1:
    st.markdown("## Markov Chain Attribution")

    st.markdown("""
    <div class="concept-box concept-box-blue">
    <strong>Core idea:</strong> Model the customer journey as a sequence of
    state transitions. To measure a channel's importance, simulate its
    removal and observe how much overall conversion probability drops.
    </div>
    """, unsafe_allow_html=True)

    # ========================================================================
    # STEP 1: TRANSITION MATRIX
    # ========================================================================

    st.markdown("### Step 1: Build the Transition Matrix")

    st.markdown("""
    From observed customer journeys, count how often each channel
    follows each other channel. After **Laplace smoothing** (adding
    α=1.0 to all valid transitions), normalize each row to obtain
    transition probabilities.
    """)

    # Load and display transition matrix
    trans_mat = load_transition_matrix()

    # Heatmap of transition matrix
    fig_trans = go.Figure(data=go.Heatmap(
        z=trans_mat.values,
        x=trans_mat.columns,
        y=trans_mat.index,
        colorscale='Blues',
        text=[[f"{v:.3f}" for v in row] for row in trans_mat.values],
        texttemplate='%{text}',
        textfont={'size': 10, 'color': 'white'},
        hovertemplate='From: %{y}<br>To: %{x}<br>Probability: %{z:.4f}<extra></extra>',
        colorbar={'title': 'P(transition)'},
    ))

    fig_trans.update_layout(
        title='Markov Chain Transition Matrix (8 × 8)',
        xaxis_title='To State',
        yaxis_title='From State',
        height=550,
    )

    st.plotly_chart(apply_dark_theme(fig_trans), use_container_width=True)

    with st.expander("🔍 How to read this matrix"):
        st.markdown("""
        - **Rows:** "From" state (where the user currently is)
        - **Columns:** "To" state (where the user goes next)
        - **Values:** Probability of that transition
        - **Each row sums to 1.0** (a state must transition somewhere)

        **Special states:**
        - `Start`: Virtual state from which all journeys begin
        - `Conversion`: Absorbing state for successful conversions
        - `Non-Conversion`: Absorbing state for dropouts

        Absorbing states have probability 1.0 of transitioning to themselves
        (you can see the diagonal values for Conversion and Non-Conversion).
        """)

    # ========================================================================
    # STEP 2: ABSORPTION PROBABILITIES
    # ========================================================================

    st.markdown("### Step 2: Compute Conversion Probabilities")

    st.markdown("""
    Using the fundamental matrix of absorbing Markov chains, we compute
    the probability that a user starting from each state eventually
    converts:
    """)

    st.latex(r"N = (I - Q)^{-1}")
    st.latex(r"B = N \cdot R")

    st.markdown("""
    Where:
    - $Q$ is the transition matrix among **transient** states (Start + channels)
    - $R$ is the transition matrix from transient to **absorbing** states
    - $N$ is the **fundamental matrix** (expected visits)
    - $B$ contains **absorption probabilities** — the chance of ending in each absorbing state

    The first row of $B$ tells us the **baseline conversion probability**
    starting from the Start state.
    """)

    # ========================================================================
    # STEP 3: REMOVAL EFFECT
    # ========================================================================

    st.markdown("### Step 3: Compute Removal Effects")

    st.markdown("""
    For each channel, **temporarily remove** it from the transition matrix,
    renormalize the remaining rows, and recompute the baseline conversion
    probability. The relative drop is the channel's **removal effect**:
    """)

    st.latex(r"\text{RE}(c) = \frac{P_{\text{base}} - P_{\text{without } c}}{P_{\text{base}}}")

    st.markdown("""
    A channel with a **large removal effect** is one whose absence
    substantially reduces overall conversion — meaning it's a key player
    in the journey graph.
    """)

    # Display actual removal effects from the thesis
    markov_df = load_markov_results()

    fig_re = go.Figure()
    sorted_df = markov_df.sort_values('markov_removal_effect', ascending=True)

    fig_re.add_trace(go.Bar(
        y=sorted_df['channel'].map(CHANNEL_DISPLAY),
        x=sorted_df['markov_removal_effect'] * 100,
        orientation='h',
        marker_color=[CHANNEL_COLORS[ch] for ch in sorted_df['channel']],
        text=[f"{v*100:.2f}%" for v in sorted_df['markov_removal_effect']],
        textposition='outside',
        textfont={'size': 12, 'color': 'white'},
        hovertemplate='<b>%{y}</b><br>Removal Effect: %{x:.2f}%<extra></extra>',
    ))

    fig_re.update_layout(
        title='Markov Removal Effects (from thesis data)',
        xaxis_title='Removal Effect (%)',
        yaxis_title='',
        height=400,
        showlegend=False,
    )

    st.plotly_chart(apply_dark_theme(fig_re), use_container_width=True)

    # ========================================================================
    # KEY INSIGHT
    # ========================================================================

    st.markdown("""
    <div class="concept-box concept-box-green">
    <strong>Strength of Markov:</strong> Excellent at identifying which
    channels are most critical to the journey graph. Spearman ρ = 0.894
    in the thesis evaluation (highest of all models).
    <br><br>
    <strong>Weakness of Markov:</strong> The propagation effect causes
    magnitudes to be exaggerated — the top channel's removal effect is
    disproportionately larger than its true contribution. MAE = 0.1466
    (worst of all models).
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# TAB 2: SHAPLEY VALUE
# ============================================================================

with tab2:
    st.markdown("## Shapley Value Attribution")

    st.markdown("""
    <div class="concept-box concept-box-blue">
    <strong>Core idea:</strong> Treat channels as players in a cooperative
    game. The value each channel "deserves" is its average marginal
    contribution across all possible coalitions, weighted by combinatorial
    factors.
    </div>
    """, unsafe_allow_html=True)

    # ========================================================================
    # THE SHAPLEY FORMULA
    # ========================================================================

    st.markdown("### The Shapley Formula")

    st.latex(r"\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! \, (n - |S| - 1)!}{n!} \left[ v(S \cup \{i\}) - v(S) \right]")

    st.markdown("""
    Where:
    - $\\phi_i$ = Shapley value (attribution) for channel $i$
    - $N$ = set of all channels (5 in our case)
    - $S$ = a subset of channels not containing $i$ ("coalition")
    - $v(S)$ = characteristic function: total "value" produced by coalition $S$
    - $v(S \\cup \\{i\\}) - v(S)$ = **marginal contribution** of channel $i$ to coalition $S$
    - $\\frac{|S|! (n-|S|-1)!}{n!}$ = combinatorial weighting

    The Shapley value averages a channel's contribution across **every possible
    permutation** of channel arrivals, weighted by how often each permutation occurs.
    """)

    # ========================================================================
    # COALITION ENUMERATION
    # ========================================================================

    st.markdown("### How Many Coalitions?")

    st.markdown("""
    For 5 channels, we have $2^5 = 32$ possible coalitions (including the
    empty set). The Shapley value for one channel requires evaluating the
    marginal contribution across **16 coalitions** (those not containing
    that channel).
    """)

    # Show all coalitions
    n_channels = 5
    coalition_sizes = list(range(n_channels + 1))
    n_coalitions = [math.comb(n_channels, k) for k in coalition_sizes]

    fig_coalitions = go.Figure()
    fig_coalitions.add_trace(go.Bar(
        x=[f"Size {k}" for k in coalition_sizes],
        y=n_coalitions,
        marker_color='#F59E0B',
        text=n_coalitions,
        textposition='outside',
        textfont={'size': 14, 'color': 'white'},
        hovertemplate='Coalition size: %{x}<br>Count: %{y}<extra></extra>',
    ))

    fig_coalitions.update_layout(
        title='Coalition Counts by Size (n = 5 channels)',
        xaxis_title='Coalition Size',
        yaxis_title='Number of Coalitions',
        height=350,
        showlegend=False,
    )

    st.plotly_chart(apply_dark_theme(fig_coalitions), use_container_width=True)

    st.caption(f"Total: {sum(n_coalitions)} coalitions (sum = $2^5$ = 32)")

    # ========================================================================
    # COMPLEXITY WARNING
    # ========================================================================

    st.markdown("### ⚠️ The Combinatorial Explosion")

    st.markdown("""
    Shapley value computation has **exponential complexity** in the number
    of channels. Here's the cost:
    """)

    complexity_data = []
    for n in [3, 5, 7, 10, 15, 20]:
        complexity_data.append({
            'Channels (n)': n,
            'Total Coalitions ($2^n$)': f"{2**n:,}",
            'Practical?': '✅ Easy' if n <= 5 else ('⚠️ Slow' if n <= 10 else '❌ Infeasible')
        })

    complexity_df = pd.DataFrame(complexity_data)

    st.dataframe(
        complexity_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("""
    This is why most marketing attribution implementations use **5-10
    channel aggregations**. Beyond that, approximation methods (Monte Carlo
    sampling, neural-network Shapley) become necessary.
    """)

    # ========================================================================
    # THE FOUR AXIOMS
    # ========================================================================

    st.markdown("### The Four Shapley Axioms")

    st.markdown("""
    The Shapley value is the **unique** allocation that satisfies these
    four axioms simultaneously:
    """)

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("""
        <div class="concept-box concept-box-blue">
        <strong>1. Efficiency</strong><br>
        The total value $v(N)$ is fully distributed:
        $\\sum_i \\phi_i = v(N)$
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="concept-box concept-box-blue">
        <strong>2. Symmetry</strong><br>
        Two players with identical marginal contributions receive
        equal Shapley values.
        </div>
        """, unsafe_allow_html=True)

    with col_a2:
        st.markdown("""
        <div class="concept-box concept-box-blue">
        <strong>3. Null Player</strong><br>
        A player that contributes nothing to any coalition receives
        zero Shapley value.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="concept-box concept-box-blue">
        <strong>4. Additivity</strong><br>
        For combined games, Shapley values add up:
        $\\phi_i(v + w) = \\phi_i(v) + \\phi_i(w)$
        </div>
        """, unsafe_allow_html=True)

    # ========================================================================
    # FRACTIONAL CREDIT
    # ========================================================================

    st.markdown("### Fractional Credit (Shao & Li, 2011)")

    st.markdown("""
    The naive characteristic function $v(S) = $ "number of converting paths
    containing any channel in $S$" gives too much credit to frequently
    occurring channels.

    The **fractional credit** formulation fixes this:
    """)

    st.latex(r"v(S) = \sum_{u \in \text{Converters}} \frac{|S \cap \text{path}_u|}{|\text{path}_u|}")

    st.markdown("""
    Each converting user contributes to a coalition's value **proportionally
    to the overlap** between their journey and the coalition. This:
    - Preserves the Shapley efficiency axiom
    - Prevents inflation of frequently observed channels
    - Captures path density effects
    """)

    # Display Shapley results
    shapley_df = load_shapley_results()

    fig_shap = go.Figure()
    sorted_shap = shapley_df.copy()
    sorted_shap['shapley_norm'] = sorted_shap['shapley_value'] / sorted_shap['shapley_value'].sum()
    sorted_shap = sorted_shap.sort_values('shapley_norm', ascending=True)

    fig_shap.add_trace(go.Bar(
        y=sorted_shap['channel'].map(CHANNEL_DISPLAY),
        x=sorted_shap['shapley_norm'] * 100,
        orientation='h',
        marker_color=[CHANNEL_COLORS[ch] for ch in sorted_shap['channel']],
        text=[f"{v*100:.2f}%" for v in sorted_shap['shapley_norm']],
        textposition='outside',
        textfont={'size': 12, 'color': 'white'},
        hovertemplate='<b>%{y}</b><br>Shapley share: %{x:.2f}%<extra></extra>',
    ))

    fig_shap.update_layout(
        title='Shapley Value Distribution (from thesis data)',
        xaxis_title='Normalized Shapley Share (%)',
        yaxis_title='',
        height=400,
        showlegend=False,
    )

    st.plotly_chart(apply_dark_theme(fig_shap), use_container_width=True)

    # ========================================================================
    # KEY INSIGHT
    # ========================================================================

    st.markdown("""
    <div class="concept-box concept-box-green">
    <strong>Strength of Shapley:</strong> Mathematically balanced magnitudes
    by construction. MAE = 0.0810 in the thesis (much better than Markov's
    0.1466).
    <br><br>
    <strong>Weakness of Shapley:</strong> The averaging across permutations
    compresses ranking signal. Spearman ρ = 0.100 (much worse than Markov's
    0.894). Channels' relative ordering becomes ambiguous.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# TAB 3: THE TRADE-OFF
# ============================================================================

with tab3:
    st.markdown("## The Markov-Shapley Trade-off")

    st.markdown("""
    <div class="concept-box concept-box-red">
    <strong>The fundamental insight of this thesis:</strong> Markov and
    Shapley attribution are <em>complementary</em>, not competing.
    Neither alone is sufficient — but their convex combination outperforms
    either.
    </div>
    """, unsafe_allow_html=True)

    # ========================================================================
    # SIDE-BY-SIDE COMPARISON
    # ========================================================================

    st.markdown("### Side-by-Side Comparison")

    metrics_df = load_final_metrics()

    # Build comparison table for just the three key models
    key_models = ['Markov', 'Shapley', 'Hybrid_M_S']
    key_metrics = metrics_df[metrics_df['Model'].isin(key_models)].copy()

    # Add display names
    key_metrics['Model'] = key_metrics['Model'].map(
        lambda m: MODEL_INFO[m]['name']
    )

    # Reorder
    order = ['Markov Chain', 'Shapley Value', 'Hybrid (M+S)']
    key_metrics['_order'] = key_metrics['Model'].map(
        lambda m: order.index(m) if m in order else 99
    )
    key_metrics = key_metrics.sort_values('_order').drop(columns='_order')

    # Rename columns
    key_metrics = key_metrics.rename(columns={
        'Spearman_Corr'   : 'Spearman ρ',
        'Ranking_Accuracy': 'Rank Accuracy',
    })

    def highlight_hybrid_row(row):
        if 'Hybrid' in str(row['Model']):
            return ['background-color: rgba(220, 38, 38, 0.2); font-weight: 700;'] * len(row)
        return [''] * len(row)

    styled = key_metrics.style.apply(highlight_hybrid_row, axis=1)

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Model'        : st.column_config.TextColumn('Model', width='medium'),
            'MAE'          : st.column_config.NumberColumn('MAE', format='%.4f'),
            'RMSE'         : st.column_config.NumberColumn('RMSE', format='%.4f'),
            'Spearman ρ'   : st.column_config.NumberColumn('Spearman ρ', format='%.3f'),
            'Rank Accuracy': st.column_config.TextColumn('Rank Acc'),
        },
    )

    # ========================================================================
    # VISUAL TRADE-OFF
    # ========================================================================

    st.markdown("### The Trade-off Visualized")

    st.markdown("""
    Plotting MAE (lower is better) against Spearman correlation (higher is
    better), each model occupies a distinct region:
    """)

    # Trade-off scatter plot
    fig_tradeoff = go.Figure()

    for model in ['Markov', 'Shapley', 'Hybrid_M_S']:
        row = metrics_df[metrics_df['Model'] == model].iloc[0]
        info = MODEL_INFO[model]

        fig_tradeoff.add_trace(go.Scatter(
            x=[float(row['MAE'])],
            y=[float(row['Spearman_Corr'])],
            mode='markers+text',
            marker={
                'size': 30,
                'color': info['color'],
                'line': {'color': 'white', 'width': 3},
                'symbol': 'star' if model == 'Hybrid_M_S' else 'circle',
            },
            text=[info['name']],
            textposition='top right',
            textfont={'size': 14, 'color': 'white'},
            name=info['name'],
            hovertemplate=f"<b>{info['name']}</b><br>" +
                          "MAE: %{x:.4f}<br>" +
                          "Spearman ρ: %{y:.3f}<extra></extra>",
        ))

    # Add ideal corner annotation
    fig_tradeoff.add_annotation(
        x=0.04,
        y=1.0,
        text="🎯 Ideal:<br>Low MAE,<br>High ρ",
        showarrow=False,
        font={'size': 13, 'color': '#10B981'},
        bgcolor='rgba(16, 185, 129, 0.1)',
        bordercolor='#10B981',
        borderwidth=1,
        borderpad=8,
    )

    fig_tradeoff.update_layout(
        title='Magnitude Accuracy (MAE) vs. Ranking Quality (Spearman ρ)',
        xaxis_title='MAE (lower is better)',
        yaxis_title='Spearman ρ (higher is better)',
        height=500,
        showlegend=False,
        xaxis={'range': [0, 0.16]},
        yaxis={'range': [0, 1.05]},
    )

    st.plotly_chart(apply_dark_theme(fig_tradeoff), use_container_width=True)

    # ========================================================================
    # THE INSIGHT
    # ========================================================================

    st.markdown("### 💡 The Hybrid Solution")

    st.markdown("""
    The hybrid model combines the two via convex linear combination:
    """)

    st.latex(r"w_c^{\text{Hybrid}} = \alpha \cdot w_c^{\text{Markov}} + (1 - \alpha) \cdot w_c^{\text{Shapley}}")

    st.markdown("""
    With $\\alpha^* = 0.25$ (determined by grid search against ground truth):

    - **25% Markov weight:** Inherits the ranking signal
    - **75% Shapley weight:** Inherits the magnitude calibration

    The hybrid achieves:
    - **MAE = 0.0571** (lower than either component)
    - **Spearman ρ = 0.600** (much better than pure Shapley)

    This is a textbook example of how **principled combination of methods**
    can outperform either individually.
    """)

    # ========================================================================
    # THE QUOTE
    # ========================================================================

    st.markdown("""
    <div class="concept-box concept-box-red" style="text-align: center; font-size: 1.1rem; font-style: italic;">
    "Markov knows which channels matter most but exaggerates how much;<br>
    Shapley approximates how much each matters but obscures which<br>
    matters most. Hybrid resolves both."
    <br><br>
    <span style="font-style: normal; font-size: 0.9rem; color: #9CA3AF;">
    — Thesis Chapter 5
    </span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# TAB 4: DIMINISHING RETURNS
# ============================================================================

with tab4:
    st.markdown("## Diminishing Returns in Budget Optimization")

    st.markdown("""
    <div class="concept-box concept-box-blue">
    <strong>Core idea:</strong> Marketing channels saturate — the first
    dollar spent produces more response than the millionth. Optimal
    allocation requires modeling this non-linearity.
    </div>
    """, unsafe_allow_html=True)

    # ========================================================================
    # THE FORMULA
    # ========================================================================

    st.markdown("### The Exponential Saturation Function")

    st.latex(r"\text{Response}(x) = \text{score} \cdot \left(1 - e^{-kx/x_{\text{half}}}\right)")

    st.markdown("""
    Where:
    - $x$ = dollars spent on a channel
    - $\\text{score}$ = the attribution weight from the hybrid model
    - $k$ = **saturation rate** — how quickly diminishing returns kick in
    - $x_{\\text{half}}$ = the spend at which response reaches half of its maximum

    This functional form is widely used in **Marketing Mix Modeling**
    (MMM) — see Jin et al. (2017) for the Bayesian formulation used by
    Google.
    """)

    # ========================================================================
    # INTERACTIVE EXPLORER
    # ========================================================================

    st.markdown("### 🎚️ Interactive Curve Explorer")

    st.markdown("""
    Adjust the parameters below to see how they shape the response curve.
    """)

    col_k, col_h = st.columns(2)

    with col_k:
        # Initialize k state
        if 'k_demo' not in st.session_state:
            st.session_state.k_demo = 2.0

        k_demo = st.slider(
            "Saturation rate (k)",
            min_value=0.5,
            max_value=5.0,
            step=0.1,
            key='k_demo',
            help="Higher k = faster saturation"
        )

    with col_h:
        # Initialize half_sat state
        if 'half_sat_demo' not in st.session_state:
            st.session_state.half_sat_demo = 20_000

        half_sat_demo = st.slider(
            "Half-saturation point (USD)",
            min_value=2_000,
            max_value=100_000,
            step=2_000,
            key='half_sat_demo',
            help="Spend level at which response reaches half its max"
        )

    # Compute curve
    spend_range = np.linspace(0, 200_000, 200)
    score_demo = 1.0  # normalized
    responses = [response_function(x, score_demo, k_demo, half_sat_demo)
                 for x in spend_range]

    fig_curve = go.Figure()

    fig_curve.add_trace(go.Scatter(
        x=spend_range,
        y=responses,
        mode='lines',
        line={'color': '#F59E0B', 'width': 4},
        fill='tozeroy',
        fillcolor='rgba(245, 158, 11, 0.1)',
        name='Response',
        hovertemplate='Spend: $%{x:,.0f}<br>Response: %{y:.3f}<extra></extra>',
    ))

    # Mark half-saturation point
    half_resp = response_function(half_sat_demo, score_demo, k_demo, half_sat_demo)
    fig_curve.add_trace(go.Scatter(
        x=[half_sat_demo],
        y=[half_resp],
        mode='markers+text',
        marker={
            'size': 18,
            'color': '#DC2626',
            'symbol': 'diamond',
            'line': {'color': 'white', 'width': 2},
        },
        text=[f"  Half-saturation<br>  ${half_sat_demo:,.0f}"],
        textposition='middle right',
        textfont={'size': 12, 'color': 'white'},
        name='Half-saturation',
        hovertemplate=f'Half-saturation: ${half_sat_demo:,.0f}<br>Response: {half_resp:.3f}<extra></extra>',
    ))

    fig_curve.add_hline(
        y=0.5,
        line_dash='dash',
        line_color='#9CA3AF',
        annotation_text='50% response',
        annotation_position='right',
    )

    fig_curve.update_layout(
        title=f'Response Curve: k = {k_demo:.1f}, half-saturation = ${half_sat_demo:,.0f}',
        xaxis_title='Channel Spend (USD)',
        yaxis_title='Expected Response',
        height=450,
        showlegend=False,
        yaxis={'range': [0, 1.05]},
    )

    fig_curve.update_xaxes(tickformat=',d')

    st.plotly_chart(apply_dark_theme(fig_curve), use_container_width=True)

    # ========================================================================
    # CHANNEL-SPECIFIC PARAMETERS
    # ========================================================================

    st.markdown("### Channel-Specific Parameters in the Thesis")

    st.markdown("""
    Each channel has its own saturation profile, calibrated from
    industry MMM benchmarks:
    """)

    params_data = []
    for ch, params in SATURATION_PARAMS.items():
        params_data.append({
            'Channel': CHANNEL_DISPLAY[ch],
            'Saturation rate (k)': params['k'],
            'Half-saturation (USD)': params['half_saturation'],
            'Interpretation': {
                'Google_Ads'    : 'Slow saturation — continuous flow of new searches',
                'Email'         : 'Rapid saturation — fixed subscriber list',
                'Social_Media'  : 'Moderate — reach vs. irritation trade-off',
                'Direct'        : 'Very rapid — limited audience',
                'Organic_Search': 'Slow — SEO investments yield gradual returns',
            }.get(ch, '')
        })

    params_df = pd.DataFrame(params_data)

    st.dataframe(
        params_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Channel'              : st.column_config.TextColumn('Channel'),
            'Saturation rate (k)'  : st.column_config.NumberColumn('k', format='%.1f'),
            'Half-saturation (USD)': st.column_config.NumberColumn('Half-sat ($)', format='$%d'),
            'Interpretation'       : st.column_config.TextColumn('Interpretation', width='large'),
        }
    )

    # ========================================================================
    # ALL CHANNELS PLOTTED TOGETHER
    # ========================================================================

    st.markdown("### Response Curves: All Channels")

    fig_all = go.Figure()

    spend_range = np.linspace(0, 100_000, 200)

    for ch in ['Google_Ads', 'Email', 'Social_Media']:
        params = SATURATION_PARAMS[ch]
        score = CHANNEL_EFFECTS[ch]  # use thesis values
        responses = [response_function(x, score, params['k'], params['half_saturation'])
                     for x in spend_range]

        fig_all.add_trace(go.Scatter(
            x=spend_range,
            y=responses,
            mode='lines',
            line={'color': CHANNEL_COLORS[ch], 'width': 3},
            name=CHANNEL_DISPLAY[ch],
            hovertemplate=f'<b>{CHANNEL_DISPLAY[ch]}</b><br>' +
                          'Spend: $%{x:,.0f}<br>' +
                          'Response: %{y:.3f}<extra></extra>',
        ))

    fig_all.update_layout(
        title='Paid Channel Response Curves (Thesis Parameters)',
        xaxis_title='Channel Spend (USD)',
        yaxis_title='Expected Response',
        height=450,
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.05,
            'xanchor': 'right',
            'x': 1.0,
        }
    )

    fig_all.update_xaxes(tickformat=',d')

    st.plotly_chart(apply_dark_theme(fig_all), use_container_width=True)

    st.markdown("""
    Notice how **Email saturates rapidly** (small subscriber base) while
    **Google Ads** continues to absorb investment for longer. This is why
    optimal allocation isn't trivially "give the best-scoring channel
    everything" — the marginal return shape matters as much as the
    attribution score.
    """)

    # ========================================================================
    # WHY DIFFERENTIAL EVOLUTION
    # ========================================================================

    st.markdown("### Why Differential Evolution?")

    st.markdown("""
    <div class="concept-box concept-box-green">
    The response surface from combining diminishing-returns curves with
    constraints is <strong>non-convex</strong>. Gradient-based methods
    like SLSQP can get stuck in local optima.
    <br><br>
    <strong>Differential Evolution</strong> (Storn & Price, 1997) is a
    population-based global optimizer that:
    <ul>
        <li>Requires no gradient information</li>
        <li>Explores the entire feasible region</li>
        <li>Converges reliably on non-convex problems</li>
        <li>Handles bounded constraints natively</li>
    </ul>
    These properties make it the standard choice in MMM literature.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# FOOTER
# ============================================================================

st.divider()

st.markdown("""
<div style="text-align: center; color: #9CA3AF; padding: 1rem;">
    📖 For full mathematical derivations and proofs, see Chapter 3 of the thesis.<br>
    💻 <a href="https://github.com/brky48/hybrid-mta-attribution" style="color: #60A5FA;">View source code on GitHub</a>
</div>
""", unsafe_allow_html=True)
