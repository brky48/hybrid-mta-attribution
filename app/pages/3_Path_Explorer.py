"""
Page 3: Path Explorer
======================

Interactive customer journey builder. Users construct paths by
clicking channel buttons; the page computes conversion probability
under the ground truth model and shows how each attribution model
assigns credit to the channels in the path.
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
    CHANNELS,
    CHANNEL_DISPLAY,
    CHANNEL_COLORS,
    MODEL_INFO,
)
from utils.models import (
    calculate_conversion_probability,
    compute_path_attributions,
    CHANNEL_EFFECTS,
    CHANNEL_INTERACTIONS,
    BASE_CONVERSION_RATE,
)
from utils.plotting import (
    plot_path_diagram,
    plot_path_attribution_heatmap,
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Path Explorer",
    page_icon="🛤️",
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
        color: #10B981;
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
        border-left: 4px solid #10B981;
        margin: 1rem 0;
        color: #E5E7EB;
    }
    .formula-block {
        background-color: #1F2937;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        color: #D1D5DB;
        border: 1px solid #374151;
    }
    .path-display {
        background-color: #1F2937;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        font-size: 1.2rem;
        color: #E5E7EB;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HEADER
# ============================================================================

st.markdown('<h1 class="page-title">🛤️ Path Explorer</h1>',
            unsafe_allow_html=True)

st.markdown('<p class="page-subtitle">Build a customer journey and see how each attribution model interprets it</p>',
            unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'path' not in st.session_state:
    st.session_state.path = []


# ============================================================================
# CALLBACK FUNCTIONS
# ============================================================================

def add_channel(channel):
    """Add a channel to the path (max 8 touchpoints)."""
    if len(st.session_state.path) < 8:
        st.session_state.path.append(channel)


def remove_last():
    """Remove the last channel from the path."""
    if st.session_state.path:
        st.session_state.path.pop()


def clear_path():
    """Clear the entire path."""
    st.session_state.path = []


def preset_awareness():
    st.session_state.path = ['Social_Media', 'Organic_Search', 'Google_Ads']


def preset_decision():
    st.session_state.path = ['Google_Ads', 'Direct']


def preset_loyal():
    st.session_state.path = ['Email', 'Organic_Search', 'Email', 'Direct']


def preset_chaotic():
    st.session_state.path = ['Social_Media', 'Organic_Search', 'Google_Ads',
                              'Email', 'Direct']


def preset_single_social():
    st.session_state.path = ['Social_Media']


# ============================================================================
# SIDEBAR — CONTROLS
# ============================================================================

with st.sidebar:
    st.markdown("### 🎮 Path Builder")

    st.markdown("**Add a channel:**")

    for ch in CHANNELS:
        col_emoji, col_btn = st.columns([1, 4])
        with col_emoji:
            st.markdown(
                f'<div style="width: 30px; height: 30px; background-color: '
                f'{CHANNEL_COLORS[ch]}; border-radius: 4px; margin-top: 0.3rem;"></div>',
                unsafe_allow_html=True
            )
        with col_btn:
            st.button(
                CHANNEL_DISPLAY[ch],
                key=f'btn_add_{ch}',
                on_click=add_channel,
                args=(ch,),
                use_container_width=True,
            )

    st.divider()

    st.markdown("**Path actions:**")
    col_back, col_clear = st.columns(2)
    with col_back:
        st.button(
            "⬅️ Remove last",
            on_click=remove_last,
            use_container_width=True,
            disabled=len(st.session_state.path) == 0,
        )
    with col_clear:
        st.button(
            "🗑️ Clear all",
            on_click=clear_path,
            use_container_width=True,
            disabled=len(st.session_state.path) == 0,
        )

    st.divider()

    st.markdown("### 📋 Try These Journeys")

    st.button("🎯 Awareness Journey",
              on_click=preset_awareness,
              use_container_width=True,
              help="Social Media → Organic Search → Google Ads")

    st.button("💡 Decision Journey",
              on_click=preset_decision,
              use_container_width=True,
              help="Google Ads → Direct (high intent)")

    st.button("💌 Loyal Customer",
              on_click=preset_loyal,
              use_container_width=True,
              help="Email → Organic → Email → Direct")

    st.button("🌪️ Chaotic Path",
              on_click=preset_chaotic,
              use_container_width=True,
              help="Long 5-touch journey")

    st.button("📱 Social-Only (Penalty)",
              on_click=preset_single_social,
              use_container_width=True,
              help="Single social touchpoint (low conversion)")

    st.divider()

    with st.expander("ℹ️ About this page"):
        st.markdown("""
        This page lets you build customer journeys and see:

        1. **Ground truth conversion probability** computed by the
           data-generating model
        2. **Per-model attribution** showing how each algorithm
           assigns credit across the path's channels

        **Maximum path length:** 8 touchpoints

        **Key insight:** Different models give wildly different answers
        for the same path. Last-Click always credits the final
        channel; Linear divides equally; Position-Based emphasizes
        first and last.
        """)


# ============================================================================
# MAIN CONTENT — PATH DISPLAY
# ============================================================================

path = st.session_state.path

if not path:
    st.info("""
    👋 **Welcome to the Path Explorer!**

    Use the channel buttons in the sidebar to build a customer journey,
    or try one of the preset journeys to get started.

    The page will then show:
    - The true conversion probability (ground truth)
    - How each attribution model credits the channels in your path
    """)
else:
    path_str = " → ".join([CHANNEL_DISPLAY[ch] for ch in path])
    st.markdown(
        f'<div class="path-display">📍 <strong>Current journey:</strong> {path_str}</div>',
        unsafe_allow_html=True
    )

    fig_path = plot_path_diagram(path)
    st.plotly_chart(fig_path, use_container_width=True)


# ============================================================================
# CONVERSION PROBABILITY COMPUTATION
# ============================================================================

if path:
    prob_result = calculate_conversion_probability(path)

    # ========================================================================
    # KEY METRICS
    # ========================================================================

    st.markdown("## 🎯 Ground Truth Conversion Probability")

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    with col_p1:
        st.metric(
            label="Conversion Probability",
            value=f"{prob_result['probability'] * 100:.3f}%",
            help="True probability under the ground truth model"
        )

    with col_p2:
        st.metric(
            label="Path Length",
            value=len(path),
            help="Number of touchpoints in this journey"
        )

    with col_p3:
        unique_channels = len(set(path))
        st.metric(
            label="Unique Channels",
            value=f"{unique_channels}/5",
            help="How many distinct channels appear in this path"
        )

    with col_p4:
        relative = prob_result['probability'] / 0.0118 * 100
        st.metric(
            label="vs. Avg Path",
            value=f"{relative:.1f}%",
            delta=f"{relative - 100:+.1f}%",
            delta_color="normal",
            help="Relative to the 1.18% average conversion rate in the dataset"
        )

    # ========================================================================
    # FORMULA BREAKDOWN — FIXED
    # ========================================================================

    st.markdown("### 🧮 Probability Breakdown")

    st.markdown("The ground truth conversion probability is computed as:")

    st.latex(r"P(\text{conv}) = R_{\text{base}} \cdot E_{\text{individual}} \cdot M_{\text{interaction}} \cdot B_{\text{recency}} \cdot N_{\text{length}}")

    col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)

    with col_b1:
        st.markdown(f"""
        <div class="formula-block">
        <strong>Base Rate</strong><br>
        R<sub>base</sub> = {BASE_CONVERSION_RATE:.3f}<br>
        <em style="color: #9CA3AF; font-size: 0.85rem;">Constant</em>
        </div>
        """, unsafe_allow_html=True)

    with col_b2:
        st.markdown(f"""
        <div class="formula-block">
        <strong>Individual Effects</strong><br>
        E = {prob_result['individual_effect']:.3f}<br>
        <em style="color: #9CA3AF; font-size: 0.85rem;">Sum of unique channels</em>
        </div>
        """, unsafe_allow_html=True)

    with col_b3:
        st.markdown(f"""
        <div class="formula-block">
        <strong>Interaction</strong><br>
        M = {prob_result['interaction_multiplier']:.3f}<br>
        <em style="color: #9CA3AF; font-size: 0.85rem;">Pairwise synergies</em>
        </div>
        """, unsafe_allow_html=True)

    with col_b4:
        st.markdown(f"""
        <div class="formula-block">
        <strong>Recency Bonus</strong><br>
        B = {prob_result['recency_bonus']:.3f}<br>
        <em style="color: #9CA3AF; font-size: 0.85rem;">Last touch boost</em>
        </div>
        """, unsafe_allow_html=True)

    with col_b5:
        st.markdown(f"""
        <div class="formula-block">
        <strong>Length Normalizer</strong><br>
        N = {prob_result['length_normalizer']:.3f}<br>
        <em style="color: #9CA3AF; font-size: 0.85rem;">1/(1+0.1·len)</em>
        </div>
        """, unsafe_allow_html=True)

    # ========================================================================
    # CONTRIBUTING INDIVIDUAL EFFECTS
    # ========================================================================

    with st.expander("📊 Individual channel effects in this path"):
        unique_in_path = list(set(path))
        effects_data = [
            {
                'Channel': CHANNEL_DISPLAY[ch],
                'Effect': CHANNEL_EFFECTS[ch],
            }
            for ch in unique_in_path
        ]
        effects_df = pd.DataFrame(effects_data).sort_values('Effect', ascending=False)

        st.dataframe(
            effects_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Channel': st.column_config.TextColumn('Channel'),
                'Effect' : st.column_config.NumberColumn('Effect Coefficient', format='%.2f'),
            }
        )

        st.caption(f"Sum: {prob_result['individual_effect']:.3f}")

    # ========================================================================
    # INTERACTION DETAILS
    # ========================================================================

    if len(path) >= 2:
        with st.expander("🔗 Pairwise interactions in this path"):
            interactions_found = []
            for i in range(len(path) - 1):
                pair = (path[i], path[i + 1])
                if pair in CHANNEL_INTERACTIONS:
                    interactions_found.append({
                        'Position': f"Step {i+1} → Step {i+2}",
                        'From'    : CHANNEL_DISPLAY[pair[0]],
                        'To'      : CHANNEL_DISPLAY[pair[1]],
                        'Multiplier': f"×{CHANNEL_INTERACTIONS[pair]:.2f}",
                    })

            if interactions_found:
                st.dataframe(
                    pd.DataFrame(interactions_found),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(f"Combined multiplier: ×{prob_result['interaction_multiplier']:.3f}")
            else:
                st.info("No pairwise synergies detected in this path.")

    # ========================================================================
    # DYNAMIC INSIGHTS
    # ========================================================================

    st.markdown("## 💡 Insights for This Journey")

    insights = []

    if len(path) == 1:
        insights.append({
            'icon' : '⚠️',
            'title': 'Single touchpoint',
            'text' : f"Single-touch journeys typically have lower conversion. "
                     f"The length normalizer is high ({prob_result['length_normalizer']:.2f}) "
                     f"but no interactions can occur."
        })
    elif len(path) >= 6:
        insights.append({
            'icon' : '🌪️',
            'title': 'Long journey',
            'text' : f"This is a {len(path)}-touchpoint journey. The length normalizer "
                     f"penalizes very long paths ({prob_result['length_normalizer']:.2f}), "
                     f"reflecting that more touches don't always mean higher conversion."
        })

    if prob_result['interaction_multiplier'] > 1.0:
        insights.append({
            'icon' : '✨',
            'title': 'Synergy detected!',
            'text' : f"This path contains channel interactions with combined "
                     f"multiplier of ×{prob_result['interaction_multiplier']:.2f}. "
                     f"This is exactly the kind of effect classical models like "
                     f"Last-Click cannot detect."
        })
    elif prob_result['interaction_multiplier'] < 1.0:
        insights.append({
            'icon' : '⚠️',
            'title': 'Penalty applied',
            'text' : f"A penalty is reducing conversion probability "
                     f"(multiplier ×{prob_result['interaction_multiplier']:.2f}). "
                     f"Single-channel social journeys, for example, are penalized "
                     f"because they reflect low purchase intent."
        })

    if prob_result['recency_bonus'] > 1.08:
        insights.append({
            'icon' : '🎯',
            'title': 'Strong recency signal',
            'text' : f"The last channel ({CHANNEL_DISPLAY[path[-1]]}) carries a "
                     f"strong recency bonus (×{prob_result['recency_bonus']:.3f}). "
                     f"This is why Last-Click attribution is so popular despite "
                     f"its limitations."
        })

    if prob_result['probability'] > 0.025:
        insights.append({
            'icon' : '🚀',
            'title': 'High-converting path',
            'text' : f"At {prob_result['probability']*100:.3f}%, this path converts "
                     f"well above the dataset average of 1.18%."
        })

    for ins in insights:
        st.markdown(f"""
        <div class="insight-box">
        {ins['icon']} <strong>{ins['title']}.</strong> {ins['text']}
        </div>
        """, unsafe_allow_html=True)

    # ========================================================================
    # PER-MODEL ATTRIBUTION
    # ========================================================================

    st.markdown("## 📊 How Each Model Credits This Journey")

    st.markdown("""
    The heatmap below shows how each attribution model distributes
    credit across the channels in your journey. **Notice the dramatic
    differences** — the same journey can be interpreted in completely
    different ways depending on the model.
    """)

    attr_df = compute_path_attributions(path)

    channels_in_path = list(set(path))
    attr_filtered = attr_df[attr_df['channel'].isin(channels_in_path)].copy()

    if len(attr_filtered) > 0:
        fig_heatmap = plot_path_attribution_heatmap(attr_filtered)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ========================================================================
    # MODEL COMPARISON TABLE
    # ========================================================================

    st.markdown("### 📋 Detailed Attribution Breakdown")

    comparison_df = attr_filtered.copy()
    comparison_df['channel'] = comparison_df['channel'].map(CHANNEL_DISPLAY)
    comparison_df = comparison_df.rename(columns={'channel': 'Channel'})

    for col in comparison_df.columns:
        if col != 'Channel':
            comparison_df[col] = (comparison_df[col] * 100).round(1)

    comparison_df = comparison_df.rename(columns={
        'Last_Click'    : 'Last-Click',
        'First_Click'   : 'First-Click',
        'Time_Decay'    : 'Time-Decay',
        'Position_Based': 'Position-Based',
    })

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            col: st.column_config.NumberColumn(col, format='%.1f%%')
            for col in comparison_df.columns if col != 'Channel'
        }
    )

    # ========================================================================
    # KEY OBSERVATIONS
    # ========================================================================

    with st.expander("🔍 What does this table tell us?"):
        first_ch = CHANNEL_DISPLAY[path[0]]
        last_ch = CHANNEL_DISPLAY[path[-1]]

        st.markdown(f"""
        ### Observations for this specific journey:

        - **Last-Click** gives **100% credit to {last_ch}** (the final touchpoint).
          All earlier channels — including their role in initiating awareness —
          receive zero credit.

        - **First-Click** gives **100% credit to {first_ch}** (the initial touchpoint).
          The opposite extreme — assumes all conversion is due to the first touch.

        - **Linear** divides credit equally across all {len(path)} touchpoints,
          giving each one **{100/len(path):.1f}%**. Simple but treats all
          touchpoints as equally important.

        - **Time-Decay** weights later touches more heavily. The final channel
          ({last_ch}) gets the largest share, but earlier touches still get
          some credit (unlike Last-Click).

        - **Position-Based (U-shaped)** gives **40% to {first_ch} (first)** and
          **40% to {last_ch} (last)**, distributing the remaining 20% across
          middle touchpoints. Reflects the intuition that beginnings and
          endings matter most.

        ### The bigger picture

        For data-driven attribution (Markov, Shapley, Hybrid), credit assignment
        requires the **full dataset** — these models cannot be computed from a
        single path alone. They learn channel-level weights from thousands of
        journeys and apply those weights globally.

        That's why the **Markov, Shapley, and Hybrid** columns are not shown
        here — they're computed across all 30,000 users in the dataset, not
        per-journey.
        """)


# ============================================================================
# EDUCATIONAL FOOTER — FIXED
# ============================================================================

st.divider()

with st.expander("📚 Understanding the Ground Truth Model"):
    st.markdown("### How the Ground Truth Works")

    st.markdown(
        "The synthetic data generator uses this formula to simulate true "
        "conversion probability for each path:"
    )

    st.latex(r"P(\text{conv}) = R_{\text{base}} \cdot E_{\text{individual}} \cdot M_{\text{interaction}} \cdot B_{\text{recency}} \cdot N_{\text{length}}")

    st.markdown(f"**Base rate:** {BASE_CONVERSION_RATE:.3f} (industry-grounded e-commerce benchmark)")

    st.markdown("""
**Individual effects** (calibrated from real-world conversion rate data):

| Channel | Effect | Interpretation |
|---------|-------:|----------------|
| Google Ads | 0.35 | Highest intent (paid search) |
| Email | 0.25 | High conversion when segmented |
| Direct | 0.20 | Decision-stage users |
| Organic Search | 0.15 | Exploration phase |
| Social Media | 0.10 | Awareness phase |

**Pairwise interactions** (synergies):

| Pair | Multiplier |
|------|-----------:|
| Social Media → Google Ads | ×1.15 |
| Email → Direct | ×1.20 |
| Organic Search → Email | ×1.10 |
| Google Ads → Email | ×1.12 |
| Social Media → Organic Search | ×1.08 |

**Recency bonus:** 1 + 0.3 × effect of last channel

**Length normalizer:** 1 / (1 + 0.1 × path length)

**Single Social Media penalty:** ×0.6 if path consists only of Social Media

### Why This Matters for Attribution

Real attribution models don't know these parameters — they have to
**estimate** them from data. The validation framework in this thesis
measures how well each model recovers the underlying ground truth.

By building custom paths here, you can see directly how different
attribution methodologies would credit each touchpoint, and develop
intuition for why classical methods systematically misattribute.
""")
