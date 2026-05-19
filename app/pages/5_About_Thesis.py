"""
Page 5: About the Thesis
=========================

Academic reference page providing thesis abstract, key findings,
citation information, author details, and visualization gallery.
"""

import sys
import os

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.data_loader import (
    load_final_metrics,
    load_multi_budget_revenue,
    MODEL_INFO,
)
from utils.plotting import apply_dark_theme


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="About the Thesis",
    page_icon="📈",
    layout="wide",
)


# ============================================================================
# PATHS
# ============================================================================

# Path to figures directory (for the gallery)
REPO_ROOT = os.path.dirname(os.path.dirname(APP_DIR))
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures')


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .page-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #60A5FA 0%, #DC2626 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .page-subtitle {
        font-size: 1.1rem;
        color: #9CA3AF;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .info-card {
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #374151;
        margin: 1rem 0;
        color: #E5E7EB;
    }
    .info-card h3 {
        color: #60A5FA;
        margin-top: 0;
    }
    .stat-card {
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #60A5FA;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stat-card-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #60A5FA;
        margin: 0.5rem 0;
    }
    .stat-card-label {
        font-size: 0.95rem;
        color: #9CA3AF;
        margin: 0;
    }
    .link-row {
        background-color: #1F2937;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #60A5FA;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HEADER
# ============================================================================

st.markdown('<h1 class="page-title">📈 About the Thesis</h1>',
            unsafe_allow_html=True)

st.markdown('<p class="page-subtitle">A Hybrid Markov Chain and Shapley Value Approach to Multi-Touch Attribution and Budget Optimization in Digital Marketing</p>',
            unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### 📋 Page Contents")
    st.markdown("""
    - [Abstract](#abstract)
    - [Author](#author)
    - [Key Findings](#key-findings)
    - [Methodology](#methodology)
    - [Results Gallery](#results-gallery)
    - [Citation](#citation)
    - [Links & Resources](#links-resources)
    - [Acknowledgments](#acknowledgments)
    """)

    st.divider()

    st.markdown("### 🎓 Quick Facts")
    st.markdown("""
    - **Year:** 2026
    - **Institution:** Istanbul University
    - **Department:** Management Information Systems
    - **Type:** Undergraduate Thesis (Graduation Project)
    - **Pages:** ~100
    - **References:** 72 academic sources
    - **Code:** Open source (MIT License)
    """)

    st.divider()

    st.markdown("### 📥 Downloads")
    st.markdown("""
    - 📄 [Thesis PDF](https://github.com/brky48/hybrid-mta-attribution)
    - 💻 [Source Code](https://github.com/brky48/hybrid-mta-attribution)
    - 📊 [Datasets](https://github.com/brky48/hybrid-mta-attribution/tree/main/data)
    """)


# ============================================================================
# ABSTRACT SECTION
# ============================================================================

st.markdown('<a name="abstract"></a>', unsafe_allow_html=True)
st.markdown("## 📄 Abstract")

st.markdown("""
Multi-touch attribution (MTA) is a foundational analytical problem in
digital marketing, concerning the assignment of conversion credit
across the channels that contribute to a customer journey. The
validity of competing attribution methodologies has historically been
difficult to assess objectively, owing to the unobservability of the
true channel contribution structure in real-world data.

This thesis addresses the validation problem through a **synthetic-data
framework with known ground truth**, and proposes a **hybrid
Markov-Shapley attribution model** that combines two complementary
data-driven approaches. The hybrid model's mixing parameter is
determined empirically through grid search, and its outputs are
integrated with a non-linear budget optimization layer employing
diminishing-returns response curves and the Differential Evolution
algorithm.

Empirical evaluation against a synthetic dataset of 30,000 customer
journeys demonstrates that the hybrid model achieves substantially
superior performance across both attribution accuracy and budget
allocation metrics, particularly in the budget range characteristic
of small-to-medium B2C e-commerce operations. At a 10,000 USD budget
level, the hybrid model generates **13.72% more conversions** than the
industry-standard Last-Click attribution model, while achieving a
**21.2% reduction in Mean Absolute Error** against the ground truth.

The framework contributes three principal advances: (1) a reproducible
validation methodology for attribution models using synthetic ground
truth, (2) empirical demonstration of the **Markov-Shapley trade-off**
between rank identification and magnitude calibration, and (3)
principled integration of attribution with non-linear budget
optimization.
""")


# ============================================================================
# AUTHOR SECTION
# ============================================================================

st.markdown('<a name="author"></a>', unsafe_allow_html=True)
st.markdown("## 👤 Author")

col_a1, col_a2 = st.columns([2, 1])

with col_a1:
    st.markdown("""
    <div class="info-card">
        <h3>Berkay Korkut</h3>
        <p><strong>Undergraduate Student</strong></p>
        <p>Department of Management Information Systems<br>
        Faculty of Economics<br>
        Istanbul University</p>
        <p><strong>Year:</strong> 2026</p>
        <p><strong>Student No:</strong> 0515220026</p>
        <p><strong>GitHub:</strong> <a href="https://github.com/brky48" style="color: #60A5FA;">@brky48</a></p>
    </div>
    """, unsafe_allow_html=True)

with col_a2:
    st.markdown("""
    <div class="info-card">
        <h3>Supervisor</h3>
        <p><strong>Asst. Prof. Dr. Mian Waqar Badshah</strong></p>
        <p>Department of Management Information Systems<br>
        Faculty of Economics<br>
        Istanbul University</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# KEY FINDINGS SECTION
# ============================================================================

st.markdown('<a name="key-findings"></a>', unsafe_allow_html=True)
st.markdown("## 🎯 Key Findings")

# Stat cards row 1 — accuracy metrics
st.markdown("### Attribution Accuracy")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    st.markdown("""
    <div class="stat-card">
        <p class="stat-card-label">Hybrid Model MAE</p>
        <p class="stat-card-value">0.0571</p>
        <p class="stat-card-label">21.2% lower than Last-Click</p>
    </div>
    """, unsafe_allow_html=True)

with col_s2:
    st.markdown("""
    <div class="stat-card">
        <p class="stat-card-label">Optimal Alpha</p>
        <p class="stat-card-value">0.25</p>
        <p class="stat-card-label">25% Markov + 75% Shapley</p>
    </div>
    """, unsafe_allow_html=True)

with col_s3:
    st.markdown("""
    <div class="stat-card">
        <p class="stat-card-label">Spearman ρ</p>
        <p class="stat-card-value">0.600</p>
        <p class="stat-card-label">2× higher than Last-Click</p>
    </div>
    """, unsafe_allow_html=True)

with col_s4:
    st.markdown("""
    <div class="stat-card">
        <p class="stat-card-label">Rank Accuracy</p>
        <p class="stat-card-value">60%</p>
        <p class="stat-card-label">3/5 channels correctly ranked</p>
    </div>
    """, unsafe_allow_html=True)

# Stat cards row 2 — revenue impact
st.markdown("### Revenue Impact (vs Last-Click)")

col_r1, col_r2, col_r3, col_r4 = st.columns(4)

with col_r1:
    st.markdown("""
    <div class="stat-card" style="border-left-color: #10B981;">
        <p class="stat-card-label">Budget: $10,000</p>
        <p class="stat-card-value" style="color: #10B981;">+13.72%</p>
        <p class="stat-card-label">More conversions</p>
    </div>
    """, unsafe_allow_html=True)

with col_r2:
    st.markdown("""
    <div class="stat-card" style="border-left-color: #10B981;">
        <p class="stat-card-label">Budget: $50,000</p>
        <p class="stat-card-value" style="color: #10B981;">+3.35%</p>
        <p class="stat-card-label">More conversions</p>
    </div>
    """, unsafe_allow_html=True)

with col_r3:
    st.markdown("""
    <div class="stat-card" style="border-left-color: #F59E0B;">
        <p class="stat-card-label">Budget: $100,000</p>
        <p class="stat-card-value" style="color: #F59E0B;">+0.77%</p>
        <p class="stat-card-label">More conversions</p>
    </div>
    """, unsafe_allow_html=True)

with col_r4:
    st.markdown("""
    <div class="stat-card" style="border-left-color: #6B7280;">
        <p class="stat-card-label">Budget: $500,000</p>
        <p class="stat-card-value" style="color: #6B7280;">+0.00%</p>
        <p class="stat-card-label">Saturation reached</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
**Key insight:** The hybrid model's value is **concentrated at moderate
budget levels** (10K–50K USD), making it particularly suitable for
small-to-medium B2C e-commerce operations. At larger budgets, all
channels saturate and model differences vanish.
""")


# ============================================================================
# PERFORMANCE TABLE
# ============================================================================

st.markdown("### Full Performance Ranking")

metrics_df = load_final_metrics().copy()
metrics_df = metrics_df.sort_values('MAE').reset_index(drop=True)
metrics_df.index = metrics_df.index + 1

# Add display names
metrics_df['Model'] = metrics_df['Model'].map(
    lambda m: MODEL_INFO.get(m, {'name': m})['name']
)

# Add medal column
def medal(rank):
    if rank == 1: return "🥇"
    elif rank == 2: return "🥈"
    elif rank == 3: return "🥉"
    else: return f"#{rank}"

metrics_df.insert(0, 'Rank', [medal(i) for i in range(1, len(metrics_df) + 1)])

metrics_df = metrics_df.rename(columns={
    'Spearman_Corr'   : 'Spearman ρ',
    'Ranking_Accuracy': 'Rank Acc',
})

def highlight_top3(row):
    rank_str = str(row['Rank'])
    if '🥇' in rank_str:
        return ['background-color: rgba(220, 38, 38, 0.2); font-weight: 700;'] * len(row)
    elif '🥈' in rank_str or '🥉' in rank_str:
        return ['background-color: rgba(96, 165, 250, 0.1);'] * len(row)
    return [''] * len(row)

styled_metrics = metrics_df.style.apply(highlight_top3, axis=1)

st.dataframe(
    styled_metrics,
    use_container_width=True,
    hide_index=True,
    column_config={
        'Rank'      : st.column_config.TextColumn('Rank', width='small'),
        'Model'     : st.column_config.TextColumn('Model', width='medium'),
        'MAE'       : st.column_config.NumberColumn('MAE', format='%.4f'),
        'RMSE'      : st.column_config.NumberColumn('RMSE', format='%.4f'),
        'Spearman ρ': st.column_config.NumberColumn('Spearman ρ', format='%.3f'),
        'Rank Acc'  : st.column_config.TextColumn('Rank Acc'),
    },
)


# ============================================================================
# METHODOLOGY SECTION
# ============================================================================

st.markdown('<a name="methodology"></a>', unsafe_allow_html=True)
st.markdown("## 🔬 Methodology")

st.markdown("""
The methodological framework comprises four sequential stages:
""")

col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown("""
    <div class="info-card">
        <h3>1. Synthetic Data Generation</h3>
        <ul>
            <li>30,000 simulated customer journeys</li>
            <li>5 marketing channels</li>
            <li>105,565 total touchpoints</li>
            <li>Average path length: 3.52 steps</li>
            <li>Conversion rate: 1.18% (industry-grounded)</li>
            <li>Monte Carlo ground truth (100K paths)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>2. Attribution Models (8 total)</h3>
        <p><strong>Heuristic baselines:</strong> Last-Click, First-Click,
        Linear, Time-Decay, Position-Based</p>
        <p><strong>Data-driven:</strong> Markov Chain (removal effects),
        Shapley Value (fractional credit)</p>
        <p><strong>Proposed:</strong> Hybrid (Markov + Shapley) with
        grid-searched optimal α</p>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown("""
    <div class="info-card">
        <h3>3. Hybrid Model</h3>
        <p>Convex linear combination:</p>
        <p style="text-align: center; font-family: 'Courier New', monospace;
        background: #111827; padding: 0.5rem; border-radius: 4px;">
        w<sub>c</sub><sup>Hybrid</sup> = α · w<sub>c</sub><sup>Markov</sup>
        + (1-α) · w<sub>c</sub><sup>Shapley</sup>
        </p>
        <p>α* = 0.25 determined via grid search (101 points over [0, 1])</p>
        <p>Minimizes MAE against ground truth</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>4. Budget Optimization</h3>
        <p><strong>Algorithm:</strong> Differential Evolution
        (Storn & Price, 1997)</p>
        <p><strong>Response curves:</strong> Exponential saturation</p>
        <p><strong>Constraints:</strong> 5% ≤ allocation ≤ 80% per channel</p>
        <p><strong>Tested budgets:</strong> 10K, 50K, 100K, 500K USD</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# RESULTS GALLERY
# ============================================================================

st.markdown('<a name="results-gallery"></a>', unsafe_allow_html=True)
st.markdown("## 🖼️ Results Gallery")

st.markdown("""
Below are key visualizations from the thesis. All figures are
reproducible by running the source code in the
[GitHub repository](https://github.com/brky48/hybrid-mta-attribution).
""")

# Gallery configuration: (filename, title, description)
gallery_items = [
    ('comparison_attribution.png',
     'Attribution Comparison',
     'Side-by-side comparison of all 8 attribution models against the ground truth distribution.'),
    ('comparison_errors.png',
     'Error Metrics',
     'MAE and RMSE across all models. Lower is better. The Hybrid model achieves the lowest error.'),
    ('alpha_grid_search.png',
     'Alpha Grid Search',
     'MAE as a function of the hybrid mixing parameter α. The minimum occurs at α* = 0.25.'),
    ('response_curves.png',
     'Response Curves',
     'Diminishing-returns saturation curves for the three paid channels at the optimal allocation.'),
    ('multi_budget_revenue.png',
     'Multi-Budget Revenue',
     'Hybrid vs Last-Click revenue performance across four budget levels. Hybrid leads at low budgets.'),
    ('comparison_correlation.png',
     'Correlation Heatmap',
     'Spearman rank correlation matrix across all models and the ground truth.'),
]

# Display in 2-column grid
for i in range(0, len(gallery_items), 2):
    col_g1, col_g2 = st.columns(2)

    for j, col in enumerate([col_g1, col_g2]):
        if i + j < len(gallery_items):
            filename, title, desc = gallery_items[i + j]
            img_path = os.path.join(FIGURES_DIR, filename)

            with col:
                if os.path.exists(img_path):
                    st.image(img_path, caption=title, use_container_width=True)
                else:
                    st.warning(f"⚠️ Figure not found: {filename}")

                st.caption(desc)


# ============================================================================
# CITATION SECTION
# ============================================================================

st.markdown('<a name="citation"></a>', unsafe_allow_html=True)
st.markdown("## 📝 Citation")

st.markdown("""
If you use this work in your research or refer to it in your own
publications, please cite it as follows:
""")

bibtex = """@thesis{korkut2026hybrid,
  author       = {Korkut, Berkay},
  title        = {A Hybrid {Markov} Chain and {Shapley} Value Approach to
                  Multi-Touch Attribution and Budget Optimization in
                  Digital Marketing},
  school       = {Istanbul University, Faculty of Economics,
                  Management Information Systems},
  year         = {2026},
  type         = {Undergraduate Thesis},
  url          = {https://github.com/brky48/hybrid-mta-attribution}
}"""

st.code(bibtex, language='bibtex')

# APA format alternative
st.markdown("**APA Format:**")
st.markdown("""
> Korkut, B. (2026). *A hybrid Markov chain and Shapley value approach
> to multi-touch attribution and budget optimization in digital
> marketing* (Undergraduate thesis). Istanbul University, Faculty of
> Economics, Department of Management Information Systems.
""")


# ============================================================================
# LINKS & RESOURCES
# ============================================================================

st.markdown('<a name="links-resources"></a>', unsafe_allow_html=True)
st.markdown("## 🔗 Links & Resources")

col_l1, col_l2 = st.columns(2)

with col_l1:
    st.markdown("""
    <div class="info-card">
        <h3>📦 Code & Data</h3>
        <p>
            🐙 <a href="https://github.com/brky48/hybrid-mta-attribution"
            style="color: #60A5FA;">GitHub Repository</a>
        </p>
        <p>
            📊 <a href="https://github.com/brky48/hybrid-mta-attribution/tree/main/data"
            style="color: #60A5FA;">Synthetic Dataset (CSV)</a>
        </p>
        <p>
            📈 <a href="https://github.com/brky48/hybrid-mta-attribution/tree/main/results"
            style="color: #60A5FA;">Result CSVs</a>
        </p>
        <p>
            🖼️ <a href="https://github.com/brky48/hybrid-mta-attribution/tree/main/figures"
            style="color: #60A5FA;">All Figures (PNG)</a>
        </p>
        <p>
            🎯 <a href="https://hybrid-attribution-explorer.streamlit.app"
            style="color: #60A5FA;">This Interactive Demo</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_l2:
    st.markdown("""
    <div class="info-card">
        <h3>🎓 Academic</h3>
        <p>
            🏛️ <a href="https://www.istanbul.edu.tr/"
            style="color: #60A5FA;">Istanbul University</a>
        </p>
        <p>
            📚 <a href="https://iktisat.istanbul.edu.tr/"
            style="color: #60A5FA;">Faculty of Economics</a>
        </p>
        <p>
            💼 <a href="https://iktisat.istanbul.edu.tr/yonetim-bilisim-sistemleri/"
            style="color: #60A5FA;">Department of Management Information Systems</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# ACKNOWLEDGMENTS
# ============================================================================

st.markdown('<a name="acknowledgments"></a>', unsafe_allow_html=True)
st.markdown("## 🙏 Acknowledgments")

st.markdown("""
This thesis was developed as the **Graduation Project** at Istanbul
University, Faculty of Economics. The methodological framework draws
on foundational contributions from the following authors:
""")

ack_items = [
    ("**Markov-based attribution**", "Anderl et al. (2016)",
     "Mapping the customer journey with graph-based models"),
    ("**Cooperative game theory**", "Shapley (1953)",
     "Original formulation of the Shapley value"),
    ("**Practical Shapley attribution**", "Shao & Li (2011)",
     "Fractional credit formulation for data-driven attribution"),
    ("**Marketing Mix Modeling**", "Jin et al. (2017), Chan & Perry (2017)",
     "Bayesian MMM with carryover and shape effects"),
    ("**Differential Evolution**", "Storn & Price (1997)",
     "Population-based global optimization heuristic"),
]

for topic, authors, desc in ack_items:
    st.markdown(f"- {topic}: **{authors}** — {desc}")

st.markdown("""
The complete bibliography of **72 academic sources** is provided in
the thesis document.
""")


# ============================================================================
# FOOTER
# ============================================================================

st.divider()

st.markdown("""
<div style="text-align: center; color: #9CA3AF; padding: 2rem 1rem;">
    <p style="font-size: 1.1rem;">
        🎓 <strong>Hybrid Attribution Explorer</strong><br>
        Interactive companion to an undergraduate thesis at Istanbul University
    </p>
    <p style="font-size: 0.9rem;">
        Developed in Python with Streamlit, Plotly, NumPy, and SciPy.<br>
        Open source under MIT License.
    </p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">
        © 2026 Berkay Korkut · <a href="https://github.com/brky48/hybrid-mta-attribution"
        style="color: #60A5FA;">Source on GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)
