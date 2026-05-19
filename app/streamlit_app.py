"""
Hybrid Attribution Explorer
============================
Interactive companion to the thesis:
"A Hybrid Markov Chain and Shapley Value Approach to Multi-Touch
Attribution and Budget Optimization in Digital Marketing"

Author: Berkay Korkut
Istanbul University, Faculty of Economics
Management Information Systems, 2026
"""

import streamlit as st

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Hybrid Attribution Explorer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
        # Hybrid Attribution Explorer

        An interactive companion to the undergraduate thesis:
        *A Hybrid Markov Chain and Shapley Value Approach to
        Multi-Touch Attribution and Budget Optimization in
        Digital Marketing*

        **Author:** Berkay Korkut
        **Institution:** Istanbul University, Faculty of Economics
        **Year:** 2026
        """
    }
)


# ============================================================================
# CUSTOM CSS - DARK ACADEMIC THEME
# ============================================================================

st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #60A5FA 0%, #DC2626 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 1.2rem;
        color: #9CA3AF;
        margin-bottom: 2rem;
        font-style: italic;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }

    /* Section headers */
    h2 {
        color: #60A5FA;
        border-bottom: 2px solid #1E3A8A;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }

    /* Page cards */
    .page-card {
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #374151;
        height: 100%;
        transition: all 0.3s ease;
    }

    .page-card:hover {
        border-color: #60A5FA;
        transform: translateY(-2px);
    }

    .page-card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #F9FAFB;
        margin-bottom: 0.5rem;
    }

    .page-card-desc {
        font-size: 0.95rem;
        color: #9CA3AF;
        line-height: 1.5;
    }

    /* Citation block */
    .citation-block {
        background-color: #1F2937;
        border-left: 4px solid #60A5FA;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        color: #D1D5DB;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### 📚 About This Demo")
    st.markdown("""
    **Hybrid Attribution Explorer** is the interactive companion to an
    undergraduate thesis on multi-touch attribution in digital marketing.

    Use the pages on the left to explore:
    - Attribution model comparison
    - Budget optimization
    - Customer journey analysis
    - Methodology explanations
    """)

    st.divider()

    st.markdown("### 🔗 Links")
    st.markdown("""
    - 💻 [GitHub Repository](https://github.com/brky48/hybrid-mta-attribution)
    - 🎓 [Istanbul University](https://www.istanbul.edu.tr/)
    - 📄 [Thesis Documentation](https://github.com/brky48/hybrid-mta-attribution)
    """)

    st.divider()

    st.caption("**Author:** Berkay Korkut")
    st.caption("**Year:** 2026")
    st.caption("**License:** MIT")


# ============================================================================
# MAIN CONTENT - HERO SECTION
# ============================================================================

st.markdown('<h1 class="main-title">🎯 Hybrid Attribution Explorer</h1>',
            unsafe_allow_html=True)

st.markdown('<p class="subtitle">An Interactive Companion to Multi-Touch Attribution Research</p>',
            unsafe_allow_html=True)

st.markdown("""
Welcome to the interactive demonstration of the **Hybrid Markov-Shapley
attribution framework** developed in the undergraduate thesis
*"A Hybrid Markov Chain and Shapley Value Approach to Multi-Touch
Attribution and Budget Optimization in Digital Marketing"*
(Istanbul University, 2026).

This application allows you to **explore, manipulate, and visualize** the
core components of the framework in real time.
""")


# ============================================================================
# KEY METRICS ROW
# ============================================================================

st.markdown("## 📊 Key Results at a Glance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Hybrid Model MAE",
        value="0.0571",
        delta="-21.2% vs Last-Click",
        delta_color="inverse"
    )

with col2:
    st.metric(
        label="Optimal Alpha (α*)",
        value="0.25",
        help="25% Markov + 75% Shapley"
    )

with col3:
    st.metric(
        label="Revenue Lift (10K USD)",
        value="+13.72%",
        delta="vs Last-Click baseline"
    )

with col4:
    st.metric(
        label="Theoretical Upper Bound",
        value="99.95%",
        help="At 100K USD budget level"
    )


# ============================================================================
# PAGE NAVIGATION CARDS
# ============================================================================

st.markdown("## 🗺️ Explore the Framework")

st.markdown("Navigate to any module using the sidebar, or click on the cards below to learn more:")

# First row
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="page-card">
        <div class="page-card-title">📊 Attribution Comparison</div>
        <div class="page-card-desc">
            Interactively adjust the hybrid mixing parameter α and observe
            how 8 attribution models compare against the ground truth.
            Watch the bars animate as you slide.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="page-card">
        <div class="page-card-title">💰 Budget Optimizer</div>
        <div class="page-card-desc">
            Set your total budget and channel constraints. The framework
            computes the optimal allocation under diminishing-returns
            response curves in real time.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Second row
col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    <div class="page-card">
        <div class="page-card-title">🛤️ Path Explorer</div>
        <div class="page-card-desc">
            Build a customer journey touchpoint-by-touchpoint and observe
            how each attribution model assigns credit. Discover why
            Last-Click misses so much information.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="page-card">
        <div class="page-card-title">📚 Methodology Explainer</div>
        <div class="page-card-desc">
            Learn how Markov chains and Shapley values produce attribution
            estimates. Interactive visualizations make the mathematics
            tangible and intuitive.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# THESIS OVERVIEW
# ============================================================================

st.markdown("## 📖 About the Thesis")

st.markdown("""
Multi-touch attribution (MTA) is a foundational analytical problem in digital
marketing: how should conversion credit be distributed across the channels
that contribute to a customer journey? Classical heuristics such as Last-Click
attribution remain dominant in practice despite well-documented limitations.

This thesis develops a **hybrid Markov-Shapley framework** that combines two
complementary data-driven approaches:

- **Markov chain attribution** excels at identifying which channels matter
  most (rank identification) but tends to exaggerate magnitudes.
- **Shapley value attribution** produces well-calibrated magnitudes but
  compresses the ranking signal.

The hybrid model resolves this trade-off through a convex linear combination
with an empirically determined mixing parameter (α* = 0.25).

**Key contributions:**
1. Synthetic-data validation framework with known ground truth
2. Empirical demonstration of the Markov-Shapley trade-off
3. Integration with non-linear budget optimization via Differential Evolution
""")


# ============================================================================
# CITATION
# ============================================================================

st.markdown("## 📝 Citation")

st.markdown("If you use this work in your research, please cite:")

st.code("""
@thesis{korkut2026hybrid,
  author       = {Korkut, Berkay},
  title        = {A Hybrid Markov Chain and Shapley Value Approach to
                  Multi-Touch Attribution and Budget Optimization in
                  Digital Marketing},
  school       = {Istanbul University, Faculty of Economics,
                  Management Information Systems},
  year         = {2026},
  type         = {Undergraduate Thesis},
  url          = {https://github.com/brky48/hybrid-mta-attribution}
}
""", language="bibtex")


# ============================================================================
# FOOTER
# ============================================================================

st.divider()

footer_col1, footer_col2, footer_col3 = st.columns([1, 1, 1])

with footer_col1:
    st.caption("**Author**")
    st.caption("Berkay Korkut")

with footer_col2:
    st.caption("**Institution**")
    st.caption("Istanbul University")

with footer_col3:
    st.caption("**License**")
    st.caption("MIT License")
