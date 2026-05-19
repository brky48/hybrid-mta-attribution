# Hybrid Markov Chain and Shapley Value Approach to Multi-Touch Attribution

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Thesis](https://img.shields.io/badge/Thesis-Istanbul%20University-red.svg)](#citation)

> **Undergraduate Thesis Project**
> *Istanbul University, Faculty of Economics, Management Information Systems*
> *2026*

This repository contains the complete Python implementation of an undergraduate thesis titled **"A Hybrid Markov Chain and Shapley Value Approach to Multi-Touch Attribution and Budget Optimization in Digital Marketing."** The project develops, implements, and empirically evaluates a hybrid attribution framework that combines the directional sensitivity of Markov chain removal-effect analysis with the axiomatic credit allocation of Shapley value theory, and integrates the resulting attribution estimates into a non-linear budget optimization layer.

---

## Table of Contents

- [Abstract](#abstract)
- [Key Findings](#key-findings)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

---

## Abstract

Multi-touch attribution (MTA) is a foundational analytical problem in digital marketing, concerning the assignment of conversion credit across the channels that contribute to a customer journey. The validity of competing attribution methodologies has historically been difficult to assess objectively, owing to the unobservability of the true channel contribution structure in real-world data.

This thesis addresses the validation problem through a synthetic-data framework with known ground truth, and proposes a **hybrid Markov-Shapley attribution model** that combines two complementary data-driven approaches. The hybrid model's mixing parameter is determined empirically through grid search, and its outputs are integrated with a non-linear budget optimization layer employing diminishing-returns response curves and the Differential Evolution algorithm.

Empirical evaluation against a synthetic dataset of 30,000 customer journeys demonstrates that the hybrid model achieves substantially superior performance across both attribution accuracy and budget allocation metrics, particularly in the budget range characteristic of small-to-medium B2C e-commerce operations.

---

## Key Findings

| Metric | Hybrid Model | Best Baseline (Last-Click) | Improvement |
|--------|-------------:|---------------------------:|-------------|
| Mean Absolute Error (MAE) | **0.0571** | 0.0725 | **21.2%** lower |
| Root Mean Squared Error (RMSE) | **0.0682** | 0.0805 | **15.3%** lower |
| Spearman Rank Correlation | 0.600 | 0.300 | **2.0x** higher |
| Theoretical Upper Bound (100K USD) | **99.95%** | 99.20% | --- |

### Revenue Improvement vs. Last-Click (Multi-Budget)

| Budget (USD) | Hybrid Response | Last-Click Response | Improvement |
|-------------:|----------------:|--------------------:|------------:|
| 10,000 | 0.2845 | 0.2502 | **+13.72%** |
| 50,000 | 0.5555 | 0.5375 | +3.35% |
| 100,000 | 0.6611 | 0.6561 | +0.77% |
| 500,000 | 0.7023 | 0.7023 | 0.00% |

**Key insight:** The hybrid model's value is concentrated at moderate budget levels (10K-50K USD), making it particularly suitable for small-to-medium B2C e-commerce operations.

---

## Methodology

The methodological framework comprises four sequential stages:

### 1. Synthetic Data Generation
- 30,000 simulated customer journeys
- 5 marketing channels: Google Ads, Email, Direct, Organic Search, Social Media
- Industry-grounded conversion rate benchmarks (WordStream, Lucky Orange)
- Monte Carlo ground truth simulation (100,000 paths)
- Overall conversion rate: 1.18% (consistent with B2C e-commerce benchmarks)

### 2. Attribution Models (8 total)
- **Heuristic baselines:** Last-Click, First-Click, Linear, Time-Decay, Position-Based
- **Data-driven models:** Markov Chain (removal effects), Shapley Value (fractional credit)
- **Proposed hybrid:** Weighted combination with grid-searched optimal alpha

### 3. Hybrid Model

The hybrid attribution weight combines the two component models:

```
w_c^Hybrid = alpha * w_c^Markov + (1 - alpha) * w_c^Shapley
```

Optimal alpha determined via grid search against the ground truth:
- **alpha\* = 0.25** (25% Markov + 75% Shapley)
- Minimizes Mean Absolute Error

### 4. Budget Optimization

Non-linear optimization with diminishing-returns response curves:

```
Response(x) = score * (1 - exp(-k * x / x_half))
```

- **Optimizer:** Differential Evolution (Storn & Price, 1997)
- **Constraints:** 5% <= allocation <= 80% per channel
- **Budget conservation** enforced via penalty term

---

## Repository Structure

```
hybrid-mta-attribution/
|
|-- README.md                          # This file
|-- LICENSE                            # MIT License
|-- requirements.txt                   # Python dependencies
|
|-- src/                               # Python implementation
|   |-- data_generator.py              # Synthetic journey generation + Monte Carlo GT
|   |-- markov_chain.py                # Absorbing Markov chain + removal effects
|   |-- shapley_value.py               # Cooperative game-theoretic Shapley values
|   |-- baseline_models.py             # 5 classical heuristic baselines
|   |-- budget_optimization.py         # Hybrid model + DE budget optimizer
|   |-- revenue_simulation.py          # Multi-budget revenue simulation
|   |-- comparison_report.py           # Final aggregation and comparison
|   |-- export_matrices.py             # Matrix export for Appendices B and C
|
|-- data/                              # Generated dataset
|   |-- customer_journey_data.csv      # 30K users, 105K touchpoints
|
|-- results/                           # Model outputs (CSV)
|   |-- ground_truth.csv               # Ground truth attribution weights
|   |-- markov_results.csv             # Markov removal effects
|   |-- shapley_results.csv            # Shapley value attributions
|   |-- baseline_results.csv           # 5 baseline model outputs
|   |-- baseline_results_pct.csv       # Baselines (normalized)
|   |-- optimal_alpha.csv              # Grid search optimal alpha
|   |-- final_comparison.csv           # All 8 models vs. ground truth
|   |-- final_metrics.csv              # MAE, RMSE, Spearman metrics
|   |-- budget_optimization_results.csv # 100K USD allocation
|   |-- multi_budget_results.csv       # 4-budget allocations
|   |-- revenue_comparison_results.csv # Revenue performance (100K)
|   |-- multi_budget_revenue_results.csv # Multi-budget revenue
|   |-- markov_transition_matrix.csv   # Appendix B
|   |-- channel_interaction_matrix.csv # Appendix C
|
|-- figures/                           # Generated visualizations (PNG)
    |-- markov_removal_effects.png
    |-- shapley_results.png
    |-- alpha_grid_search.png
    |-- response_curves.png
    |-- multi_budget_comparison.png
    |-- comparison_attribution.png
    |-- comparison_errors.png
    |-- comparison_correlation.png
    |-- revenue_comparison.png
    |-- multi_budget_revenue.png
```

---

## Installation

### Prerequisites

- Python 3.12 or higher
- pip (Python package manager)

### Setup

Clone the repository:

```bash
git clone https://github.com/brky48/hybrid-mta-attribution.git
cd hybrid-mta-attribution
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

| Package | Version |
|---------|---------|
| numpy | >= 2.3.0 |
| pandas | >= 2.3.0 |
| scipy | >= 1.16.0 |
| matplotlib | >= 3.10.0 |

---

## Usage

Run the modules in the following order to reproduce the complete analysis:

```bash
cd src/

# 1. Generate synthetic customer journey data and Monte Carlo ground truth
python data_generator.py

# 2. Run Markov chain attribution (removal effects)
python markov_chain.py

# 3. Run Shapley value attribution (fractional credit)
python shapley_value.py

# 4. Run 5 classical baseline models
python baseline_models.py

# 5. Find optimal alpha and run budget optimization
python budget_optimization.py

# 6. Compute multi-budget revenue simulation
python revenue_simulation.py

# 7. Generate final comparison report and visualizations
python comparison_report.py

# 8. (Optional) Export transition and interaction matrices
python export_matrices.py
```

**Total execution time:** Approximately 15 minutes on consumer hardware.

### Reproducibility

All random processes are seeded for full reproducibility:

| Component | Seed |
|-----------|-----:|
| Synthetic data generation | 42 |
| Monte Carlo ground truth | 123 |
| Differential Evolution optimizer | 42 |

---

## Results

### Attribution Accuracy Ranking

| Rank | Model | MAE | RMSE | Spearman rho |
|:----:|-------|----:|-----:|-------------:|
| 1 | **Hybrid (Markov-Shapley)** | **0.0571** | **0.0682** | 0.600 |
| 2 | Last-Click | 0.0725 | 0.0805 | 0.300 |
| 3 | Time-Decay | 0.0773 | 0.0851 | 0.100 |
| 4 | Linear | 0.0797 | 0.0876 | 0.100 |
| 5 | Position-Based | 0.0797 | 0.0878 | 0.100 |
| 6 | Shapley | 0.0810 | 0.0909 | 0.100 |
| 7 | First-Click | 0.0872 | 0.0962 | 0.100 |
| 8 | Markov | 0.1466 | 0.1772 | **0.894** |

### Key Methodological Insight: The Markov-Shapley Trade-off

The Markov chain framework excels at **rank identification** (highest Spearman correlation) but exhibits **magnitude exaggeration** (highest MAE). The Shapley framework produces **balanced magnitude estimates** but **compresses rankings**. The hybrid model resolves this trade-off by combining both paradigms.

> *"Markov knows which channels matter most but exaggerates how much; Shapley approximates how much each matters but obscures which matters most."*

---

## Limitations

This framework is subject to several limitations discussed in detail in Chapter 5 of the thesis:

1. **Synthetic data** --- External validity depends on real-world validation
2. **Five-channel constraint** --- Exact Shapley computation is exponential in channel count
3. **Static ground truth** --- Real-world channel effects evolve over time
4. **Cookieless future** --- Tracking infrastructure dependencies
5. **Touchpoint-centric assumption** --- Does not model creative quality, brand perception, contextual factors, or social media response dynamics

Future work directions include multi-modal attribution incorporating creative quality, context-aware modeling, real-world validation, and decision support system integration.

---

## Citation

If you use this work in your research, please cite the thesis:

```bibtex
@thesis{korkut2026hybrid,
  author       = {Korkut, Berkay},
  title        = {A Hybrid {Markov} Chain and {Shapley} Value Approach to
                  Multi-Touch Attribution and Budget Optimization in
                  Digital Marketing},
  school       = {Istanbul University, Faculty of Economics,
                  Management Information Systems},
  year         = {2026},
  type         = {Undergraduate Thesis},
  url          = {https://github.com/brky48/hybrid-mta-attribution}
}
```

---

## License

This project is licensed under the MIT License --- see the [LICENSE](LICENSE) file for details.

The MIT License permits use, modification, and redistribution of this code for both academic and commercial purposes, subject to the inclusion of the original copyright notice.

---

## Author

**Berkay Korkut**
Undergraduate Student, Management Information Systems
Faculty of Economics, Istanbul University
2026

GitHub: [@brky48](https://github.com/brky48)

---

## Acknowledgments

This thesis was developed as part of the Graduation Project requirements at Istanbul University. The methodological framework draws on the foundational contributions of:

- Anderl et al. (2016) on Markov-based attribution
- Shapley (1953) and Shao & Li (2011) on Shapley value attribution
- Jin et al. (2017) and Chan & Perry (2017) on Marketing Mix Modeling
- Storn & Price (1997) on Differential Evolution

For the complete bibliography (72 academic sources), please refer to the thesis document.

---

*Last updated: May 2026*
