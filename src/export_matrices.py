#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Matrices for Thesis Appendices B and C
==============================================

This utility script exports two matrices used in the thesis appendices:

    - Appendix B: Markov Chain Transition Matrix (8x8)
    - Appendix C: Channel Interaction Matrix (5x5)

The Markov transition matrix is constructed from the synthetic customer
journey dataset using the same Laplace-smoothed procedure as in
markov_chain.py (alpha=1.0). The channel interaction matrix is
reconstructed from the CHANNEL_INTERACTIONS dictionary defined in
data_generator.py.

Usage:
    Run after data_generator.py has produced customer_journey_data.csv.

Output files:
    - markov_transition_matrix.csv
    - channel_interaction_matrix.csv

Part of the thesis:
    "A Hybrid Markov Chain and Shapley Value Approach to Multi-Touch
    Attribution and Budget Optimization in Digital Marketing"
    Istanbul University, Management Information Systems, 2026.
"""

import numpy as np
import pandas as pd

# Import from existing modules
from data_generator import CHANNELS, CHANNEL_INTERACTIONS
from markov_chain import load_and_build_paths, build_transition_matrix


# =============================================================================
# 1. MARKOV TRANSITION MATRIX (Appendix B)
# =============================================================================

print("="*65)
print("  APPENDIX B: Markov Transition Matrix")
print("="*65)

print("\n[1/2] Loading customer journey data...")
path_df = load_and_build_paths('customer_journey_data.csv')
print(f"      {len(path_df):,} users loaded.")

print("\n[2/2] Building transition matrix (Laplace alpha=1.0)...")
transition_matrix = build_transition_matrix(path_df, CHANNELS, alpha=1.0)

# Save to CSV
transition_matrix.round(4).to_csv('markov_transition_matrix.csv', float_format='%.4f')
print("      Saved: 'markov_transition_matrix.csv'")

print("\n--- Transition Matrix (4 decimal places) ---")
print(transition_matrix.round(4).to_string())


# =============================================================================
# 2. CHANNEL INTERACTION MATRIX (Appendix C)
# =============================================================================

print("\n" + "="*65)
print("  APPENDIX C: Channel Interaction Matrix")
print("="*65)

# Build a 5x5 symmetric matrix (diagonal = 1.0 by convention,
# since a channel has no self-interaction)
n = len(CHANNELS)
interaction_matrix = np.ones((n, n))

for (c1, c2), value in CHANNEL_INTERACTIONS.items():
    i = CHANNELS.index(c1)
    j = CHANNELS.index(c2)
    interaction_matrix[i, j] = value
    interaction_matrix[j, i] = value  # Symmetric

df_interaction = pd.DataFrame(
    interaction_matrix,
    index=CHANNELS,
    columns=CHANNELS
)

# Save to CSV
df_interaction.round(3).to_csv('channel_interaction_matrix.csv', float_format='%.3f')
print("\n      Saved: 'channel_interaction_matrix.csv'")

print("\n--- Interaction Matrix (synergy multipliers) ---")
print(df_interaction.round(3).to_string())

print("\n" + "="*65)
print("  All matrices exported successfully.")
print("="*65)
