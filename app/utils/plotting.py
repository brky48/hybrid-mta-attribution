"""
Plotting Module
===============

Reusable Plotly chart functions for the Hybrid Attribution Explorer.
All charts use a consistent dark theme and color palette derived
from the thesis visualizations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.data_loader import (
    CHANNELS,
    CHANNEL_COLORS,
    CHANNEL_DISPLAY,
    MODEL_INFO,
)


# ============================================================================
# THEME & STYLING
# ============================================================================

PLOTLY_TEMPLATE = "plotly_dark"

LAYOUT_DEFAULTS = {
    'paper_bgcolor'  : 'rgba(0,0,0,0)',
    'plot_bgcolor'   : 'rgba(0,0,0,0)',
    'font'           : {'family': 'sans-serif', 'size': 13, 'color': '#E5E7EB'},
    'margin'         : {'l': 50, 'r': 30, 't': 60, 'b': 50},
    'hoverlabel'     : {
        'bgcolor': '#1F2937',
        'font_size': 13,
        'font_family': 'sans-serif',
    },
}


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent dark theme to a Plotly figure."""
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(gridcolor='#374151', zerolinecolor='#374151')
    fig.update_yaxes(gridcolor='#374151', zerolinecolor='#374151')
    return fig


# ============================================================================
# ATTRIBUTION COMPARISON CHARTS
# ============================================================================

def plot_attribution_bars(
    df: pd.DataFrame,
    models_to_show: list = None,
    title: str = None,
) -> go.Figure:
    """
    Grouped bar chart comparing attribution distributions across
    multiple models.

    Args:
        df: DataFrame with 'channel' column + model columns
            (each row a channel, each model column attribution %).
        models_to_show: Subset of model columns to display.
        title: Optional chart title.
    """
    if models_to_show is None:
        models_to_show = [c for c in df.columns if c != 'channel']

    fig = go.Figure()

    for model in models_to_show:
        info = MODEL_INFO.get(model, {'name': model, 'color': '#9CA3AF'})
        fig.add_trace(go.Bar(
            x=df['channel'].map(CHANNEL_DISPLAY),
            y=df[model] * 100,  # convert to %
            name=info['name'],
            marker_color=info['color'],
            text=[f"{v*100:.1f}%" for v in df[model]],
            textposition='outside',
            textfont={'size': 10},
            hovertemplate='<b>%{x}</b><br>' +
                          info['name'] + ': %{y:.2f}%<extra></extra>',
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Channel',
        yaxis_title='Attribution Share (%)',
        barmode='group',
        height=500,
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.05,
            'xanchor': 'right',
            'x': 1.0,
        }
    )

    return apply_dark_theme(fig)


def plot_alpha_curve(
    alpha_grid_df: pd.DataFrame,
    current_alpha: float,
    current_mae: float,
) -> go.Figure:
    """
    Plot the MAE-vs-alpha curve with the current alpha marked.

    Args:
        alpha_grid_df: DataFrame with columns 'alpha' and 'mae'.
        current_alpha: Currently selected alpha value.
        current_mae: MAE at the current alpha.
    """
    fig = go.Figure()

    # Main curve
    fig.add_trace(go.Scatter(
        x=alpha_grid_df['alpha'],
        y=alpha_grid_df['mae'],
        mode='lines',
        line={'color': '#60A5FA', 'width': 3},
        fill='tozeroy',
        fillcolor='rgba(96, 165, 250, 0.1)',
        name='MAE curve',
        hovertemplate='α = %{x:.2f}<br>MAE = %{y:.4f}<extra></extra>',
    ))

    # Current alpha marker
    fig.add_trace(go.Scatter(
        x=[current_alpha],
        y=[current_mae],
        mode='markers',
        marker={
            'color': '#DC2626',
            'size': 16,
            'line': {'color': 'white', 'width': 2},
            'symbol': 'circle',
        },
        name=f'Current α = {current_alpha:.2f}',
        hovertemplate='Current α = %{x:.2f}<br>MAE = %{y:.4f}<extra></extra>',
    ))

    # Optimal alpha vertical line
    optimal_alpha = 0.25
    fig.add_vline(
        x=optimal_alpha,
        line_dash='dash',
        line_color='#10B981',
        annotation_text=f'Optimal α* = {optimal_alpha}',
        annotation_position='top',
        annotation_font_color='#10B981',
    )

    fig.update_layout(
        title='Mean Absolute Error vs. Mixing Parameter α',
        xaxis_title='α (Markov weight)',
        yaxis_title='MAE (against Ground Truth)',
        height=400,
        showlegend=False,
    )

    return apply_dark_theme(fig)


# ============================================================================
# BUDGET OPTIMIZATION CHARTS
# ============================================================================

def plot_budget_pie(allocation_df: pd.DataFrame) -> go.Figure:
    """
    Pie chart showing budget allocation across paid channels.

    Args:
        allocation_df: DataFrame with 'channel' and 'allocated_pct'.
    """
    fig = go.Figure(data=[go.Pie(
        labels=allocation_df['channel'].map(CHANNEL_DISPLAY),
        values=allocation_df['allocated_pct'],
        hole=0.4,
        marker={
            'colors': [CHANNEL_COLORS[ch] for ch in allocation_df['channel']],
            'line': {'color': '#111827', 'width': 2},
        },
        textinfo='label+percent',
        textfont={'size': 14, 'color': 'white'},
        hovertemplate='<b>%{label}</b><br>' +
                      'Share: %{percent}<br>' +
                      'Amount: $%{customdata:,.0f}<extra></extra>',
        customdata=allocation_df['allocated_budget'],
    )])

    fig.update_layout(
        title='Optimal Budget Allocation',
        height=400,
        showlegend=False,
    )

    return apply_dark_theme(fig)


def plot_budget_bars(allocation_df: pd.DataFrame, total_budget: float) -> go.Figure:
    """
    Horizontal bar chart showing absolute budget amounts.
    """
    df_sorted = allocation_df.sort_values('allocated_budget', ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_sorted['channel'].map(CHANNEL_DISPLAY),
        x=df_sorted['allocated_budget'],
        orientation='h',
        marker_color=[CHANNEL_COLORS[ch] for ch in df_sorted['channel']],
        text=[f"${v:,.0f}" for v in df_sorted['allocated_budget']],
        textposition='outside',
        textfont={'size': 12, 'color': 'white'},
        hovertemplate='<b>%{y}</b><br>' +
                      'Budget: $%{x:,.0f}<br>' +
                      'Share: %{customdata:.1f}%<extra></extra>',
        customdata=df_sorted['allocated_pct'],
    ))

    fig.update_layout(
        title=f'Absolute Allocation (Total: ${total_budget:,.0f})',
        xaxis_title='Budget (USD)',
        yaxis_title='',
        height=400,
        showlegend=False,
    )

    return apply_dark_theme(fig)


def plot_response_curves(
    allocation_df: pd.DataFrame,
    saturation_params: dict,
    response_function_,
    max_spend: float = 100_000,
) -> go.Figure:
    """
    Plot diminishing-returns response curves for each paid channel,
    with the current allocation marked.
    """
    fig = go.Figure()

    spend_range = np.linspace(0, max_spend, 200)

    for _, row in allocation_df.iterrows():
        ch    = row['channel']
        score = row['score']
        k     = saturation_params[ch]['k']
        hs    = saturation_params[ch]['half_saturation']

        responses = [response_function_(x, score, k, hs) for x in spend_range]

        fig.add_trace(go.Scatter(
            x=spend_range,
            y=responses,
            mode='lines',
            line={'color': CHANNEL_COLORS[ch], 'width': 3},
            name=f'{CHANNEL_DISPLAY[ch]} (k={k}, half={hs/1000:.0f}K)',
            hovertemplate='<b>' + CHANNEL_DISPLAY[ch] + '</b><br>' +
                          'Spend: $%{x:,.0f}<br>' +
                          'Response: %{y:.4f}<extra></extra>',
        ))

        # Current allocation point
        current_spend = row['allocated_budget']
        current_response = row['expected_response']

        fig.add_trace(go.Scatter(
            x=[current_spend],
            y=[current_response],
            mode='markers',
            marker={
                'color': CHANNEL_COLORS[ch],
                'size': 14,
                'line': {'color': 'white', 'width': 2},
                'symbol': 'star',
            },
            showlegend=False,
            hovertemplate='<b>' + CHANNEL_DISPLAY[ch] + ' (Current)</b><br>' +
                          'Spend: $%{x:,.0f}<br>' +
                          'Response: %{y:.4f}<extra></extra>',
        ))

    fig.update_layout(
        title='Diminishing Returns Response Curves<br><sub>Stars indicate current optimal allocations</sub>',
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

    return apply_dark_theme(fig)


# ============================================================================
# PATH EXPLORER CHARTS
# ============================================================================

def plot_path_diagram(path: list) -> go.Figure:
    """
    Horizontal flow diagram visualizing a customer journey.
    """
    if not path:
        fig = go.Figure()
        fig.add_annotation(
            text="Click channel buttons above to build a journey",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font={'size': 16, 'color': '#9CA3AF'},
        )
        fig.update_layout(
            height=200,
            xaxis={'visible': False},
            yaxis={'visible': False},
        )
        return apply_dark_theme(fig)

    n = len(path)
    fig = go.Figure()

    # Nodes
    for i, ch in enumerate(path):
        fig.add_trace(go.Scatter(
            x=[i],
            y=[0],
            mode='markers+text',
            marker={
                'size': 80,
                'color': CHANNEL_COLORS[ch],
                'line': {'color': 'white', 'width': 3},
            },
            text=[CHANNEL_DISPLAY[ch]],
            textposition='bottom center',
            textfont={'size': 12, 'color': 'white'},
            showlegend=False,
            hovertemplate=f'<b>Step {i+1}</b><br>{CHANNEL_DISPLAY[ch]}<extra></extra>',
        ))

    # Arrows between nodes
    for i in range(n - 1):
        fig.add_annotation(
            x=i + 0.5,
            y=0,
            ax=i,
            ay=0,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor='#9CA3AF',
        )

    fig.update_layout(
        title=f'Customer Journey ({n} touchpoint{"s" if n != 1 else ""})',
        xaxis={'visible': False, 'range': [-0.5, n - 0.5]},
        yaxis={'visible': False, 'range': [-1, 1]},
        height=250,
        showlegend=False,
    )

    return apply_dark_theme(fig)


def plot_path_attribution_heatmap(attribution_df: pd.DataFrame) -> go.Figure:
    """
    Heatmap showing how each model attributes credit across
    channels for a given path.
    """
    model_cols = [c for c in attribution_df.columns if c != 'channel']

    z_data = attribution_df[model_cols].values.T * 100  # transpose, percentage

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[CHANNEL_DISPLAY.get(ch, ch) for ch in attribution_df['channel']],
        y=[MODEL_INFO[m]['name'] for m in model_cols],
        colorscale='Viridis',
        text=[[f'{v:.1f}%' for v in row] for row in z_data],
        texttemplate='%{text}',
        textfont={'size': 11, 'color': 'white'},
        hovertemplate='Model: %{y}<br>' +
                      'Channel: %{x}<br>' +
                      'Credit: %{z:.2f}%<extra></extra>',
        colorbar={'title': 'Credit (%)'},
    ))

    fig.update_layout(
        title='Per-Model Credit Assignment for This Journey',
        height=350,
    )

    return apply_dark_theme(fig)
