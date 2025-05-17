import base64
import io
import json
import math
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, callback, dash_table, dcc, html
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import cov_to_corr

from AlmEfficientFrontier import AlmEfficientFrontier
from data import corr_matrix, cov_matrix, risk_factors
from globals import global_style_cell, output_table_params, percentage, seed


def df_to_picke_bytes_base64(df):
    buffer = io.BytesIO()
    df.to_pickle(buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def df_from_picke_bytes_base64(s):
    return pickle.loads(base64.b64decode(s))


new_risk_factors_input_table = dash_table.DataTable(
    id="new-risk-factorsinput-table",
    style_cell=global_style_cell,
    editable=True,
    columns=[
        {
            "id": "name",
            "name": "Name",
            "type": "text",
        },
        {
            "id": "exposure",
            "name": "Exposure",
            "type": "numeric",
            "format": percentage,
        },
    ]
    + [{"id": i, "name": i, "type": "numeric"} for i in risk_factors]
    + [
        {
            "id": "idiosyncratic_vol",
            "name": "sigma(epsilon)",
            "type": "numeric",
            "format": percentage,
        },
        {
            "id": "alpha",
            "name": "alpha",
            "type": "numeric",
            "format": percentage,
        },
    ],
    data=[
        {
            "name": "VENTURE CAPITAL",
            "idiosyncratic_vol": 1.6,
            "exposure": 0.2,
            "alpha": 0.05,
            **{
                risk_factor: 1 if risk_factor == "MAG7" else 0
                for risk_factor in risk_factors
            },
        }
    ]
    if seed
    else [],
    row_deletable=True,
)


new_risk_factors_output_table = dash_table.DataTable(
    id="new-risk-factorsoutput-table",
    columns=[
        {
            "id": "name",
            "name": "Name",
            "type": "text",
            "hideable": True,
        },
        {
            "id": "total_vol",
            "name": "Sigma",
            "type": "numeric",
            "format": percentage,
        },
        {
            "id": "expected_return",
            "name": "E[R-Rf]",
            "type": "numeric",
            "format": percentage,
        },
    ],
    data=[{"total_vol": 0.236, "expected_return": 0.129}] if seed else [],
    hidden_columns=["name"],
    **output_table_params,
)


@callback(
    Output("new-risk-factorsinput-table", "data"),
    Output("new-risk-factorsoutput-table", "data", allow_duplicate=True),
    Output("bounds-table", "data", allow_duplicate=True),
    Input("add-risk-factor-button", "n_clicks"),
    State("new-risk-factorsinput-table", "data"),
    State("new-risk-factorsoutput-table", "data"),
    State("bounds-table", "data"),
    State("new-risk-factor-name", "value"),
    prevent_initial_call=True,
)
def add_risk_factor(
    n_clicks, input_rows, output_rows, bounds_table, new_risk_factor_name
):
    if len(new_risk_factor_name) == 0:
        return input_rows, output_rows, bounds_table

    if n_clicks > 0:
        input_rows.append(
            dict(
                name=new_risk_factor_name,
                idiosyncratic_vol=0.05,
                alpha=0,
                **{risk_factor: 0 for risk_factor in risk_factors},
            )
        )
        output_rows.append(
            dict(total_vol=0, expected_return=0, exposure=0, name=new_risk_factor_name)
        )
    return input_rows, output_rows, bounds_table


@callback(
    Output("new-risk-factorsoutput-table", "data", allow_duplicate=True),
    Input("new-risk-factorsinput-table", "data_timestamp"),
    Input("blacklitterman-model-store", "data"),
    State("new-risk-factorsinput-table", "data"),
    State("new-risk-factorsoutput-table", "data"),
    prevent_initial_call=True,
)
def update_risk_factors_output_table(
    timestamp, blacklitterman_model, input_rows, output_rows
):
    output_rows_map = {row["name"]: i for i, row in enumerate(output_rows)}
    input_rows_map = {row["name"]: i for i, row in enumerate(input_rows)}

    for deleted_view in [
        deleted_view
        for deleted_view in output_rows_map.keys()
        if deleted_view not in input_rows_map.keys()
    ]:
        output_rows.pop(output_rows_map[deleted_view])

    for i, input_row in enumerate(input_rows):
        weights = pd.Series({key: input_row[key] for key in risk_factors})

        output_rows[i]["total_vol"] = input_row["idiosyncratic_vol"] + np.sqrt(
            weights.dot(cov_matrix.loc[risk_factors, risk_factors]).dot(weights)
        )

        output_rows[i]["expected_return"] = input_row["alpha"] + weights.dot(
            blacklitterman_model["returns"]
        )

    return output_rows


new_risk_factors_div = html.Div(
    [
        html.Div(
            [
                dcc.Input(
                    id="new-risk-factor-name",
                    placeholder="Enter a new risk factor name...",
                    value="",
                    style={"padding": 10},
                ),
                html.Button("Add Risk Factor", id="add-risk-factor-button", n_clicks=0),
            ],
            style={"height": 50},
        ),
        html.Div(
            children=[
                html.Div(
                    style={"display": "inline-block"},
                    children=[new_risk_factors_input_table],
                ),
                html.Div(
                    style={"display": "inline-block"},
                    children=[new_risk_factors_output_table],
                ),
            ],
        ),
    ]
)


@callback(
    Output("extended-risk-factors-store", "data"),
    Input("blacklitterman-model-store", "data"),
    Input("new-risk-factorsinput-table", "data"),
    Input("new-risk-factorsoutput-table", "data"),
)
def update_extended_risk_factors_store(
    blacklitterman_model, new_risk_factors_input, new_risk_factors_output
):
    new_factors = {
        row["name"]: {
            "vol": row["idiosyncratic_vol"],
            "eps": f"eps_{row['name']}",
            "expected_return": new_risk_factors_output[i]["expected_return"],
            "exposure": row["exposure"],
        }
        for (i, row) in enumerate(new_risk_factors_input)
    }
    new_epsilons = [f"eps_{row['name']}" for row in new_risk_factors_input]

    cov_matrix_with_epsilons = cov_matrix.copy()
    # Extend the covariance matrix to include epsilon factors
    # First, add new rows and columns for each epsilon
    for epsilon in new_epsilons:
        # Add a new column
        cov_matrix_with_epsilons[epsilon] = 0.0
        # Add a new row
        cov_matrix_with_epsilons.loc[epsilon] = 0.0

    for row in new_risk_factors_input:
        new_epsilon = f"eps_{row['name']}"
        cov_matrix_with_epsilons.loc[new_epsilon, new_epsilon] = (
            row["idiosyncratic_vol"] ** 2
        )

    # Create identity matrix for risk factors
    w = pd.DataFrame(
        np.eye(len(risk_factors)), index=risk_factors, columns=risk_factors
    )

    w[new_epsilons] = 0

    # Create a matrix from new_risk_factors with risk_factors as columns and name as index
    new_factors_new_risk_factors = pd.DataFrame(
        [
            [row[risk_factor] for risk_factor in risk_factors]
            for row in new_risk_factors_input
        ],
        columns=risk_factors,
        index=[row["name"] for row in new_risk_factors_input],
    )

    new_factors_new_risk_factors[new_epsilons] = 0

    for name, factor in new_factors.items():
        new_factors_new_risk_factors.loc[name, factor["eps"]] = 1

    w = pd.concat([w, new_factors_new_risk_factors])

    cov_matrix_extended = w.dot(cov_matrix_with_epsilons).dot(w.T)

    corr_matrix_extended = cov_to_corr(cov_matrix_extended)
    # Check the rank of the covariance matrix
    rank = np.linalg.matrix_rank(cov_matrix_extended)

    extended_risk_factors = list(cov_matrix_extended.index)

    # Create a Series from the new_factors dictionary
    # Extract expected returns and use factor names as index
    new_returns = pd.Series(
        [factor["expected_return"] for name, factor in new_factors.items()],
        index=[name for name, factor in new_factors.items()],
    )

    bl_returns = pd.Series()
    for i, factor in enumerate(risk_factors):
        bl_returns[factor] = blacklitterman_model["returns"][i]

    return json.dumps(
        {
            "rank": int(rank),
            "risk_factors": extended_risk_factors,
            "cov_matrix": df_to_picke_bytes_base64(cov_matrix_extended),
            "corr_matrix": df_to_picke_bytes_base64(corr_matrix_extended),
            "expected_returns": df_to_picke_bytes_base64(
                pd.concat([bl_returns, new_returns])
            ),
            "benchmark_weights": [
                factor["exposure"] for factor in new_risk_factors_input
            ],
        }
    )


def decode_extended_risk_factors(extended_risk_factors):
    extended_risk_factors = json.loads(extended_risk_factors)
    extended_risk_factors["cov_matrix"] = df_from_picke_bytes_base64(
        extended_risk_factors["cov_matrix"]
    )
    extended_risk_factors["corr_matrix"] = df_from_picke_bytes_base64(
        extended_risk_factors["corr_matrix"]
    )
    extended_risk_factors["expected_returns"] = df_from_picke_bytes_base64(
        extended_risk_factors["expected_returns"]
    )
    return extended_risk_factors


@callback(
    Output("risk-factors-extended-heatmap", "figure"),
    Input("extended-risk-factors-store", "data"),
)
def update_correl_heatmap(extended_risk_factors):
    extended_risk_factors = decode_extended_risk_factors(extended_risk_factors)
    return correl_heatmat(
        extended_risk_factors["corr_matrix"], extended_risk_factors["risk_factors"]
    )


def correl_heatmat(correl_matrix, risk_factors):
    return px.imshow(
        correl_matrix,
        labels=dict(x="Asset", y="Asset", color="Correlation"),
        x=risk_factors,
        y=risk_factors,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=True,
    )


statistics_div = html.Div(
    children=[
        dcc.Store(id="extended-risk-factors-store"),
        dcc.Store(id="efficient-frontier-min-vol-store"),
        dcc.Store(id="efficient-frontier-tgt-vol-store"),
        html.Div(
            style={"display": "inline-block"},
            children=[
                dcc.Graph(
                    id="risk-factors-heatmap",
                    figure=correl_heatmat(corr_matrix, risk_factors),
                )
            ],
        ),
        html.Div(
            style={"display": "inline-block"},
            children=[
                dcc.Graph(id="risk-factors-extended-heatmap"),
            ],
        ),
    ]
)


bounds_table = dash_table.DataTable(
    id="bounds-table",
    style_cell=global_style_cell,
    style_table={
        "width": "500px",
    },
    editable=True,
    columns=[
        {
            "id": "factor",
            "name": "Factor",
            "type": "text",
            "editable": False,
        },
        {
            "id": "min",
            "name": "Min",
            "type": "numeric",
            "format": percentage,
        },
        {
            "id": "max",
            "name": "Max",
            "type": "numeric",
            "format": percentage,
        },
    ],
    data=[{"factor": risk_factor, "min": 0, "max": 1} for risk_factor in risk_factors]
    if seed
    else [],
)


@callback(
    Output("efficient-frontier-min-vol-store", "data"),
    Input("blacklitterman-model-store", "data"),
    Input("extended-risk-factors-store", "data"),
    Input("bounds-table", "data"),
    Input("allow-shorting-toggle", "value"),
)
def update_min_variance_portfolios(
    blacklitterman_model, extended_risk_factors, bounds_table, allow_shorting
):
    extended_risk_factors = decode_extended_risk_factors(extended_risk_factors)

    ef_black_litterman = EfficientFrontier(
        cov_matrix=cov_matrix,
        expected_returns=blacklitterman_model["returns"],
        weight_bounds=(-1, 1) if allow_shorting else (0, 1),
    )

    ef_alm = AlmEfficientFrontier(
        cov_matrix=extended_risk_factors["cov_matrix"],
        expected_returns=extended_risk_factors["expected_returns"],
        n_benchmark_assets=len(extended_risk_factors["expected_returns"])
        - len(blacklitterman_model["returns"]),
        benchmkar_weights=extended_risk_factors["benchmark_weights"],
        weight_bounds=(-1, 1) if allow_shorting else (0, 1),
    )

    bl_weights = ef_black_litterman.min_volatility()
    bl_weights_series = pd.Series(bl_weights)
    bl_vol = np.sqrt(bl_weights_series.dot(cov_matrix).dot(bl_weights_series))

    alm_weights = ef_alm.min_volatility()
    alm_weights_series = pd.Series(alm_weights)
    alm_vol = np.sqrt(
        alm_weights_series.dot(extended_risk_factors["cov_matrix"]).dot(
            alm_weights_series
        )
    )

    return {
        "bl_weights": bl_weights,
        "bl_vol": bl_vol,
        "alm_weights": alm_weights,
        "alm_vol": alm_vol,
    }


@callback(
    Output("min-variance-portfolios-graph", "figure"),
    Input("efficient-frontier-min-vol-store", "data"),
)
def plot_min_variance_portfolios(efficient_frontier_store):
    bl_weights = efficient_frontier_store["bl_weights"]
    alm_weights = efficient_frontier_store["alm_weights"]
    alm_weights = {k: v for k, v in alm_weights.items() if k in bl_weights.keys()}
    df_weights = pd.DataFrame({"Black-Litterman": bl_weights, "ALM": alm_weights})
    df_melted_weights = df_weights.reset_index().melt(
        id_vars="index", value_name="Min Variance Weights", var_name="Model"
    )

    chart_weights = px.bar(
        df_melted_weights,
        x="Min Variance Weights",
        y="index",
        color="Model",
        orientation="h",
        labels=dict(x="Min Variance Weights", y="Asset", index="Asset"),
        barmode="group",
    )
    chart_weights.update_xaxes(tickformat=". h%")
    chart_weights.update_layout(
        yaxis=dict(
            showgrid=True,
        ),
    )

    chart_weights.update_layout(
        showlegend=False,
        margin=dict(r=5),
        yaxis=dict(
            showgrid=True,
        ),
    )

    return chart_weights


# Callback to display the slider value
@callback(
    Output("risk-target-slider", "min"),
    Output("risk-target-slider", "max"),
    Input("efficient-frontier-min-vol-store", "data"),
)
def update_slider_range(efficient_frontier_min_vol_store):
    min_vol = max(
        efficient_frontier_min_vol_store["bl_vol"],
        efficient_frontier_min_vol_store["alm_vol"],
    )
    min_vol = math.ceil(min_vol * 100) / 100  # Round up to nearest 1%
    max_vol = min_vol + 0.20
    return min_vol, max_vol


@callback(
    Output("efficient-frontier-tgt-vol-store", "data"),
    Input("risk-target-slider", "value"),
    Input("blacklitterman-model-store", "data"),
    Input("extended-risk-factors-store", "data"),
    Input("bounds-table", "data"),
    Input("allow-shorting-toggle", "value"),
)
def update_target_volatility_portfolios(
    risk_target,
    blacklitterman_model,
    extended_risk_factors,
    bounds_table,
    allow_shorting,
):
    extended_risk_factors = decode_extended_risk_factors(extended_risk_factors)

    ef_black_litterman = EfficientFrontier(
        cov_matrix=cov_matrix,
        expected_returns=blacklitterman_model["returns"],
        weight_bounds=(-1, 1) if allow_shorting else (0, 1),
    )

    ef_alm = AlmEfficientFrontier(
        cov_matrix=extended_risk_factors["cov_matrix"],
        expected_returns=extended_risk_factors["expected_returns"],
        n_benchmark_assets=len(extended_risk_factors["expected_returns"])
        - len(blacklitterman_model["returns"]),
        benchmkar_weights=extended_risk_factors["benchmark_weights"],
        weight_bounds=(-1, 1) if allow_shorting else (0, 1),
    )

    bl_weights = ef_black_litterman.efficient_risk(risk_target)
    bl_weights_series = pd.Series(bl_weights)
    bl_vol = np.sqrt(bl_weights_series.dot(cov_matrix).dot(bl_weights_series))

    alm_weights = ef_alm.efficient_risk(risk_target)
    alm_weights_series = pd.Series(alm_weights)
    alm_vol = np.sqrt(
        alm_weights_series.dot(extended_risk_factors["cov_matrix"]).dot(
            alm_weights_series
        )
    )

    return {
        "bl_weights": bl_weights,
        "bl_vol": bl_vol,
        "alm_weights": alm_weights,
        "alm_vol": alm_vol,
    }


@callback(
    Output("tgt-variance-portfolios-graph", "figure"),
    Input("efficient-frontier-tgt-vol-store", "data"),
)
def plot_target_variance_portfolios(efficient_frontier_store):
    bl_weights = efficient_frontier_store["bl_weights"]
    alm_weights = efficient_frontier_store["alm_weights"]
    alm_weights = {k: v for k, v in alm_weights.items() if k in bl_weights.keys()}
    df_weights = pd.DataFrame({"Black-Litterman": bl_weights, "ALM": alm_weights})
    df_melted_weights = df_weights.reset_index().melt(
        id_vars="index", value_name="Target Variance Weights", var_name="Model"
    )

    chart_weights = px.bar(
        df_melted_weights,
        x="Target Variance Weights",
        y="index",
        color="Model",
        orientation="h",
        labels=dict(x="Target Variance Weights", y="Asset", index="Asset"),
        barmode="group",
    )
    chart_weights.update_xaxes(tickformat=". h%")
    chart_weights.update_layout(
        yaxis=dict(
            showgrid=True,
        ),
    )

    chart_weights.update_layout(
        margin=dict(l=5),
        yaxis=dict(
            showticklabels=False,
            showgrid=True,
            zeroline=False,
            showline=False,
            title=None,
        ),
    )

    return chart_weights


resulting_portfolios_div = html.Div(
    children=[
        html.Div(
            style={"width": "900px"},
            children=[
                html.H3("Risk Target"),
                dcc.Slider(
                    id="risk-target-slider",
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.05,
                ),
            ],
        ),
        html.Div(
            style={"display": "inline-block"},
            children=[
                dcc.Graph(
                    id="min-variance-portfolios-graph",
                    style={"height": 500, "width": 750},
                )
            ],
        ),
        html.Div(
            style={"display": "inline-block"},
            children=[
                dcc.Graph(
                    id="tgt-variance-portfolios-graph",
                    style={"height": 500, "width": 850},
                )
            ],
        ),
        # html.Div(
        #     style={"display": "inline-block"},
        #     children=[
        #         dcc.Graph(id="risk-factors-extended-heatmap"),
        #     ],
        # ),
    ]
)
