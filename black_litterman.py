import pandas as pd
import plotly.express as px
from dash import Input, Output, State, callback, dash_table, dcc, html
from pypfopt import BlackLittermanModel, EfficientFrontier, black_litterman

from data import cov_matrix, market_caps, risk_factors
from globals import global_style_cell, output_table_params, percentage, seed

delta = 2.5
mkt_returns = black_litterman.market_implied_prior_returns(
    market_caps, delta, cov_matrix
)

views_input_table = dash_table.DataTable(
    id="views-input-table",
    data=[
        {
            "view": "MAG Rocks",
            "expected_return": 0.05,
            "confidence": 0.5,
            **{
                risk_factor: 1
                if risk_factor == "MAG7"
                else -1
                if risk_factor == "SP493"
                else 0
                for risk_factor in risk_factors
            },
        }
    ]
    if seed
    else [],
    editable=True,
    style_cell=global_style_cell,
    columns=[
        {
            "id": "view",
            "name": "View",
            "type": "text",
            "editable": False,
        }
    ]
    + [{"id": i, "name": i, "type": "numeric"} for i in risk_factors]
    + [
        {
            "id": "expected_return",
            "name": "E[R-Rf] view",
            "type": "numeric",
            "format": percentage,
        },
        {
            "id": "confidence",
            "name": "Confidence",
            "type": "numeric",
            "format": percentage,
        },
    ],
    row_deletable=True,
)

views_output_table = dash_table.DataTable(
    id="views-output-table",
    data=[{"mkt_return": mkt_returns.loc["MAG7"] - mkt_returns.loc["SP493"]}]
    if seed
    else [],
    columns=[
        {
            "id": "view",
            "name": "View",
            "type": "text",
            "editable": False,
            "hideable": True,
        },
        {
            "id": "mkt_return",
            "name": "E[R-Rf] Market",
            "type": "numeric",
            "format": percentage,
        },
    ],
    hidden_columns=["view"],
    **output_table_params,
)

views_div = html.Div(
    [
        html.Div(
            [
                dcc.Input(
                    id="new-view-name",
                    placeholder="Enter a new view name...",
                    value="",
                    style={"padding": 10},
                ),
                html.Button("Add view", id="add-view-button", n_clicks=0),
            ],
            style={"height": 50},
        ),
        html.Div(
            children=[
                html.Div(
                    style={"display": "inline-block"},
                    children=[views_input_table],
                ),
                html.Div(
                    style={"display": "inline-block"},
                    children=[views_output_table],
                ),
            ],
        ),
    ]
)


@callback(
    Output("views-input-table", "data"),
    Output("views-output-table", "data", allow_duplicate=True),
    Input("add-view-button", "n_clicks"),
    State("views-input-table", "data"),
    State("views-output-table", "data"),
    State("new-view-name", "value"),
    prevent_initial_call=True,
)
def add_view(n_clicks, input_rows, output_rows, new_view_name):
    if len(new_view_name) == 0:
        return input_rows, output_rows

    if n_clicks > 0:
        input_rows.append(
            dict(
                view=new_view_name,
                expected_return=0,
                confidence=0.5,
                **{risk_factor: 0 for risk_factor in risk_factors},
            )
        )
        output_rows.append(dict(view=new_view_name, mkt_return=0))
    return input_rows, output_rows


@callback(
    Output("views-output-table", "data", allow_duplicate=True),
    Input("views-input-table", "data_timestamp"),
    State("views-input-table", "data"),
    State("views-output-table", "data"),
    prevent_initial_call=True,
)
def update_views(timestamp, input_rows, output_rows):
    output_rows_map = {row["view"]: i for i, row in enumerate(output_rows)}
    input_rows_map = {row["view"]: i for i, row in enumerate(input_rows)}

    for deleted_view in [
        deleted_view
        for deleted_view in output_rows_map.keys()
        if deleted_view not in input_rows_map.keys()
    ]:
        output_rows.pop(output_rows_map[deleted_view])

    for i, input_row in enumerate(input_rows):
        weights = pd.Series({key: input_row[key] for key in risk_factors})

        output_rows[i]["mkt_return"] = weights.dot(mkt_returns)

    return output_rows


@callback(
    Output("blacklitterman-model-store", "data"),
    Input("views-input-table", "data_timestamp"),
    Input("views-input-table", "data"),
    Input("allow-shorting-toggle", "value"),
)
def update_bl_model_returns(timestamp, views, allow_shorting):
    if len(views) == 0:
        return {
            "returns": mkt_returns,
            "weights": market_caps / market_caps.sum(),
        }

    P = pd.DataFrame(
        [
            {risk_factor: view[risk_factor] for risk_factor in risk_factors}
            for view in views
        ]
    ).to_numpy()

    Q = pd.Series([view["expected_return"] for view in views]).to_numpy()
    sigma = cov_matrix.to_numpy()
    view_confidences = [view["confidence"] for view in views]

    omega = BlackLittermanModel.idzorek_method(
        view_confidences=view_confidences,
        cov_matrix=sigma,
        pi=mkt_returns,
        Q=Q,
        P=P,
        tau=0.01,
        risk_aversion=2.5,
    )

    bl_model = BlackLittermanModel(
        cov_matrix=cov_matrix, pi=mkt_returns, P=P, Q=Q, omega=omega
    )

    bl_returns = bl_model.bl_returns()

    ef_black_litterman = EfficientFrontier(
        cov_matrix=cov_matrix,
        expected_returns=bl_returns,
        weight_bounds=(-1, 1) if allow_shorting else (0, 1),
    )
    bl_weights = ef_black_litterman.max_sharpe()

    return {
        "returns": bl_returns,
        "weights": bl_weights,
    }


@callback(
    Output(component_id="black-litterman-returns-graph", component_property="figure"),
    Output(component_id="black-litterman-weights-graph", component_property="figure"),
    Input(component_id="blacklitterman-model-store", component_property="data"),
)
def update_returns_graph(bl_model):
    bl_returns = bl_model["returns"]
    # Create a DataFrame with both market returns and Black-Litterman returns
    df = pd.DataFrame({"Market": mkt_returns, "Black-Litterman": bl_returns})

    # Melt the DataFrame to create a format suitable for grouped bar chart
    df_melted = df.reset_index().melt(
        id_vars="index",
        value_vars=["Market", "Black-Litterman"],
        var_name="Model",
        value_name="E[R-Rf]",
    )

    chart_returns = px.bar(
        df_melted,
        x="E[R-Rf]",
        y="index",
        color="Model",
        orientation="h",
        labels=dict(x="E[R-Rf]", y="Asset", index="Asset"),
        barmode="group",
    )
    chart_returns.update_xaxes(tickformat=".00%", dtick=0.01)
    chart_returns.update_layout(
        showlegend=False,
        margin=dict(r=5),
        yaxis=dict(
            showgrid=True,
        ),
    )

    df_weights = pd.DataFrame(
        {
            "Market": market_caps / market_caps.sum(),
            "Black-Litterman": bl_model["weights"],
        }
    )

    df_melted_weights = df_weights.reset_index().melt(
        id_vars="index",
        value_vars=["Market", "Black-Litterman"],
        var_name="Model",
        value_name="Weights",
    )

    chart_weights = px.bar(
        df_melted_weights,
        x="Weights",
        y="index",
        color="Model",
        orientation="h",
        labels=dict(x="Weights", y="Asset", index="Asset"),
        barmode="group",
    )
    chart_weights.update_xaxes(tickformat=".00%", dtick=0.05)
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

    return chart_returns, chart_weights


black_litterman_outputs = html.Div(
    children=[
        dcc.Store(id="blacklitterman-model-store"),
        html.Div(
            style={"display": "inline-block"},
            children=[
                dcc.Graph(
                    id="black-litterman-returns-graph",
                    style={"height": 500, "width": 750},
                )
            ],
        ),
        html.Div(
            style={"display": "inline-block"},
            children=[
                dcc.Graph(
                    id="black-litterman-weights-graph",
                    style={"height": 500, "width": 850},
                )
            ],
        ),
    ]
)
