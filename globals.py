from dash import dash_table

seed = False

percentage = dash_table.FormatTemplate.percentage(2)

global_style_cell = {
    "width": "100px",
    "minWidth": "120px",
    "maxWidth": "120px",
    "overflow": "hidden",
    "textOverflow": "ellipsis",
}

output_table_params = {
    "style_cell": global_style_cell,
    "editable": False,
    "cell_selectable": False,
    "style_header": {
        "backgroundColor": "#2c3e50",
        "color": "white",
        "fontWeight": "bold",
    },
    "css": [{"selector": ".show-hide", "rule": "display: none"}],
}
