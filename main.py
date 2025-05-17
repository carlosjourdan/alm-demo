import warnings

import dash_daq as daq
from dash import Dash, dcc, html

from alm import (
    bounds_table,
    new_risk_factors_div,
    resulting_portfolios_div,
    statistics_div,
)
from black_litterman import black_litterman_outputs, views_div

warnings.filterwarnings("ignore", category=Warning)

app = Dash("ALM Black Litterman")


MATHJAX_CDN = '''
https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/
MathJax.js?config=TeX-MML-AM_CHTML'''

external_scripts = [
                    {'type': 'text/javascript',
                     'id': 'MathJax-script',
                     'src': MATHJAX_CDN,
                     },
                    ]


app.layout = [
    daq.ToggleSwitch(id="allow-shorting-toggle", label="Allow shorting", value=False),
    html.H1("Black-Litterman"),
    dcc.Markdown(
        """
        $E(R) = [\\tau\\Sigma^{-1} + P^T\\Omega^{-1}P]^{-1}[(\\tau \\Sigma)^{-1} \\Pi + P^T \\Omega^{-1} Q]$

        - $E(R)$ is a Nx1 vector of expected returns, where *N* is the number of assets.
        - $E(R)$ is a Nx1 vector of expected returns, where *N* is the number of assets.
        - $Q$ is a Kx1 vector of views.
        - $P$ is the KxN **picking matrix** which maps views to the universe of assets.
        Essentially, it tells the model which view corresponds to which asset(s).
        - $\\Omega$ is the KxK **uncertainty matrix** of views. 
        - $\\Pi$ is the Nx1 vector of prior expected returns. 
        - $\\Sigma$ is the NxN covariance matrix of asset returns (as always)
        - $\\tau$ is a scalar tuning constant. 


        To calculate the market-implied returns, we then use the following formula, where we arbitrarilly set $\\delta = 2.5$:

        $\\Pi = \\delta \\Sigma w_{mkt}$


        """
        , mathjax=True
    ),
    black_litterman_outputs,
    dcc.Markdown(
        """
        The table below is used to input views on (a combination of) asset returns.
        To calibrate the $\\Omega$ matrix, we set $\\tau = 0.01$ and use Idzorek's method, to calibrate confidence levels on a 0-1 scale, where

        $\\alpha =(1-confidence)/confidence$

        $\\Omega = \\tau\\alpha*P*\\Sigma*P^T$

        """
        , mathjax=True
    ),
    views_div,
    html.H1("ALM"),
    dcc.Markdown("""
                 
        The classical mean-variance optimisation problem is as follows:
                 
        $\\begin{equation*}
        \\begin{aligned}
        & \\underset{w}{\\text{minimise}} & & w^t*\\Sigma*w \\\\
        & \\text{subject to} & & \\Pi^t*w = \mu\\\\
        &&& \sum_{i} w_i = 1,\\\\
        &&& 0 \\leq w_i \\leq 1 \\text{ (or } -1 \\leq w_i \\leq 1 \\text{ if shorting is allowed)}\\\\\\
        \\end{aligned}
        \\end{equation*}$
                 
        If an investor has as benchmark that is different than the risk-free rate, we can adjust the objective function to minimize expected variance relative to the benchmark.
                 

        $\\begin{equation*}
        \\begin{aligned}
        & \\underset{w}{\\text{minimise}} & & (w-w_{bl})^t*\\Sigma*(w-w_{bl}) \\\\
        & \\text{subject to} & & \\Pi^t*w = \mu\\\\
        &&& \sum_{i} w_i = 1,\\\\
        &&& 0 \\leq w_i \\leq 1 \\text{ (or } -1 \\leq w_i \\leq 1 \\text{ if shorting is allowed)}\\\\\\
        \\end{aligned}
        \\end{equation*}$
    """, mathjax=True),
    resulting_portfolios_div,
    new_risk_factors_div,
    html.Div(children=[statistics_div], style={"display": "none"}),
    html.Div(children=[bounds_table], style={"display": "none"}),
]


if __name__ == "__main__":
    app.run(debug=True)
