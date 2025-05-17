import warnings

import cvxpy as cp
import numpy as np
import pypfopt.objective_functions as objective_functions
from pypfopt.efficient_frontier import EfficientFrontier


class AlmEfficientFrontier(EfficientFrontier):
    def __init__(self, *args, n_benchmark_assets=0, benchmkar_weights=[], **kwargs):
        super().__init__(*args, **kwargs)
        assert len(benchmkar_weights) == n_benchmark_assets
        assert n_benchmark_assets < self.n_assets
        self.n_benchmark_assets = n_benchmark_assets
        self.benchmark_weights = benchmkar_weights

    def _make_weight_sum_constraint(self, is_market_neutral):
        """
        Helper method to make the weight sum constraint. If market neutral,
        validate the weights proided in the constructor.
        """

        standard_assets_selector = [
            i < self.n_assets - self.n_benchmark_assets for i in range(self.n_assets)
        ]

        if is_market_neutral:
            #  Check and fix bounds
            portfolio_possible = np.any(self._lower_bounds < 0)
            if not portfolio_possible:
                warnings.warn(
                    "Market neutrality requires shorting - bounds have been amended",
                    RuntimeWarning,
                )
                self._map_bounds_to_constraints((-1, 1))
                # Delete original constraints
                del self._constraints[0]
                del self._constraints[0]

            # To make the sum on a subset of w, you can use indexing or slicing
            # For example, if you want to sum only the first n elements:
            # self.add_constraint(lambda w: cp.sum(w[:n]) == 0)
            # Or if you have specific indices in a list called 'indices':
            self.add_constraint(lambda w: cp.sum(w[standard_assets_selector]) == 0)
            # self.add_constraint(lambda w: cp.sum(w) == 0)
        else:
            self.add_constraint(lambda w: cp.sum(w[standard_assets_selector]) == 1)

        for i in range(self.n_benchmark_assets):
            self.add_constraint(
                lambda w: w[(self.n_assets - self.n_benchmark_assets) + i]
                == self.benchmark_weights[i]
            )

        self._market_neutral = is_market_neutral

    def min_volatility(self):
        """
        Minimise volatility.

        :return: asset weights for the volatility-minimising portfolio
        :rtype: OrderedDict
        """
        self._objective = objective_functions.portfolio_variance(
            self._w, self.cov_matrix
        )
        for obj in self._additional_objectives:
            self._objective += obj

        self._make_weight_sum_constraint(is_market_neutral=False)
        return self._solve_cvxpy_opt_problem()
