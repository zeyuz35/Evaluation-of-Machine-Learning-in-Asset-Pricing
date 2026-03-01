import numpy as np
import pytest
from main import sim_panel_data, sim_tune_statistics

def test_simulation_run_g1():
    np.random.seed(42)
    sim_N = 1
    char_rho_a = 0.5
    char_rho_b = 1
    cross_corr = 0
    cross_corr_degree = 0.01
    A1 = np.array([
        [0.95, 0, 0],
        [0, 0.95, 0],
        [0, 0, 0.95]
    ])
    xt_multi = 1
    g_function = "g1"
    theta = np.array([[0.015, 0.015, 0.015]])
    error_sv = 0
    error_ep_sd = 0.05
    error_omega = -0.736
    error_gamma = 0.9
    error_w = 0.363
    error_v_sd = 0.05
    predictor_format = "kronecker"

    sim_list = sim_panel_data(sim_N, char_rho_a, char_rho_b, cross_corr, cross_corr_degree,
                       A1, xt_multi, g_function, theta, error_sv, error_ep_sd,
                       error_omega, error_gamma, error_w, error_v_sd, predictor_format)

    stats_df = sim_tune_statistics(sim_list)

    assert len(stats_df) == 1
    assert 'time_series_fitted.rsquare' in stats_df.columns
    assert 'annual_vol' in stats_df.columns
    assert 'true_rsquare' in stats_df.columns
    assert 'cross_section_rsquare' in stats_df.columns

    # Make sure values are not NaN and roughly what we'd expect
    assert not stats_df['time_series_fitted.rsquare'].isna().any()
    assert not stats_df['annual_vol'].isna().any()
    assert not stats_df['true_rsquare'].isna().any()
    assert not stats_df['cross_section_rsquare'].isna().any()

    assert stats_df['annual_vol'][0] > 0
