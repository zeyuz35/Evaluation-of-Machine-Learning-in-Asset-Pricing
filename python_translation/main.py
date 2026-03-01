import numpy as np
import pandas as pd
import scipy.stats as stats
import warnings
from sklearn.metrics import r2_score
import statsmodels.api as sm

warnings.filterwarnings('ignore')
np.random.seed(27935248)

N = 200
P_c = 100
Time = 180

def gen_C_bar(rho_a, rho_b):
    C_bar = np.zeros((N, P_c, Time + 2))
    rho = np.random.uniform(rho_a, rho_b, P_c)

    for t in range(Time + 1):
        for j in range(P_c):
            noise = np.random.normal(0, np.sqrt(1 - rho[j]**2), N)
            C_bar[:, j, t + 1] = C_bar[:, j, t] * rho[j] + noise

    C_bar = C_bar[:, :, 1:]
    return C_bar

def gen_W(lambda_degree):
    Lambda = np.random.normal(0, lambda_degree, (N, 4))
    B = np.dot(Lambda, Lambda.T)
    B = B + 0.1 * np.eye(N)
    W = np.linalg.cholesky(B)
    return W

def gen_C_hat(C_bar, cross_corr_degree):
    W = gen_W(cross_corr_degree)
    C_hat = np.zeros((N, P_c, Time + 1))

    for t in range(Time + 1):
        C_hat[:, :, t] = np.dot(W, C_bar[:, :, t])

    return C_hat

def gen_C(C_matrix):
    C = np.zeros((N, P_c, Time + 1))

    for t in range(Time + 1):
        ranks = stats.rankdata(C_matrix[:, :, t], axis=0)
        C[:, :, t] = (2 / (N * P_c + 1)) * ranks - 1

    return C

def gen_xt(A):
    xt = np.zeros((1, 3, Time + 1))
    Axt = np.copy(xt)
    for t in range(1, Time + 1):
        ut = np.random.normal(0, np.sqrt(1 - 0.95**2), 3)
        Axt[:, :, t] = np.dot(A, Axt[0, :, t-1]) + ut

    return Axt[:, :, 1:]

def gen_xt_univariate():
    xt = np.zeros((1, Time))
    rho = 0.95
    xt[0, 0] = np.random.normal(0, np.sqrt(1 - rho**2))

    for t in range(1, Time):
        xt[0, t] = xt[0, t-1] * rho + np.random.normal(0, np.sqrt(1 - rho**2))

    return xt

def logit(x):
    return 1 / (1 + np.exp(-x))

def gen_g_factor_panel(g_function, C, x):
    if g_function == "g1":
        g_factor_panel = np.zeros((N, 3, Time))
        for i in range(N):
            for t in range(Time):
                if x.shape[0] == 1:
                    g_factor_panel[i, :, t] = np.array([C[i, 0, t], C[i, 1, t], C[i, 2, t] * x[0, 0, t]])
                else:
                    g_factor_panel[i, :, t] = np.array([C[i, 0, t], C[i, 1, t], C[i, 2, t] * x[0, 2, t]])

    elif g_function == "g2":
        g_factor_panel = np.zeros((N, 3, Time))
        for i in range(N):
            for t in range(Time):
                if x.shape[0] == 1:
                    g_factor_panel[i, :, t] = np.array([C[i, 0, t]**2, C[i, 0, t] * C[i, 1, t], np.sign(C[i, 2, t] * x[0, 0, t])])
                else:
                    g_factor_panel[i, :, t] = np.array([C[i, 0, t]**2, C[i, 0, t] * C[i, 1, t], np.sign(C[i, 2, t] * x[0, 2, t])])

    elif g_function == "g3":
        g_factor_panel = np.zeros((N, 4, Time))
        for i in range(N):
            for t in range(Time):
                g_factor_panel[i, :, t] = np.array([
                    int(C[i, 0, t] > 0),
                    C[i, 1, t]**3,
                    C[i, 0, t] * C[i, 1, t] * int(C[i, 2, t] > 0),
                    logit(C[i, 2, t])
                ])

    elif g_function == "g4":
        g_factor_panel = np.zeros((N, 3, Time))
        for i in range(N):
            for t in range(Time):
                if x.shape[0] == 1:
                    g_factor_panel[i, :, t] = np.array([C[i, 0, t], C[i, 1, t], C[i, 2, t] * x[0, 0, t]])
                else:
                    g_factor_panel[i, :, t] = np.array([C[i, 0, t], C[i, 1, t], C[i, 2, t] * x[0, 2, t]])

    return g_factor_panel

def gen_g_panel(g_factor_panel, theta):
    g_panel = np.zeros((N, 1, Time))
    for i in range(N):
        for t in range(Time):
            g_panel[i, 0, t] = np.dot(g_factor_panel[i, :, t], theta.T)[0]
    return g_panel

def gen_error(sv, ep_sd, omega, gamma, w, C, v_sd):
    error = np.zeros((N, 1, Time))
    Beta = C[:, :3, :]
    Beta_v = np.zeros((N, 1, Time))

    for t in range(Time):
        v = np.random.normal(0, v_sd, (3, 1))
        for i in range(N):
            Beta_v[i, 0, t] = np.dot(Beta[i, :, t], v)[0]

    if sv == 1:
        logsigma2 = np.zeros(Time)
        logsigma2[0] = omega / (1 - gamma)
        for t in range(1, Time):
            logsigma2[t] = omega + gamma * logsigma2[t-1] + np.random.normal(0, w)

        for t in range(Time):
            for i in range(N):
                error[i, 0, t] = np.sqrt(np.exp(logsigma2[t])) * np.random.normal(0, ep_sd)

        return Beta_v + error
    else:
        for t in range(Time):
            # student t df=5
            error[:, 0, t] = np.random.standard_t(5, N) * np.sqrt(ep_sd**2 * (5-2)/5)

        return error + Beta_v

def panel_tune_stats(return_panel, signal_panel, true_factor_panel):
    rsquared_list = []
    annual_vol_list = []
    coeff_betas = np.zeros((N, true_factor_panel.shape[1]))
    r_bar = np.zeros(N)

    for i in range(N):
        rs = return_panel[i, 0, :]
        r_bar[i] = np.mean(rs)
        xs = true_factor_panel[i, :, :].T

        # intercept is automatically added by sm.OLS if we add constant, but like R lm(Rs ~ ., df), we need constant
        xs_with_const = sm.add_constant(xs)
        model = sm.OLS(rs, xs_with_const)
        results = model.fit()

        # Save betas (skip intercept)
        coeff_betas[i, :] = results.params[1:]

        rsquared_list.append(results.rsquared)
        annual_vol_list.append(np.std(rs, ddof=1) * np.sqrt(12))

    time_series_fitted_rsquare = np.mean(rsquared_list)
    annual_vol = np.mean(annual_vol_list)

    # R2 using metrics R2 is a traditional form
    # traditional R2 = 1 - sum(y-y_pred)^2 / sum(y-y_mean)^2
    flat_return = return_panel.flatten()
    flat_signal = signal_panel.flatten()

    # Custom R2 calculation from R package caret form="traditional"
    ss_res = np.sum((flat_return - flat_signal)**2)
    ss_tot = np.sum((flat_return - np.mean(flat_return))**2)
    true_rsquare = 1 - (ss_res / ss_tot)

    # Cross sectional rsquared
    r_bar = np.mean(return_panel[:, 0, :], axis=1)
    # Regression of r_bar on coeff_betas with intercept
    coeff_with_const = sm.add_constant(coeff_betas)
    model_cross = sm.OLS(r_bar, coeff_with_const)
    results_cross = model_cross.fit()
    cross_section_rsquare = results_cross.rsquared

    return {
        "time_series_fitted.rsquare": time_series_fitted_rsquare,
        "annual_vol": annual_vol,
        "true_rsquare": true_rsquare,
        "cross_section_rsquare": cross_section_rsquare
    }

def gen_predictor_z(C, x):
    if x.ndim == 3:
        x_reshaped = x[0, :, :]
        xt_set = np.vstack([np.ones((1, Time)), x_reshaped])
    else:
        xt_set = np.vstack([np.ones((1, Time)), x])

    z_panel = np.zeros((N, xt_set.shape[0] * P_c, Time))

    for i in range(N):
        for t in range(Time):
            z_panel[i, :, t] = np.kron(xt_set[:, t], C[i, :, t])

    return z_panel

def bind_rt_predictor(rt_panel, z_panel):
    # rt_panel (N, 1, Time)
    # z_panel (N, K, Time)
    dfs = []

    for t in range(Time):
        # time values in R start from 2 according to bind_rt_predictor
        # cbind(data.frame(rt_panel[, , 1]), time = 2, stock = paste0("stock_", c(1:N)), data.frame(z_panel[, , 1]))

        rt_t = rt_panel[:, 0, t]
        z_t = z_panel[:, :, t]

        df_t = pd.DataFrame(z_t)
        df_t.insert(0, 'stock', [f'stock_{i+1}' for i in range(N)])
        df_t.insert(0, 'time', t + 2)
        df_t.insert(0, 'rt', rt_t)

        dfs.append(df_t)

    return pd.concat(dfs, ignore_index=True)

def sim_panel_data(sim_N, char_rho_a, char_rho_b, cross_corr, cross_corr_degree,
                   A_matrix, xt_multi, g_function, theta, error_sv, error_ep_sd,
                   error_omega, error_gamma, error_w, error_v_sd, predictor_format):

    sim_list = []

    for i in range(sim_N):
        np.random.seed(27925248 + i + 1)

        C_bar = gen_C_bar(char_rho_a, char_rho_b)

        if cross_corr == 0:
            C = gen_C(C_bar)
        else:
            C_hat = gen_C_hat(C_bar, cross_corr_degree)
            C = gen_C(C_hat)

        if xt_multi == 1:
            xt = gen_xt(A_matrix)
        else:
            xt = gen_xt_univariate()

        g_factor_panel = gen_g_factor_panel(g_function, C, xt)
        g_panel = gen_g_panel(g_factor_panel, theta)

        error = gen_error(error_sv, error_ep_sd, error_omega, error_gamma, error_w, C, error_v_sd)

        rt_panel = g_panel + error

        if predictor_format == "twoway":
            # Skipping implementation of twoway due to memory requirements and focus
            pass
        else:
            z_panel = gen_predictor_z(C, xt)
            panel = bind_rt_predictor(rt_panel, z_panel)

        statistics = panel_tune_stats(rt_panel, g_panel, g_factor_panel)

        sim_list.append({
            'panel': panel,
            'statistics': statistics
        })

    return sim_list

def sim_tune_statistics(sim_panel_list):
    stats_list = []
    for sim in sim_panel_list:
        stats_list.append(sim['statistics'])

    return pd.DataFrame(stats_list)


from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import PredefinedSplit, GridSearchCV


def lm_ave_forecast_resids(lm_model, test):
    time_periods = np.unique(test['time'])
    ave_forecast_resids_vector = []
    for t in time_periods:
        test_cross_section = test[test['time'] == t]
        test_x = test_cross_section.iloc[:, 3:]
        test_x_with_const = sm.add_constant(test_x, has_constant='add')
        predictions = lm_model.predict(test_x_with_const)
        residuals = test_cross_section['rt'] - predictions
        ave_forecast_resids_vector.append(np.mean(residuals))
    return ave_forecast_resids_vector

def eln_ave_forecast_resids(eln_model, test, alpha, lambda_val):
    time_periods = np.unique(test['time'])
    ave_forecast_resids_vector = []
    for t in time_periods:
        test_cross_section = test[test['time'] == t]
        test_x = test_cross_section.iloc[:, 3:]
        predictions = eln_model.predict(test_x)
        residuals = test_cross_section['rt'] - predictions
        ave_forecast_resids_vector.append(np.mean(residuals))
    return ave_forecast_resids_vector

def rf_ave_forecast_resids(rf_model, test):
    time_periods = np.unique(test['time'])
    ave_forecast_resids_vector = []
    for t in time_periods:
        test_cross_section = test[test['time'] == t]
        test_x = test_cross_section.iloc[:, 3:]
        predictions = rf_model.predict(test_x)
        residuals = test_cross_section['rt'] - predictions
        ave_forecast_resids_vector.append(np.mean(residuals))
    return ave_forecast_resids_vector

def nnet_ave_forecast_resids(nnet_model, test):
    time_periods = np.unique(test['time'])
    ave_forecast_resids_vector = []
    for t in time_periods:
        test_cross_section = test[test['time'] == t]
        test_x = test_cross_section.iloc[:, 3:]
        predictions = nnet_model.predict(test_x)
        if predictions.ndim == 2:
            predictions = predictions.flatten()
        residuals = test_cross_section['rt'] - predictions
        ave_forecast_resids_vector.append(np.mean(residuals))
    return ave_forecast_resids_vector

def LM_variable_importance(test, lm_model):
    test_x = test.iloc[:, 3:]
    importance_df = []
    test_x_with_const = sm.add_constant(test_x, has_constant='add')
    original_predictions = lm_model.predict(test_x_with_const)
    y = test['rt']
    ss_tot = np.sum((y - np.mean(y))**2)
    original_ss_res = np.sum((y - original_predictions)**2)
    original_r2 = 1 - (original_ss_res / ss_tot)
    for i in range(test_x.shape[1]):
        test_x_zero = test_x.copy()
        test_x_zero.iloc[:, i] = 0
        test_x_zero_with_const = sm.add_constant(test_x_zero, has_constant='add')
        new_predictions = lm_model.predict(test_x_zero_with_const)
        new_ss_res = np.sum((y - new_predictions)**2)
        new_r2 = 1 - (new_ss_res / ss_tot)
        importance = original_r2 - new_r2
        importance_df.append({'variable': test_x.columns[i], 'importance': importance})
    return pd.DataFrame(importance_df)

def ELN_variable_importance(test, eln_model):
    test_x = test.iloc[:, 3:]
    importance_df = []
    original_predictions = eln_model.predict(test_x)
    y = test['rt']
    ss_tot = np.sum((y - np.mean(y))**2)
    original_ss_res = np.sum((y - original_predictions)**2)
    original_r2 = 1 - (original_ss_res / ss_tot)
    for i in range(test_x.shape[1]):
        test_x_zero = test_x.copy()
        test_x_zero.iloc[:, i] = 0
        new_predictions = eln_model.predict(test_x_zero)
        new_ss_res = np.sum((y - new_predictions)**2)
        new_r2 = 1 - (new_ss_res / ss_tot)
        importance = original_r2 - new_r2
        importance_df.append({'variable': test_x.columns[i], 'importance': importance})
    return pd.DataFrame(importance_df)

def RF_variable_importance(test, rf_model):
    test_x = test.iloc[:, 3:]
    importance_df = []
    original_predictions = rf_model.predict(test_x)
    y = test['rt']
    ss_tot = np.sum((y - np.mean(y))**2)
    original_ss_res = np.sum((y - original_predictions)**2)
    original_r2 = 1 - (original_ss_res / ss_tot)
    for i in range(test_x.shape[1]):
        test_x_zero = test_x.copy()
        test_x_zero.iloc[:, i] = 0
        new_predictions = rf_model.predict(test_x_zero)
        new_ss_res = np.sum((y - new_predictions)**2)
        new_r2 = 1 - (new_ss_res / ss_tot)
        importance = original_r2 - new_r2
        importance_df.append({'variable': test_x.columns[i], 'importance': importance})
    return pd.DataFrame(importance_df)

def NNet_variable_importance(test, nnet_model):
    test_x = test.iloc[:, 3:]
    importance_df = []
    original_predictions = nnet_model.predict(test_x)
    if original_predictions.ndim == 2:
        original_predictions = original_predictions.flatten()
    y = test['rt']
    ss_tot = np.sum((y - np.mean(y))**2)
    original_ss_res = np.sum((y - original_predictions)**2)
    original_r2 = 1 - (original_ss_res / ss_tot)
    for i in range(test_x.shape[1]):
        test_x_zero = test_x.copy()
        test_x_zero.iloc[:, i] = 0
        new_predictions = nnet_model.predict(test_x_zero)
        if new_predictions.ndim == 2:
            new_predictions = new_predictions.flatten()
        new_ss_res = np.sum((y - new_predictions)**2)
        new_r2 = 1 - (new_ss_res / ss_tot)
        importance = original_r2 - new_r2
        importance_df.append({'variable': test_x.columns[i], 'importance': importance})
    return pd.DataFrame(importance_df)

def customTimeSlices(start, initialWindow, horizon, validation_size, test_size, set_no):
    time_slices = []
    for t in range(1, set_no + 1):
        train = list(range(start, initialWindow + (t-1) * horizon + 2))
        validation = list(range(initialWindow + (t-1) * horizon + 2, (initialWindow + (t-1) * horizon) + validation_size + 2))
        test = list(range((initialWindow + (t-1) * horizon) + validation_size + 2, (initialWindow + (t-1) * horizon) + validation_size + test_size + 2))
        time_slices.append({'train': train, 'validation': validation, 'test': test})
    return time_slices

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def sse(y_true, y_pred):
    return np.sum((y_true - y_pred)**2)

def R2(y_pred, y_true):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def LM_fit(pooled_panel, timeSlices, loss_function, f=None):
    LM_stats = []
    for set_idx in range(3):
        time_slice = timeSlices[set_idx]
        train = pooled_panel[pooled_panel['time'].isin(time_slice['train'])]
        validation = pooled_panel[pooled_panel['time'].isin(time_slice['validation'])]
        test = pooled_panel[pooled_panel['time'].isin(time_slice['test'])]
        train_x = train.iloc[:, 3:]
        train_y = train['rt']
        validation_x = validation.iloc[:, 3:]
        validation_y = validation['rt']
        test_x = test.iloc[:, 3:]
        test_y = test['rt']
        train_x_with_const = sm.add_constant(train_x, has_constant='add')
        if loss_function == "mse":
            model = sm.OLS(train_y, train_x_with_const)
            lm = model.fit()
        else:
            model = sm.QuantReg(train_y, train_x_with_const)
            lm = model.fit(q=0.5)
        train_predict = lm.predict(train_x_with_const)
        validation_x_with_const = sm.add_constant(validation_x, has_constant='add')
        validation_predict = lm.predict(validation_x_with_const)
        test_x_with_const = sm.add_constant(test_x, has_constant='add')
        test_predict = lm.predict(test_x_with_const)
        loss_stats = {
            'train_MAE': mae(train_y, train_predict),
            'train_MSE': mse(train_y, train_predict),
            'train_RMSE': rmse(train_y, train_predict),
            'train_RSquare': R2(train_predict, train_y),
            'validation_MAE': mae(validation_y, validation_predict),
            'validation_MSE': mse(validation_y, validation_predict),
            'validation_RMSE': rmse(validation_y, validation_predict),
            'validation_RSquare': R2(validation_predict, validation_y),
            'test_MAE': mae(test_y, test_predict),
            'test_MSE': mse(test_y, test_predict),
            'test_RMSE': rmse(test_y, test_predict),
            'test_RSquare': R2(test_predict, test_y),
        }
        forecast_resids = lm_ave_forecast_resids(lm, test)
        variable_importance = LM_variable_importance(test, lm)
        LM_stats.append({
            'loss_stats': pd.DataFrame([loss_stats]),
            'forecast_resids': forecast_resids,
            'variable_importance': variable_importance,
            'model': lm
        })
    return LM_stats


def ELN_fit_stats(alpha_grid, nlamb, timeSlices, pooled_panel, loss_function):
    ELN_stats = []

    for set_idx in range(3):
        time_slice = timeSlices[set_idx]

        train = pooled_panel[pooled_panel['time'].isin(time_slice['train'])]
        validation = pooled_panel[pooled_panel['time'].isin(time_slice['validation'])]
        test = pooled_panel[pooled_panel['time'].isin(time_slice['test'])]

        # Combine train and validation to use with PredefinedSplit
        cv_x = pd.concat([train.iloc[:, 3:], validation.iloc[:, 3:]])
        cv_y = pd.concat([train['rt'], validation['rt']])

        # Create an array where -1 means "train" and 0 means "validation"
        test_fold = np.concatenate([
            np.full(len(train), -1),
            np.full(len(validation), 0)
        ])

        ps = PredefinedSplit(test_fold)

        train_x = train.iloc[:, 3:]
        train_y = train['rt']
        validation_x = validation.iloc[:, 3:]
        validation_y = validation['rt']
        test_x = test.iloc[:, 3:]
        test_y = test['rt']

        # Scikit-learn ElasticNetCV automatically tunes lambda using the validation set
        # (alpha in sklearn is penalty magnitude, l1_ratio is alpha in R)
        model = ElasticNetCV(l1_ratio=alpha_grid, cv=ps, random_state=42)
        model.fit(cv_x, cv_y)

        best_alpha = model.alpha_
        best_l1_ratio = model.l1_ratio_

        train_predict = model.predict(train_x)
        valid_predict = model.predict(validation_x)
        test_predict = model.predict(test_x)

        loss_stats = {
            'train_MAE': mae(train_y, train_predict),
            'train_MSE': mse(train_y, train_predict),
            'train_RMSE': rmse(train_y, train_predict),
            'train_RSquare': R2(train_predict, train_y),

            'validation_MAE': mae(validation_y, valid_predict),
            'validation_MSE': mse(validation_y, valid_predict),
            'validation_RMSE': rmse(validation_y, valid_predict),
            'validation_RSquare': R2(valid_predict, validation_y),

            'test_MAE': mae(test_y, test_predict),
            'test_MSE': mse(test_y, test_predict),
            'test_RMSE': rmse(test_y, test_predict),
            'test_RSquare': R2(test_predict, test_y),
        }

        forecast_resids = eln_ave_forecast_resids(model, test, best_l1_ratio, best_alpha)
        variable_importance = ELN_variable_importance(test, model)

        ELN_stats.append({
            'loss_stats': pd.DataFrame([loss_stats]),
            'forecasts': test_predict,
            'forecast_resids': forecast_resids,
            'model': model,
            'hyperparameters': {'alpha': best_l1_ratio, 'lambda': best_alpha},
            'variable_importance': variable_importance
        })

    return ELN_stats

def RF_fit_stats(pooled_panel, RF_grid, timeSlices, loss_function, f=None):
    RF_stats = []

    for set_idx in range(3):
        time_slice = timeSlices[set_idx]

        train = pooled_panel[pooled_panel['time'].isin(time_slice['train'])]
        validation = pooled_panel[pooled_panel['time'].isin(time_slice['validation'])]
        test = pooled_panel[pooled_panel['time'].isin(time_slice['test'])]

        # Combine train and validation to use with PredefinedSplit
        cv_x = pd.concat([train.iloc[:, 3:], validation.iloc[:, 3:]])
        cv_y = pd.concat([train['rt'], validation['rt']])

        # Create an array where -1 means "train" and 0 means "validation"
        test_fold = np.concatenate([
            np.full(len(train), -1),
            np.full(len(validation), 0)
        ])

        ps = PredefinedSplit(test_fold)

        train_x = train.iloc[:, 3:]
        train_y = train['rt']
        validation_x = validation.iloc[:, 3:]
        validation_y = validation['rt']
        test_x = test.iloc[:, 3:]
        test_y = test['rt']

        # Grid search logic for Random Forest using sklearn GridSearchCV
        param_grid = {
            'max_depth': list(set([p.get('max_depth', None) for p in RF_grid])),
            'max_features': list(set([p.get('mtry', 'auto') for p in RF_grid]))
        }

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        scoring = 'neg_mean_squared_error' if loss_function == "mse" else 'neg_mean_absolute_error'

        grid_search = GridSearchCV(rf_model, param_grid, cv=ps, scoring=scoring, n_jobs=-1)
        grid_search.fit(cv_x, cv_y)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        train_predict = best_model.predict(train_x)
        valid_predict = best_model.predict(validation_x)
        test_predict = best_model.predict(test_x)

        loss_stats = {
            'train_MAE': mae(train_y, train_predict),
            'train_MSE': mse(train_y, train_predict),
            'train_RMSE': rmse(train_y, train_predict),
            'train_RSquare': R2(train_predict, train_y),

            'validation_MAE': mae(validation_y, valid_predict),
            'validation_MSE': mse(validation_y, valid_predict),
            'validation_RMSE': rmse(validation_y, valid_predict),
            'validation_RSquare': R2(valid_predict, validation_y),

            'test_MAE': mae(test_y, test_predict),
            'test_MSE': mse(test_y, test_predict),
            'test_RMSE': rmse(test_y, test_predict),
            'test_RSquare': R2(test_predict, test_y),
        }

        forecast_resids = rf_ave_forecast_resids(best_model, test)
        variable_importance = RF_variable_importance(test, best_model)

        RF_stats.append({
            'loss_stats': pd.DataFrame([loss_stats]),
            'forecasts': test_predict,
            'forecast_resids': forecast_resids,
            'model': best_model,
            'hyperparameters': best_params,
            'variable_importance': variable_importance
        })

    return RF_stats

def NNet_fit_stats(pooled_panel, timeSlices, hidden_layers, loss_function, batch_size, patience):
    NNet_stats = []

    for set_idx in range(3):
        time_slice = timeSlices[set_idx]

        train = pooled_panel[pooled_panel['time'].isin(time_slice['train'])]
        validation = pooled_panel[pooled_panel['time'].isin(time_slice['validation'])]
        test = pooled_panel[pooled_panel['time'].isin(time_slice['test'])]

        train_x = train.iloc[:, 3:]
        train_y = train['rt']

        validation_x = validation.iloc[:, 3:]
        validation_y = validation['rt']

        test_x = test.iloc[:, 3:]
        test_y = test['rt']

        # Scikit-learn MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=hidden_layers, batch_size=batch_size,
                             n_iter_no_change=patience, random_state=42)
        model.fit(train_x, train_y)

        train_predict = model.predict(train_x)
        valid_predict = model.predict(validation_x)
        test_predict = model.predict(test_x)

        loss_stats = {
            'train_MAE': mae(train_y, train_predict),
            'train_MSE': mse(train_y, train_predict),
            'train_RMSE': rmse(train_y, train_predict),
            'train_RSquare': R2(train_predict, train_y),

            'validation_MAE': mae(validation_y, valid_predict),
            'validation_MSE': mse(validation_y, valid_predict),
            'validation_RMSE': rmse(validation_y, valid_predict),
            'validation_RSquare': R2(valid_predict, validation_y),

            'test_MAE': mae(test_y, test_predict),
            'test_MSE': mse(test_y, test_predict),
            'test_RMSE': rmse(test_y, test_predict),
            'test_RSquare': R2(test_predict, test_y),
        }

        forecast_resids = nnet_ave_forecast_resids(model, test)
        variable_importance = NNet_variable_importance(test, model)

        NNet_stats.append({
            'loss_stats': pd.DataFrame([loss_stats]),
            'forecasts': test_predict,
            'forecast_resids': forecast_resids,
            'model': model,
            'variable_importance': variable_importance
        })

    return NNet_stats

def fit_all_models(dataset_list, batch_process_range, LM=1, ELN=1, RF=1, NNet=1):
    simulation_results_list = [None] * len(batch_process_range)

    for idx, batch in enumerate(batch_process_range):
        # We assume dataset_list is 0-indexed in Python
        current_dataset = dataset_list[batch]
        pooled_panel = current_dataset['panel']

        simulation_results_list[idx] = {
            'Dataset_stats': current_dataset['statistics'],
            'returns': pooled_panel['rt'],
            'LM_MSE': None, 'LM_MAE': None,
            'ELN_MSE': None, 'ELN_MAE': None,
            'RF_MSE': None, 'RF_MAE': None,
            'NN1_MSE': None, 'NN1_MAE': None,
            'NN2_MSE': None, 'NN2_MAE': None,
            'NN3_MSE': None, 'NN3_MAE': None,
            'NN4_MSE': None, 'NN4_MAE': None,
            'NN5_MSE': None, 'NN5_MAE': None
        }

        timeSlices = customTimeSlices(start=2, initialWindow=84, horizon=12, validation_size=60, test_size=12, set_no=3)

        f = None

        if LM == 1:
            simulation_results_list[idx]['LM_MSE'] = LM_fit(pooled_panel, timeSlices, loss_function="mse")
            simulation_results_list[idx]['LM_MAE'] = LM_fit(pooled_panel, timeSlices, loss_function="mae")

        if ELN == 1:
            alpha_grid = np.arange(0, 1.01, 0.01)
            simulation_results_list[idx]['ELN_MSE'] = ELN_fit_stats(alpha_grid, 100, timeSlices, pooled_panel, loss_function="mse")
            simulation_results_list[idx]['ELN_MAE'] = ELN_fit_stats(alpha_grid, 100, timeSlices, pooled_panel, loss_function="mae")

        if RF == 1:
            RF_grid = []
            max_features = int((pooled_panel.shape[1] - 3) / 4)
            for mtry in range(10, max_features + 1, 10):
                RF_grid.append({'mtry': mtry, 'max_depth': 50})

            simulation_results_list[idx]['RF_MSE'] = RF_fit_stats(pooled_panel, RF_grid, timeSlices, loss_function="mse")
            simulation_results_list[idx]['RF_MAE'] = RF_fit_stats(pooled_panel, RF_grid, timeSlices, loss_function="mae")

        if NNet == 1:
            batch_size = 1000
            patience = 5

            # Neural Networks architectures
            hidden_layers_1 = (32,)
            hidden_layers_2 = (32, 16)
            hidden_layers_3 = (32, 16, 8)
            hidden_layers_4 = (32, 16, 8, 4)
            hidden_layers_5 = (32, 16, 8, 4, 2)

            simulation_results_list[idx]['NN1_MSE'] = NNet_fit_stats(pooled_panel, timeSlices, hidden_layers_1, "mse", batch_size, patience)
            simulation_results_list[idx]['NN1_MAE'] = NNet_fit_stats(pooled_panel, timeSlices, hidden_layers_1, "mae", batch_size, patience)

            simulation_results_list[idx]['NN2_MSE'] = NNet_fit_stats(pooled_panel, timeSlices, hidden_layers_2, "mse", batch_size, patience)
            simulation_results_list[idx]['NN2_MAE'] = NNet_fit_stats(pooled_panel, timeSlices, hidden_layers_2, "mae", batch_size, patience)

            simulation_results_list[idx]['NN3_MSE'] = NNet_fit_stats(pooled_panel, timeSlices, hidden_layers_3, "mse", batch_size, patience)
            simulation_results_list[idx]['NN3_MAE'] = NNet_fit_stats(pooled_panel, timeSlices, hidden_layers_3, "mae", batch_size, patience)

            simulation_results_list[idx]['NN4_MSE'] = NNet_fit_stats(pooled_panel, timeSlices, hidden_layers_4, "mse", batch_size, patience)
            simulation_results_list[idx]['NN4_MAE'] = NNet_fit_stats(pooled_panel, timeSlices, hidden_layers_4, "mae", batch_size, patience)

            simulation_results_list[idx]['NN5_MSE'] = NNet_fit_stats(pooled_panel, timeSlices, hidden_layers_5, "mse", batch_size, patience)
            simulation_results_list[idx]['NN5_MAE'] = NNet_fit_stats(pooled_panel, timeSlices, hidden_layers_5, "mae", batch_size, patience)

    return simulation_results_list
