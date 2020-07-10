import pandas as pd
import numpy as np
import numpy.linalg as la
from sklearn.preprocessing import PolynomialFeatures
import plotly as ply
import plotly.graph_objects as go
import webviz_core_components as wcc
from itertools import combinations
from pathlib import Path
from scipy.stats import f_oneway
from numba import jit
import time
import statsmodels.formula.api as smf

def load_data(parameter_path: Path = None,
              timeseries_path: Path = None,
              inplace_path: Path = None
              ):
    para_df = pd.read_parquet(parameter_path)
    inpl_df = pd.read_parquet(inplace_path)
    ts_df = pd.read_parquet(timeseries_path)

    ts_df.columns = [col.replace(":", "_") for col in ts_df.columns]
    inpl_df.columns = [col.replace(":", "_") for col in inpl_df.columns]
    para_df.columns = [col.replace(":", "_") for col in para_df.columns]

    return (para_df, inpl_df, ts_df)


def filter_inplace_volumes(
        param_df: pd.DataFrame,
        inplace_df: pd.DataFrame,
        inplace_filters: dict,
        global_filters: dict,
        params_exclude: list=[],
        params_include: list=[]):
    inplace_filtered = inplace_df.filter(
            items=[
                "ENSEMBLE",
                "REAL",
                "ZONE",
                "REGION",
                inplace_filters["RESPONSE"]
                ])
    inplace_filtered = inplace_filtered.loc[
        (inplace_filtered["ZONE"] == inplace_filters["ZONE"]) &
        (inplace_filtered["REGION"] == inplace_filters["REGION"]) &
        (inplace_filtered["ENSEMBLE"] == inplace_filters["ENSEMBLE"])]
    inplace_filtered = inplace_filtered.drop(columns=["ZONE", "REGION"])
    merged = pd.merge(param_df, inplace_filtered,
                    on=["ENSEMBLE", "REAL"]).drop(
                        columns=["ENSEMBLE", "REAL",
                                 ]+params_exclude)
    if params_include:
        return merged.filter(items=params_include)
    else:
        return merged 


def gen_column_names(df: pd.DataFrame, response: str=None):
    if response:
        combine = ["*".join(combination) for combination in combinations(df.drop(columns=response).columns, 2)]
        originals = list(df.drop(columns=response).columns)
        return originals + combine + [response]
    else:
        combine = ["*".join(combination) for combination in combinations(df,2)]
        originals = list(df.columns)
    print(len(combine+originals))
    return originals + combine 


def filter_timeseries(param_df: pd.DataFrame,
                      timeseries_df: pd.DataFrame,
                      timeseries_filters: dict,
                      global_filters: dict,
                      params_exclude: list=[],
                      params_include: list=[]):
    
    for i in params_include:
        if i in params_exclude:
            raise ValueError(" The same parameter cant be excluded and included in the data")

    timeseries_filtered = timeseries_df.filter(items=[
        "ENSEMBLE",
        "REAL",
        "DATE",
        timeseries_filters["RESPONSE"]])
    timeseries_filtered = timeseries_filtered.loc[
        (timeseries_filtered["DATE"] == timeseries_filters["DATE"]) &
        (timeseries_filtered["ENSEMBLE"] == timeseries_filters["ENSEMBLE"])]
    merged = pd.merge(param_df, timeseries_filtered,
                        on=["ENSEMBLE", "REAL"]).drop(
                            columns=[
                                    "ENSEMBLE",
                                    "REAL",
                                    "DATE",
                                    ] + params_exclude)

    if params_include:
        return merged.filter(items=[timeseries_filters["RESPONSE"]] + params_include  )
    else:
        return merged

def gen_interaction_df(
    df: pd.DataFrame,
    response: str,
    degree: int=2,
    inter_only: bool=True,
    bias: bool=False):

    x_interaction = PolynomialFeatures(
        degree=2,
        interaction_only=inter_only,
        include_bias=False).fit_transform(df.drop(columns=response))
    print(x_interaction.shape)
    interaction_df = pd.DataFrame(
        x_interaction,
        columns=gen_column_names(df=df.drop(columns=response)))
    return interaction_df.join(df[response])


def opt_forward_select(data: pd.DataFrame, vars: np.ndarray, response: str, maxvars: int=5):
    remaining = set(vars)
    remaining.remove(response)
    selected = []

    y = data[response].to_numpy(dtype="float32")
    print("y shape:", y.shape)
    n = len(y)
    y_mean = np.mean(y)
    SST = np.sum((y-y_mean) ** 2)
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score and len(selected) < maxvars:
        scores_with_candidates = []
        p = len(selected)+1
        for candidate in remaining:
            X = data.filter(items=selected+[candidate]).to_numpy(dtype="float32")
            #print("X shape: ", X.T.shape)
            #print(X.T)
            beta = la.inv(X.T @ X) @ X.T @ y 
            f_vec = beta @ X.T

            SS_RES = np.sum((y-f_vec) ** 2)
            R_2_adj = 1-(1 - SS_RES / SST)*((n-1)/n-p-1)
            scores_with_candidates.append((R_2_adj, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    print(scores_with_candidates)
    return model

def fit_regression(
    df: pd.DataFrame,
    response: str,
    max_vars: int=9,
    interaction: bool=False
    ):
    """ ### Function for fiting a linear regression model with forward stepwise modelselection

        **Note** highly correlated parameters should be removed, especially copied variables and transformed copies.  
        =========================================================

        Parameters:
        df: pandas DataFrame, containing all covariates and the desired response

        response: string indicateing column of response in df

        max_vars: int, number of covariates to include in model, if the model includes interaction it can 

        testsize: float proportion of the data you want to use for testing

        interaction: boolean, for activating interaction, if flaged will genereate a a model with variables of polynomial degree 2 with cross terms
    """

    model = opt_forward_select(df,df.columns, response, maxvars=max_vars)
    #model = opt_forward_select(df, df.columns, response, maxvars=max_vars) 

    #print(model.summary())==ip_inter.iat[0]

    #print(model.pvalues.values)==ip_inter.iat[0]
    return model

param_path = "equinor-R-models/data files/100 realizations, 4 iterations (ensembles)/parameters100realizations.parquet"
ts_path = "equinor-R-models/data files/100 realizations, 4 iterations (ensembles)/response_timeseries_100realizations.parquet"
inplace_volumes_path = "equinor-R-models/data files/100 realizations, 4 iterations (ensembles)/response_grid_volumes_100realizations.parquet"
ip_filters = {"ENSEMBLE": "iter-0",
              "ZONE": "UpperReek",
              "REGION": 1,
              "RESPONSE": "BULK_OIL"}
gl_filters = {"ENSEMBLE": "iter-0"}
ts_filters = {"ENSEMBLE": "iter-0",
              "DATE": pd.to_datetime("2001-01-01"),
              "RESPONSE": "FPR"}
parameter_filters = [
    'RMSGLOBPARAMS_FWL',
    'MULTFLT_MULTFLT_F1',
    'MULTFLT_MULTFLT_F2',
    'MULTFLT_MULTFLT_F3',
    'MULTFLT_MULTFLT_F4',
    'MULTFLT_MULTFLT_F5',
    'MULTZ_MULTZ_MIDREEK',
    'INTERPOLATE_RELPERM_INTERPOLATE_GO',
    'INTERPOLATE_RELPERM_INTERPOLATE_WO',
    'LOG10_MULTFLT_MULTFLT_F1',
    'LOG10_MULTFLT_MULTFLT_F2',
    'LOG10_MULTFLT_MULTFLT_F3',
    'LOG10_MULTFLT_MULTFLT_F4',
    'LOG10_MULTFLT_MULTFLT_F5',
    'LOG10_MULTZ_MULTZ_MIDREEK',
    "RMSGLOBPARAMS_COHIBA_MODEL_MODE",
    "COHIBA_MODEL_MODE"]
para_df, ip_df, ts_df = load_data(parameter_path=param_path,
                                  timeseries_path=ts_path,
                                  inplace_path=inplace_volumes_path)
ip_filtered = filter_inplace_volumes(param_df=para_df,
                                     inplace_df=ip_df,
                                     inplace_filters=ip_filters,
                                     global_filters=gl_filters,
                                     params_exclude=parameter_filters
                                     )
ts_filtered = filter_timeseries(param_df=para_df,

                                timeseries_df=ts_df,
                                timeseries_filters=ts_filters,
                                global_filters=gl_filters,
                                params_exclude=parameter_filters,
                                #params_include=parameters_include
                                )

ip_inter = gen_interaction_df(ip_filtered,ip_filters["RESPONSE"]) 
print(ip_inter.head())
ts = time.perf_counter()
ipfit = fit_regression(ip_inter, ip_filters["RESPONSE"],max_vars=9)

print(ipfit.summary())
te = time.perf_counter()
print(te-ts)