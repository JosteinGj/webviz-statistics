import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from sklearn.preprocessing import PolynomialFeatures
import plotly as ply
import plotly.graph_objects as go
import webviz_core_components as wcc
from pathlib import Path

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
    merged =  pd.merge(param_df, inplace_filtered,
                    on=["ENSEMBLE", "REAL"]).drop(
                        columns=["ENSEMBLE", "REAL",
                                 ]+params_exclude)
    if params_include:
        return merged.filter(items=params_include)
    else:
        return merged 


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


def forward_selected_interaction(data, response, maxvars=3):
    # TODO find way to remove non-significant variables form entering model.
    # TODO implement forced entry to model when interaction is present
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score and len(selected) < maxvars:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            candidate_split = best_candidate.split(sep=":")
            if len(candidate_split) == 2:  # hvis candidat er en interaksjon, og de underliggende variablene ikke er valgt enda, legg dem til
                if candidate_split[0] not in selected and candidate_split[0] in remaining: 
                    remaining.remove(candidate_split[0])
                    selected.append(candidate_split[0])
                    maxvars += 1
                if candidate_split[1] not in selected and candidate_split[1] in remaining:
                    remaining.remove(candidate_split[1])
                    selected.append(candidate_split[1])
                    maxvars += 1
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model


def forward_selected(data, response, maxvars=3):
    # TODO find way to remove non-significant variables form entering model. 
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []

    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score and len(selected) < maxvars:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model


def transform_df(target_df: pd.DataFrame, target_column: str, transform):
    target_df[target_column] = transform(target_df[target_column])
    return target_df


def gen_column_names(df, interaction_only):
    output = list(df.columns)
    if interaction_only:
        for colname1 in df.columns:
            for colname2 in df.columns:
                if (
                    (colname1 != colname2) and
                    (f"{colname1}:{colname2}" not in output) or
                    (f"{colname2}:{colname1}" not in output)
                        ):
                        output.append(f"{colname1}:{colname2}")
    else:
        for colname1 in df.columns:
            for colname2 in df.columns:
                if (f"{colname1}:{colname2}" not in output) and (f"{colname2}:{colname1}" not in output):
                    output.append(f"{colname1}:{colname2}")
    return output


def gen_interaction_df(df: pd.DataFrame, response: str, degree: int=2, inter_only: bool=False, bias: bool=False):

    x_interaction = PolynomialFeatures(degree=2, interaction_only=inter_only, include_bias=False).fit_transform(df.drop(columns=response))
    interaction_df = pd.DataFrame(x_interaction, columns=gen_column_names(df.drop(columns=response), inter_only))
    return interaction_df.join(df[response])


def make_p_values_plot(model):
    """ Sorting the dictionary in ascending order and making lists for parameters and p-values """
    p_sorted = model.pvalues.sort_values()
    parameters = p_sorted.index
    values = p_sorted.values
    print(parameters)
    print(values)
    print(p_sorted)
    colors = ["#FF1243" if val<0.05 else "slategray" for val in values]
    x_vals = np.arange(-1,len(values)+1)
    """ testing dict things """
    dict_fig = dict(
        {"data": [
                {
                    "type": "bar",
                    "x": parameters,
                    "y": values,
                    "marker_color": colors
                }], 
        })
    return go.Figure(dict_fig)


def col_to_catagorical(df: pd.DataFrame,column: str, levels: int=2):
    
    col= df[column]
    colmean= col.mean()
    colstd = col.std()
    def triple_category(val, low=colmean-colstd, high=colmean+colstd):
        if val < low:
            return "low"
        elif low < val < high:
            return "med"
        else:
            return "high"
    
    if levels == 2:
        col = col.apply(lambda x: "high" if x>colmean else "low")
        print(col)
        df[column] = col.astype("category")
    elif levels == 3:
        col = col.apply(triple_category)
        df[column] = col.astype("category")
    return df


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
    
    if interaction:
        df = gen_interaction_df(df, response)
        model = forward_selected_interaction(df, response, maxvars=max_vars)
    else:
        model = forward_selected(df, response, maxvars=max_vars) 

    print(model.summary())
    #print(sm.stats.anova_lm(model, typ=2))
    #print(model.pvalues.values)
    return model


def col_percentile(df: pd.DataFrame, column: str, percentile: int):
    col = df[column].sort_values()
    bottom_index = int((100 - percentile) * len(col) / 100)
    top_index = int(percentile * len(col) / 100)


    col[top_index:] = "high"
    col[:bottom_index] = "low"
    col[bottom_index:top_index] = "middle"
    df[column] = col.astype("category")
    return df


def df_to_discrete(df: pd.DataFrame, percentile: int=90):
    for col in df.columns:
        df = col_percentile(df, col, 90)
    return df

def anova(df: pd.DataFrame, grouping_parameters: list):
    groups = df.groupby("FWL")
    print("grouped", groups)



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
parameters_include = ["FWL"]

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
parameters = para_df.columns



#ip_model = fit_regression(ip_filtered, ip_filters["RESPONSE"], max_vars=9, interaction=True)
# interaction_dict_ip = fit_regression(ip_filtered, ip_filters["RESPONSE"], max_vars=9, testsize=0.1, interaction=True)


# Implement higher degree interaction, no polynomials
# Implement Discrete variables for interaction
# Implement
discrete_df = df_to_discrete(ip_filtered, 90)
print(anova(discrete_df,["FWL"]))

#data=ip_filtered
#formula = "BULK_OIL ~ FWL*MULTFLT_F5"
#model = smf.ols(formula, data).fit()
#print(model.summary())
#print(col_percentile(ip_filtered, "FWL", 95))
#ip_fig = make_p_values_plot(ip_model)
#ip_fig.show()