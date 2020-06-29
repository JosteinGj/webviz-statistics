
import pandas as pd


df_ts_resp = pd.read_parquet(
    "/home/jg/Documents/work/equinor/Equinor_R_models/equinor-R-models/\
        data files/100 realizations, 4 iterations (ensembles)/\
            response_timeseries_100realizations.parquet")
df_para = pd.read_parquet(
    "/home/jg/Documents/work/equinor/Equinor_R_models/equinor-R-models/\
        data files/100 realizations, 4 iterations (ensembles)/\
            parameters100realizations.parquet")
df_ip_resp = pd.read_parquet(
    "/home/jg/Documents/work/equinor/Equinor_R_models/equinor-R-models/\
        data files/100 realizations, 4 iterations (ensembles)/\
            response_grid_volumes_100realizations.parquet")

df_ts_resp.columns = [col.replace(":", "_") for col in df_ts_resp.columns]
df_ip_resp.columns = [col.replace(":", "_") for col in df_ip_resp.columns]
df_para.columns = [col.replace(":", "_") for col in df_para.columns]


df_ts_resp.to_csv("/home/jg/Documents/work/equinor/Equinor_R_models/\
    equinor-R-models/data files/100 realizations, 4 iterations (ensembles)/\
        response_timeseries_100realizations.csv")
df_ip_resp.to_csv("/home/jg/Documents/work/equinor/Equinor_R_models/\
    equinor-R-models/data files/100 realizations, 4 iterations (ensembles)/\
        response_grid_volumes_100realizations.csv")
df_para.to_csv("/home/jg/Documents/work/equinor/Equinor_R_models/\
    equinor-R-models/data files/100 realizations, 4 iterations (ensembles)/\
        parameters100realizations.csv")