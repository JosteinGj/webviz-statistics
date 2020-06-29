
import pandas as pd
import numpy as np
from statsmodels.regression import linear_model
from sklearn.preprocessing import PolynomialFeatures

# Leser inn tabellene
df_parameters = pd.read_parquet( "/home/jg/Documents/work/equinor/Equinor_R_models/equinor-R-models/data files/100 realizations, 4 iterations (ensembles)/parameters100realizations.parquet")
df_response = pd.read_parquet("/home/jg/Documents/work/equinor/Equinor_R_models/equinor-R-models/data files/100 realizations, 4 iterations (ensembles)/response_grid_volumes_100realizations.parquet")

# Velger ut hvilken response man vil se på (f.eks. bulk oil) ved eliminasjonsmetoden hehe
filtered_response = df_response.drop(columns=['PORE_OIL', 'HCPV_OIL', 'STOIIP_OIL'])

# Merger dataframesene på ensemble og real
merged_df = pd.merge(filtered_response, df_parameters, on=["ENSEMBLE", "REAL"])

# Filterer ut rader, vil se på f.eks. iter-1, upper reek og region 1
row_filtered_df = merged_df[(merged_df.ENSEMBLE == 'iter-1') & (merged_df.ZONE == 'UpperReek') & (merged_df.REGION == 1)]

# Filtrerer ut kolonner og får den ferdige dataframen som inneholder response (kolonne 1) og paramterere (resten av kolonnene)
filtered_df = row_filtered_df.drop(columns=['ENSEMBLE', 'REAL', 'ZONE', 'REGION', 'COHIBA_MODEL_MODE', 
'RMSGLOBPARAMS:COHIBA_MODEL_MODE', 'LOG10_MULTFLT:MULTFLT_F1', 'LOG10_MULTFLT:MULTFLT_F2', 
'LOG10_MULTFLT:MULTFLT_F3', 'LOG10_MULTFLT:MULTFLT_F4', 'LOG10_MULTFLT:MULTFLT_F5', 'LOG10_MULTZ:MULTZ_MIDREEK'])

# Fjerner duplikate kolonner fra dataframen
def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break

    return dups

duplicates = duplicate_columns(filtered_df)
filtered_df.drop(duplicates, inplace=True, axis=1)

# Printer ut den ferdige dataframen med 1 respons (bulk oil) og de 9 gjenværende parameterne
print(filtered_df)

# Lager variabler som skal brukes i regresjonen
X = filtered_df.drop('BULK_OIL', axis=1)
y = filtered_df['BULK_OIL']
print(X.columns)
# Kjører regresjon med OLS
model = linear_model.OLS(y, X).fit()
print(model.summary())