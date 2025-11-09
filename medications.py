import pandas as pd

MEDI_CSV_PATH = "MEDI_01212013.csv"

df_medi = pd.read_csv(MEDI_CSV_PATH)

df_medi.columns = df_medi.columns.str.strip()

df_medi['IndicationName'] = df_medi['IndicationName'].str.lower()
df_medi['RxNormName'] = df_medi['RxNormName'].str.title()


def get_medications(diagnosis: str):
    """
    Returns a list of medications for a given diagnosis string.
    """
    diagnosis = diagnosis.lower().strip()
    meds = df_medi[df_medi['IndicationName'].str.contains(diagnosis, na=False)]['RxNormName'].unique()
    return list(meds)


'''MEDI_CSV_PATH = "MEDI_01212013.csv"

# Load CSV
df_medi = pd.read_csv(MEDI_CSV_PATH)

# Print columns and first 5 rows for debugging
print("Columns:", df_medi.columns.tolist())
print(df_medi.head())'''

