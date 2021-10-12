import pandas as pd

df = pd.read_csv("./biomimicry_functions_enumerated.csv")

def enumerate_functions(*argv):
  for arg in argv:
    enum = df.loc[df['function'] == f'{arg}','function_enumerated'].values[0]
    y = f'{arg}:', enum
    a = str(y)
    b = ''.join(a)
    b= b.replace(',', '')
    # b = b.strip('\"')
    return b

