# importar pacotes
import pandas as pd
import matplotlib.pyplot as plt

# import dataframe
df = pd.read_csv('2018-2019_Daily_Attendance_20240429.csv', sep=',')

print(df.head())

print(df.info())

print(df.duplicated().sum())

print(df.describe())

print(df.isnull().sum())

def busca_mes(dado):
    dado = str(dado)
    dado = dado[4:-2]
    dado = int(dado)
    return dado

df['month'] = df['Date'].apply(busca_mes)

df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

df['Date_month'] = df['Date'].dt.month

df_group = df.groupby(['School DBN'])['Enrolled'].agg(
    media='mean',
    minima='min',
    maxima='max',
    desvio_padrao='std',
    primeira_contagem='first',
    ultima_contagema='last'
).reset_index()

print(f'Total escolas: {len(df_group)}')

df_group['diff_comeco_fim'] = df_group['ultima_contagema'] - df_group['primeira_contagem']

df_group = df_group.sort_values(['diff_comeco_fim','desvio_padrao'], ascending=[True, False]).reset_index(drop=True)

print(df_group.drop_duplicates(['School DBN']).head())

df_group_queda = df_group.loc[df_group['diff_comeco_fim'] < 0].reset_index(drop=True)

df_group_queda_top = df_group_queda.head(10)

print(df_group_queda)

lista_escolas_queda = list(df_group_queda_top['School DBN'])

plt.barh(df_group_queda_top['School DBN'], df_group_queda_top['diff_comeco_fim'])
plt.show()

df_group_piores = df.groupby(['School DBN', 'month'])['Enrolled'].agg(
    media='mean',
    minima='min',
    maxima='max',
    desvio_padrao='std',
    primeira_contagem='first',
    ultima_contagema='last'
).reset_index()

df_group_piores['diff_comeco_fim'] = df_group_piores['ultima_contagema'] - df_group_piores['primeira_contagem']

def mod(dado):
    if dado < 0:
        dado = dado *-1
    
    return dado

df_group_piores['diff_comeco_fim'] = df_group_piores['diff_comeco_fim'].apply(mod)

df_group_piores = df_group_piores.sort_values(['diff_comeco_fim','desvio_padrao'], ascending=[True, False]).reset_index(drop=True)

def achar_quartil(dado):
    if dado <= 1:
        return 1
    elif dado <= 2:
        return 2
    elif dado <= 4:
        return 3
    return 4
df_group_piores['quartil'] = df_group_piores['diff_comeco_fim'].apply(achar_quartil)

print(df_group_piores.corr())
print(df_group_piores)

print(df_group_piores.describe())

df_group_piores.describe().plot()
plt.show()

print(df_group_piores.loc[df_group_piores['quartil'] == 4])

top_mes = df.groupby(['month'])['Enrolled'].agg(media = 'mean').reset_index().sort_values('media', ascending=False)
print(top_mes)