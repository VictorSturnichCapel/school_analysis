import pandas as pd
import numpy as np

# Configurar seed para reprodutibilidade
np.random.seed(42)

# Criar dados fictícios
n_alunos = 200
data = {
    'id_aluno': np.arange(1, n_alunos+1),
    'idade': np.random.randint(14, 19, n_alunos),
    'genero': np.random.choice(['M', 'F'], n_alunos),
    'nota_media': np.round(np.random.normal(loc=6.5, scale=1.5, size=n_alunos), 1),
    'presenca_porcentagem': np.round(np.random.normal(loc=75, scale=15, size=n_alunos), 1),
    'advertencias': np.random.poisson(lam=1.2, size=n_alunos),
    'atividades_extras': np.random.choice([0, 1], n_alunos, p=[0.6, 0.4]),
    'distancia_escola_km': np.round(np.random.exponential(scale=5, size=n_alunos), 1),
    'renda_familiar': np.random.choice(['Baixa', 'Média', 'Alta'], n_alunos, p=[0.5, 0.3, 0.2]),
    'envolvimento_pais': np.random.randint(1, 6, n_alunos),  # Escala de 1 a 5
    'churn': np.random.choice([0, 1], n_alunos, p=[0.85, 0.15])  # 15% de evasão
}

df = pd.DataFrame(data)

df.sort_values(by=['churn', 'nota_media'], inplace=True)

# Ajustar limites (notas entre 0 e 10, presença entre 0% e 100%)
df['nota_media'] = df['nota_media'].clip(0, 10)
df['presenca_porcentagem'] = df['presenca_porcentagem'].clip(0, 100)

# Exibir primeiras linhas
print(df.head(10))

print(df.info())

print(df.duplicated().sum())

print(df.describe())

print(df.isnull().sum())

# Agrupar por churn e calcular estatísticas
desc_stats = df.groupby(['churn'])['nota_media'].agg(
    media='mean',
    minima='min',
    maxima='max',
    desvio_padrao='std',
    primeira_contagem='first',
    ultima_contagema='last'
).reset_index()

print("Estatísticas Descritivas por Grupo (Churn):")
print(desc_stats)

import matplotlib.pyplot as plt
import seaborn as sns

# Gráfico 1: Presença vs. Churn
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='churn', y='presenca_porcentagem', data=df)
plt.title("Presença vs. Churn")

# Gráfico 2: Nota Média vs. Churn
plt.subplot(1, 2, 2)
sns.boxplot(x='churn', y='nota_media', data=df)
plt.title("Nota Média vs. Churn")

plt.tight_layout()
plt.show()

plt.plot(df['churn'], df['nota_media'])
plt.show()

# Converter variáveis categóricas em numéricas
df_numeric = pd.get_dummies(df, columns=['genero', 'renda_familiar'], drop_first=True)

# Calcular correlações
corr = df_numeric.corr()

# Plotar heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlação")
plt.show()

####################################################################

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Codificar variáveis categóricas
df['genero'] = LabelEncoder().fit_transform(df['genero'])
df['renda_familiar'] = LabelEncoder().fit_transform(df['renda_familiar'])

# Separar features (X) e target (y)
X = df.drop(['id_aluno', 'churn'], axis=1)  # Remover colunas não relevantes
y = df['churn']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Treinar o modelo
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train, y_train)

# Prever no teste
y_pred = model_lr.predict(X_test)
y_proba = model_lr.predict_proba(X_test)[:, 1]  # Probabilidades de churn

# Avaliação
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))
print("\nAUC-ROC:", roc_auc_score(y_test, y_proba))

# Criar DataFrame com coeficientes
coeficientes = pd.DataFrame({
    'Variável': X.columns,
    'Coeficiente': model_lr.coef_[0]
}).sort_values(by='Coeficiente', ascending=False)

print("\nCoeficientes do Modelo (Impacto no Churn):")
print(coeficientes)

# Criar DataFrame com previsões
df_pred = X_test.copy()
df_pred['prob_churn'] = y_proba
df_pred['churn_previsto'] = y_pred
df_pred['alvo'] = y_test.values  # Adicionar o valor real para comparação

# Filtrar alunos com alta probabilidade de churn
alunos_risco = df_pred[df_pred['prob_churn'] > 0.5].sort_values(by='prob_churn', ascending=False)

print("\nAlunos com Risco de Churn:")
print(alunos_risco[['prob_churn', 'churn_previsto', 'alvo']].head(10))

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(coeficientes['Variável'], coeficientes['Coeficiente'], color='#FF6B6B')
plt.xlabel('Coeficiente (Impacto no Churn)')
plt.title('Impacto das Variáveis na Evasão Escolar')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()