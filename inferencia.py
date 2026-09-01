# ==============================================================================
# PROJETO INTEGRADOR IV - UNIVESP
# MOTOR DE INFERÊNCIA E INGESTÃO DE DADOS (VERSÃO FINAL)
# ==============================================================================
import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
import warnings
import os
from supabase import create_client

warnings.filterwarnings('ignore')

print("Iniciando o Pipeline de Inferência...")

# ==============================================================================
# 1. CONEXÃO COM SUPABASE
# ==============================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL não foi encontrada nos GitHub Secrets.")

if not SUPABASE_URL.startswith("https://"):
    raise RuntimeError("SUPABASE_URL deve começar com https://.")

if "supabase.co" not in SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL não parece ser uma URL válida de projeto Supabase.")

if "/rest/" in SUPABASE_URL or "/functions/" in SUPABASE_URL:
    raise RuntimeError(
        "SUPABASE_URL deve conter somente a URL-base do projeto, "
        "sem /rest/v1, /functions ou nome de tabela."
    )

if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_KEY não foi encontrada nos GitHub Secrets.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("✅ Conectado ao Supabase!")

# ==============================================================================
# 2. LER DADOS DO SUPABASE (TABELA: casos_dengue_sjc)
# ==============================================================================
print("Lendo dados do Supabase...")
data = supabase.table("casos_dengue_sjc").select("*").order(
    "data_semana",
    desc=False
).execute()
df = pd.DataFrame(data.data)

print(f"Total de registros: {len(df)}")

# ==============================================================================
# 3. PREPARAR DADOS (SE NECESSÁRIO)
# ==============================================================================
# Converter data para datetime
df['data_semana'] = pd.to_datetime(df['data_semana'])
df = df.sort_values('data_semana').reset_index(drop=True)

# ==============================================================================
# 3.1 BUSCAR ADL REAL (TABELA: adl_indice_sjc) E MESCLAR COM OS DADOS SEMANAIS
# ==============================================================================
print("Lendo índice ADL (LIRAa) do Supabase...")
adl_data = supabase.table("adl_indice_sjc").select("*").order("data_referencia").execute()
df_adl = pd.DataFrame(adl_data.data)

if df_adl.empty:
    raise RuntimeError("A tabela adl_indice_sjc está vazia. Cadastre ao menos um ciclo do LIRAa antes de rodar a inferência.")

df_adl['data_referencia'] = pd.to_datetime(df_adl['data_referencia'])
df_adl = df_adl.sort_values('data_referencia').reset_index(drop=True)

df = pd.merge_asof(
    df.sort_values('data_semana'),
    df_adl[['data_referencia', 'indice_adl']].rename(columns={'indice_adl': 'indice_breteu_adl'}),
    left_on='data_semana',
    right_on='data_referencia',
    direction='backward'
)

# Preenche somente semanas anteriores ao primeiro ciclo de ADL, se houver.
df['indice_breteu_adl'] = df['indice_breteu_adl'].bfill()

if df['indice_breteu_adl'].isna().any():
    raise RuntimeError(
        "Existem semanas sem ADL após o cruzamento. "
        "Verifique as datas da tabela adl_indice_sjc."
    )

# Coluna auxiliar; não é usada pelo XGBoost.
df = df.drop(columns=['data_referencia'])

print(
    f"ADL mesclado com sucesso. Último ciclo usado: "
    f"{df_adl['data_referencia'].max().strftime('%d/%m/%Y')}"
    
# ==============================================================================
# 4. ENGENHARIA DE RECURSOS (LAG FEATURES)
# ==============================================================================
for lag in [1, 2, 3]:
    df[f"casos_lag_{lag}"] = df["casos_confirmados"].shift(lag)
    df[f"temp_min_lag_{lag}"] = df["temperatura_minima"].shift(lag)
    df[f"umidade_lag_{lag}"] = df["umidade_maxima"].shift(lag)

# Remover NaNs (primeiras 3 semanas)
df = df.dropna().reset_index(drop=True)

# ==============================================================================
# 5. PREPARAR FEATURES PARA TREINAMENTO
# ==============================================================================
features = [
    'temp_min_lag_1', 'umidade_lag_1',
    'temp_min_lag_2', 'umidade_lag_2',
    'temp_min_lag_3', 'umidade_lag_3',
    'densidade_populacional', 'taxa_coleta_residuos', 'indice_breteu_adl',
    'casos_lag_1', 'casos_lag_2', 'casos_lag_3'
]

# Recorte D-60 (esconder últimos 60 dias)
data_limite = df['data_semana'].max() - pd.Timedelta(days=60)
df_treino = df[df['data_semana'] <= data_limite]

X = df_treino[features]
y = df_treino['casos_confirmados']

print(f"IA treinada com dados até: {df_treino['data_semana'].max().strftime('%d/%m/%Y')}")

# ==============================================================================
# 6. TREINAR MODELO XGBOOST
# ==============================================================================
modelo_producao = xgb.XGBRegressor(
    learning_rate=0.2,
    max_depth=3,
    n_estimators=200,
    random_state=42
)
modelo_producao.fit(X, y)

print("✅ Modelo XGBoost treinado!")

# ==============================================================================
# 8. GERAR PREDIÇÃO PARA PRÓXIMAS 2 SEMANAS
# ==============================================================================
print("Gerando predições...")

# Pegar última semana disponível
ultima_semana = df.iloc[-1]

# Preparar features para predição
X_future = pd.DataFrame({
    "temp_min_lag_1": [df["temperatura_minima"].iloc[-1]],
    "umidade_lag_1": [df["umidade_maxima"].iloc[-1]],

    "temp_min_lag_2": [df["temperatura_minima"].iloc[-2]],
    "umidade_lag_2": [df["umidade_maxima"].iloc[-2]],

    "temp_min_lag_3": [df["temperatura_minima"].iloc[-3]],
    "umidade_lag_3": [df["umidade_maxima"].iloc[-3]],

    "densidade_populacional": [df["densidade_populacional"].iloc[-1]],
    "taxa_coleta_residuos": [df["taxa_coleta_residuos"].iloc[-1]],
    "indice_breteu_adl": [df["indice_breteu_adl"].iloc[-1]],

    "casos_lag_1": [df["casos_confirmados"].iloc[-1]],
    "casos_lag_2": [df["casos_confirmados"].iloc[-2]],
    "casos_lag_3": [df["casos_confirmados"].iloc[-3]]
})

# Garante, explicitamente, a mesma ordem usada no treinamento.
X_future = X_future[features]
predicao_semana_1 = modelo_producao.predict(X_future)[0]

# Semana 2: desloca a janela autorregressiva de casos.
X_future_2 = X_future.copy()

X_future_2["casos_lag_1"] = predicao_semana_1
X_future_2["casos_lag_2"] = df["casos_confirmados"].iloc[-1]
X_future_2["casos_lag_3"] = df["casos_confirmados"].iloc[-2]

X_future_2 = X_future_2[features]

predicao_semana_2 = modelo_producao.predict(X_future_2)[0]
print(f"Predição semana 1: {predicao_semana_1:.0f} casos")
print(f"Predição semana 2: {predicao_semana_2:.0f} casos")

# ==============================================================================
# 9. SALVAR PREDIÇÕES NO SUPABASE (TABELA: predicoes_dengue)
# ==============================================================================
data_ultima = ultima_semana['data_semana']
data_semana_1 = data_ultima + pd.Timedelta(days=7)
data_semana_2 = data_ultima + pd.Timedelta(days=14)

print("Salvando predições no Supabase...")

supabase.table("predicoes_dengue").insert({
    "semana_predita": int(data_semana_1.strftime('%Y%m')),
    "data_predicao": data_semana_1.strftime('%Y-%m-%d'),
    "casos_previstos": int(predicao_semana_1),
    "modelo_usado": "XGBoost_v1_github_actions",
    "data_geracao": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}).execute()

supabase.table("predicoes_dengue").insert({
    "semana_predita": int(data_semana_2.strftime('%Y%m')),
    "data_predicao": data_semana_2.strftime('%Y-%m-%d'),
    "casos_previstos": int(predicao_semana_2),
    "modelo_usado": "XGBoost_v1_github_actions",
    "data_geracao": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}).execute()

print("✅ Predições salvas no Supabase!")
print("🎉 Pipeline finalizado com sucesso!")
