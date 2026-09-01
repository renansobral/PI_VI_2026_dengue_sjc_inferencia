# ==============================================================================
# PROJETO INTEGRADOR IV - UNIVESP
# MOTOR DE INFERÊNCIA E INGESTÃO DE DADOS (VERSÃO FINAL)
# ==============================================================================
import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
import warnings
import joblib
import os
from supabase import create_client

warnings.filterwarnings('ignore')

print("Iniciando o Pipeline de Inferência...")

# ==============================================================================
# 1. CONEXÃO COM SUPABASE
# ==============================================================================
SUPABASE_URL = os.getenv("https://bwpezxfpgtdwyjirwfbv.supabase.co/rest/v1/")
SUPABASE_KEY = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ3cGV6eGZwZ3Rkd3lqaXJ3ZmJ2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODc3NTA4OTksImV4cCI6MjEwMzMyNjg5OX0.PtZkUhyynYkPULhxK1W4b1nwEgTdySMPyPljYHPmKrg")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("✅ Conectado ao Supabase!")

# ==============================================================================
# 2. LER DADOS DO SUPABASE (TABELA: casos_dengue_sjc)
# ==============================================================================
print("Lendo dados do Supabase...")
data = supabase.table("casos_dengue_sjc").select("*").order("data_semana", asc=True).execute()
df = pd.DataFrame(data.data)

print(f"Total de registros: {len(df)}")

# ==============================================================================
# 3. PREPARAR DADOS (SE NECESSÁRIO)
# ==============================================================================
# Converter data para datetime
df['data_semana'] = pd.to_datetime(df['data_semana'])
df = df.sort_values('data_semana').reset_index(drop=True)

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
# 7. SALVAR MODELO
# ==============================================================================
joblib.dump(modelo_producao, 'modelo_xgboost.pkl')
print("✅ Modelo salvo como 'modelo_xgboost.pkl'!")

# ==============================================================================
# 8. GERAR PREDIÇÃO PARA PRÓXIMAS 2 SEMANAS
# ==============================================================================
print("Gerando predições...")

# Pegar última semana disponível
ultima_semana = df.iloc[-1]

# Preparar features para predição
X_future = pd.DataFrame({
    'lag_casos_1': [df['casos_confirmados'].iloc[-1]],
    'lag_casos_2': [df['casos_confirmados'].iloc[-2]],
    'lag_casos_3': [df['casos_confirmados'].iloc[-3]],
    'lag_temp_1': [df['temperatura_minima'].iloc[-1]],
    'lag_temp_2': [df['temperatura_minima'].iloc[-2]],
    'lag_temp_3': [df['temperatura_minima'].iloc[-3]],
    'lag_umid_1': [df['umidade_maxima'].iloc[-1]],
    'lag_umid_2': [df['umidade_maxima'].iloc[-2]],
    'lag_umid_3': [df['umidade_maxima'].iloc[-3]],
    'densidade_populacional': [df['densidade_populacional'].iloc[-1]],
    'taxa_coleta_residuos': [df['taxa_coleta_residuos'].iloc[-1]],
    'indice_breteu_adl': [df['indice_breteu_adl'].iloc[-1]]
})

predicao_semana_1 = modelo_producao.predict(X_future)[0]

# Semana 2: usar predição da semana 1 como lag
X_future_2 = X_future.copy()
X_future_2['lag_casos_1'] = predicao_semana_1
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
