import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from supabase import create_client
from datetime import datetime, timedelta
import os

# =====================
# 1. CONEXÃO COM SUPABASE
# =====================
SUPABASE_URL = os.getenv("https://bwpezxfpgtdwyjirwfbv.supabase.co/rest/v1/")
SUPABASE_KEY = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ3cGV6eGZwZ3Rkd3lqaXJ3ZmJ2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODc3NTA4OTksImV4cCI6MjEwMzMyNjg5OX0.PtZkUhyynYkPULhxK1W4b1nwEgTdySMPyPljYHPmKrg")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =====================
# 2. LER DADOS DO SUPABASE
# =====================
print("Lendo dados do Supabase...")
data = supabase.table("casos_dengue_sjc").select("*").order("data_semana", desc=True).limit(100).execute()
df = pd.DataFrame(data.data)
df = df.sort_values("data_semana").reset_index(drop=True)
print(f"Total de registros: {len(df)}")

# =====================
# 3. PREPARAR FEATURES
# =====================
df['lag_casos_1'] = df['casos_confirmados'].shift(1)
df['lag_casos_2'] = df['casos_confirmados'].shift(2)
df['lag_casos_3'] = df['casos_confirmados'].shift(3)
df['lag_temp_1'] = df['temperatura_minima'].shift(1)
df['lag_temp_2'] = df['temperatura_minima'].shift(2)
df['lag_temp_3'] = df['temperatura_minima'].shift(3)
df['lag_umid_1'] = df['umidade_maxima'].shift(1)
df['lag_umid_2'] = df['umidade_maxima'].shift(2)
df['lag_umid_3'] = df['umidade_maxima'].shift(3)

df_model = df.dropna().reset_index(drop=True)

# =====================
# 4. CARREGAR MODELO
# =====================
print("Carregando modelo XGBoost...")

# Carregar modelo do repositório
with open("modelo_xgboost.pkl", "rb") as f:
    model = pickle.load(f)

# =====================
# 5. GERAR PREDIÇÃO
# =====================
ultima_semana = df_model.iloc[-1]
print(f"Última semana: {ultima_semana['data_semana']}")

X_future = pd.DataFrame({
    'lag_casos_1': [df_model['casos_confirmados'].iloc[-1]],
    'lag_casos_2': [df_model['casos_confirmados'].iloc[-2]],
    'lag_casos_3': [df_model['casos_confirmados'].iloc[-3]],
    'lag_temp_1': [df_model['temperatura_minima'].iloc[-1]],
    'lag_temp_2': [df_model['temperatura_minima'].iloc[-2]],
    'lag_temp_3': [df_model['temperatura_minima'].iloc[-3]],
    'lag_umid_1': [df_model['umidade_maxima'].iloc[-1]],
    'lag_umid_2': [df_model['umidade_maxima'].iloc[-2]],
    'lag_umid_3': [df_model['umidade_maxima'].iloc[-3]],
    'densidade_populacional': [df_model['densidade_populacional'].iloc[-1]] if 'densidade_populacional' in df_model.columns else [None],
    'taxa_coleta_residuos': [df_model['taxa_coleta_residuos'].iloc[-1]] if 'taxa_coleta_residuos' in df_model.columns else [None],
    'indice_breteu_adl': [df_model['indice_breteu_adl'].iloc[-1]] if 'indice_breteu_adl' in df_model.columns else [None]
})

predicao_semana_1 = model.predict(X_future)[0]
print(f"Predição semana 1: {predicao_semana_1:.0f} casos")

X_future_2 = X_future.copy()
X_future_2['lag_casos_1'] = predicao_semana_1
predicao_semana_2 = model.predict(X_future_2)[0]
print(f"Predição semana 2: {predicao_semana_2:.0f} casos")

# =====================
# 6. SALVAR NO SUPABASE
# =====================
data_ultima = pd.to_datetime(ultima_semana['data_semana'])
data_semana_1 = data_ultima + timedelta(days=7)
data_semana_2 = data_ultima + timedelta(days=14)

print("Salvando predições no Supabase...")

supabase.table("predicoes_dengue").insert({
    "semana_predita": int(data_semana_1.strftime('%Y%m')),
    "data_predicao": data_semana_1.strftime('%Y-%m-%d'),
    "casos_previstos": int(predicao_semana_1),
    "modelo_usado": "XGBoost_v1_github_actions",
    "data_geracao": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}).execute()

supabase.table("predicoes_dengue").insert({
    "semana_predita": int(data_semana_2.strftime('%Y%m')),
    "data_predicao": data_semana_2.strftime('%Y-%m-%d'),
    "casos_previstos": int(predicao_semana_2),
    "modelo_usado": "XGBoost_v1_github_actions",
    "data_geracao": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}).execute()

print("✅ Predições salvas com sucesso!")
