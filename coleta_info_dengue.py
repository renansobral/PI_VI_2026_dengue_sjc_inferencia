# ==============================================================================
# COLETA DE DADOS DO INFODENGUE PARA SJC
# Script de ingestão de dados no Supabase
# ==============================================================================
import os
import sys
from io import StringIO

import pandas as pd
import requests
from supabase import create_client

print("Iniciando a coleta de dados do InfoDengue...")

# ==============================================================================
# 1. CONEXÃO COM SUPABASE
# ==============================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print(
        "ERRO: SUPABASE_URL ou SUPABASE_KEY não foram encontradas "
        "nas variáveis de ambiente."
    )
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("✅ Conectado ao Supabase!")

# ==============================================================================
# 2. PARÂMETROS DA API INFODENGUE
# São José dos Campos/SP: código IBGE 3549904
# ==============================================================================
url = "https://info.dengue.mat.br/api/alertcity"

params = {
    "geocode": 3549904,
    "disease": "dengue",
    "format": "csv",
    "ew_start": 1,
    "ew_end": 53,
    "ey_start": 2021,
    "ey_end": 2026
}

print(
    f"Requisitando dados do InfoDengue para SJC "
    f"(geocode {params['geocode']})..."
)

# ==============================================================================
# 3. REQUISIÇÃO HTTP
# ==============================================================================
try:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
except requests.exceptions.RequestException as erro:
    print(f"ERRO ao requisitar InfoDengue: {erro}")
    sys.exit(1)

df = pd.read_csv(StringIO(response.text))

if df.empty:
    print("ERRO: o InfoDengue retornou uma base vazia.")
    sys.exit(1)

print(f"InfoDengue retornou {len(df)} registros brutos.")

# ==============================================================================
# 4. LIMPEZA E PADRONIZAÇÃO DE COLUNAS
# ==============================================================================
colunas_necessarias = [
    "data_iniSE",
    "casos",
    "tempmin",
    "umidmax"
]

colunas_ausentes = [
    coluna for coluna in colunas_necessarias
    if coluna not in df.columns
]

if colunas_ausentes:
    print(
        "ERRO: a API retornou um formato inesperado. "
        f"Colunas ausentes: {colunas_ausentes}"
    )
    print(f"Colunas disponíveis: {df.columns.tolist()}")
    sys.exit(1)

df = df[colunas_necessarias].copy()

df = df.rename(
    columns={
        "data_iniSE": "data_semana",
        "casos": "casos_confirmados",
        "tempmin": "temperatura_minima",
        "umidmax": "umidade_maxima"
    }
)

df["data_semana"] = pd.to_datetime(df["data_semana"])
df = df.dropna().sort_values("data_semana").drop_duplicates(
    subset=["data_semana"],
    keep="last"
)

df["semana_epi"] = df["data_semana"].dt.isocalendar().week.astype(int)
df["data_semana"] = df["data_semana"].dt.strftime("%Y-%m-%d")

print(f"Registros válidos após limpeza: {len(df)}")

# ==============================================================================
# 5. UPSERT NO SUPABASE
# Atualiza somente os dados que pertencem ao InfoDengue.
# ==============================================================================
registros = df[
    [
        "semana_epi",
        "data_semana",
        "casos_confirmados",
        "temperatura_minima",
        "umidade_maxima"
    ]
].to_dict("records")

try:
    supabase.table("casos_dengue_sjc").upsert(
        registros,
        on_conflict="data_semana"
    ).execute()

    print(
        f"✅ {len(registros)} registros inseridos ou atualizados "
        "em casos_dengue_sjc."
    )
except Exception as erro:
    print(f"ERRO ao fazer upsert no Supabase: {erro}")
    sys.exit(1)

print("🎉 Coleta finalizada com sucesso!")
