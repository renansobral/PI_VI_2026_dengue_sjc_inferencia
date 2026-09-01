# ==============================================================================
# COLETA DE DADOS DO INFO DENGUE PARA SJC
# Script separado para ingestao de dados no Supabase
# ==============================================================================
import requests
import pandas as pd
from supabase import create_client
import os
import sys

print("Iniciando a coleta de dados do InfoDengue...")

# ==============================================================================
# 1. CONEXAO COM SUPABASE
# ==============================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERRO: SUPABASE_URL ou SUPABASE_KEY nao encontradas nas variaveis de ambiente.")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("✅ Conectado ao Supabase!")

# ==============================================================================
# 2. PARAMETROS DA API INFO DENGUE (SJC = geocode 3549904)
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

print(f"Requisitando dados do InfoDengue para SJC (geocode {params['geocode']})...")

# ==============================================================================
# 3. REQUISICAO HTTP E TRATAMENTO
# ==============================================================================
try:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"ERRO ao requisitar InfoDengue: {e}")
    sys.exit(1)

# Parse do CSV para DataFrame
df = pd.read_csv(pd.io.common.StringIO(response.text))

print(f"InfoDengue retornou {len(df)} registros brutos.")

# ==============================================================================
# 4. LIMPEZA E MAPEAMENTO DE COLUNAS
# ==============================================================================
df['data'] = pd.to_datetime(df['datainiSE'])
df = df[['data', 'casos', 'tempmin', 'umidmax', 'pop']].copy()
df.columns = ['data_semana', 'casos_confirmados', 'temperatura_minima', 'umidade_maxima', 'populacao']

# Converte data para string ISO para o Supabase
df['data_semana'] = df['data_semana'].dt.strftime('%Y-%m-%d')

# ==============================================================================
# 5. UPSERT NO SUPABASE (evita duplicatas)
# ==============================================================================
registros = df.to_dict('records')

try:
    resultado = supabase.table("casos_dengue_sjc").upsert(
        registros,
        on_conflict="data_semana"
    ).execute()
    print(f"✅ {len(registros)} registros inseridos/atualizados na tabela casos_dengue_sjc.")
except Exception as e:
    print(f"ERRO ao fazer upsert no Supabase: {e}")
    sys.exit(1)

print("🎉 Coleta finalizada com sucesso!")
