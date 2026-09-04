# ==============================================================================
# PROJETO INTEGRADOR IV - UNIVESP
# SIMULADOR DE PREVISÕES HISTÓRICAS — DENGUE EM SÃO JOSÉ DOS CAMPOS
#
# Metodologia:
# Para cada semana-alvo t, o XGBoost é retreinado exclusivamente com dados
# anteriores a t. A previsão é gravada em predicoes_dengue por upsert.
# ==============================================================================
import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from supabase import create_client

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PARÂMETROS DA SIMULAÇÃO
# ==============================================================================
ANO_SIMULACAO = 2025

# Teste inicial seguro:
# - 4 = processa apenas as quatro primeiras semanas de 2025.
# - None = processa todas as semanas de 2025.
LIMITE_SEMANAS = None

MODELO_USADO = "XGBoost_v2_17features_simulacao_2025"

PARAMETROS_XGBOOST = {
    "learning_rate": 0.2,
    "max_depth": 3,
    "n_estimators": 200,
    "random_state": 42
}

FEATURES = [
    "temp_min_lag_1", "umidade_lag_1",
    "temp_min_lag_2", "umidade_lag_2",
    "temp_min_lag_3", "umidade_lag_3",
    "densidade_populacional", "taxa_coleta_residuos", "indice_breteu_adl",
    "casos_lag_1", "casos_lag_2", "casos_lag_3",
    "casos_media_2s", "casos_media_4s", "variacao_casos_1s",
    "semana_sin", "semana_cos"
]

print("=" * 80)
print("SIMULADOR DE PREVISÕES HISTÓRICAS — DENGUE SJC")
print("=" * 80)
print(f"Ano simulado: {ANO_SIMULACAO}")
print(f"Limite de semanas: {LIMITE_SEMANAS}")
print("Regra: retreinamento completo antes de cada previsão.")
print("=" * 80)

# ==============================================================================
# 2. CONEXÃO COM SUPABASE
# ==============================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL não foi encontrada nos GitHub Secrets.")

if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_KEY não foi encontrada nos GitHub Secrets.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("✅ Conectado ao Supabase.")

# ==============================================================================
# 3. CARREGAR DADOS EPIDEMIOLÓGICOS
# ==============================================================================
print("Lendo dados da tabela casos_dengue_sjc...")

casos_data = (
    supabase
    .table("casos_dengue_sjc")
    .select("*")
    .order("data_semana", desc=False)
    .execute()
)

df_casos = pd.DataFrame(casos_data.data)

if df_casos.empty:
    raise RuntimeError("A tabela casos_dengue_sjc não possui registros.")

df_casos["data_semana"] = pd.to_datetime(df_casos["data_semana"])
df_casos = df_casos.sort_values("data_semana").reset_index(drop=True)

print(f"Registros epidemiológicos carregados: {len(df_casos)}")

# ==============================================================================
# 4. CARREGAR E MESCLAR ADL / LIRAA
# ==============================================================================
print("Lendo índice ADL (LIRAa)...")

adl_data = (
    supabase
    .table("adl_indice_sjc")
    .select("*")
    .order("data_referencia", desc=False)
    .execute()
)

df_adl = pd.DataFrame(adl_data.data)

if df_adl.empty:
    raise RuntimeError(
        "A tabela adl_indice_sjc está vazia. "
        "Cadastre ao menos um ciclo de ADL antes da simulação."
    )

df_adl["data_referencia"] = pd.to_datetime(df_adl["data_referencia"])
df_adl = df_adl.sort_values("data_referencia").reset_index(drop=True)

df_completo = pd.merge_asof(
    df_casos.sort_values("data_semana"),
    df_adl[["data_referencia", "indice_adl"]].rename(
        columns={"indice_adl": "indice_breteu_adl"}
    ),
    left_on="data_semana",
    right_on="data_referencia",
    direction="backward"
)

# Semanas anteriores ao primeiro ciclo recebem o primeiro ADL disponível.
df_completo["indice_breteu_adl"] = df_completo["indice_breteu_adl"].bfill()

if df_completo["indice_breteu_adl"].isna().any():
    raise RuntimeError(
        "Existem semanas sem ADL após o cruzamento. "
        "Verifique as datas em adl_indice_sjc."
    )

df_completo = df_completo.drop(columns=["data_referencia"])

# Converter as colunas numéricas necessárias.
colunas_numericas = [
    "casos_confirmados",
    "temperatura_minima",
    "umidade_maxima",
    "densidade_populacional",
    "taxa_coleta_residuos",
    "indice_breteu_adl"
]

for coluna in colunas_numericas:
    df_completo[coluna] = pd.to_numeric(
        df_completo[coluna],
        errors="coerce"
    )

df_completo = df_completo.dropna(
    subset=colunas_numericas
).sort_values("data_semana").reset_index(drop=True)

print(f"Registros válidos após preparação: {len(df_completo)}")

# ==============================================================================
# 5. DEFINIR AS SEMANAS-ALVO
# ==============================================================================
semanas_alvo = df_completo[
    df_completo["data_semana"].dt.year == ANO_SIMULACAO
].copy()

if semanas_alvo.empty:
    raise RuntimeError(
        f"Não foram encontradas semanas do ano {ANO_SIMULACAO} "
        "na tabela casos_dengue_sjc."
    )

semanas_alvo = semanas_alvo.sort_values("data_semana").reset_index(drop=True)

if LIMITE_SEMANAS is not None:
    semanas_alvo = semanas_alvo.head(LIMITE_SEMANAS).copy()

print(
    f"Semanas selecionadas para simulação: {len(semanas_alvo)} "
    f"({semanas_alvo['data_semana'].min().strftime('%d/%m/%Y')} até "
    f"{semanas_alvo['data_semana'].max().strftime('%d/%m/%Y')})"
)

# ==============================================================================
# 6. FUNÇÃO PARA PREPARAR TREINO HISTÓRICO
# ==============================================================================
def criar_features_treino(dados):
    """
    Cria as 17 features para as semanas de treinamento.
    Cada linha usa somente semanas anteriores por meio de shift(1).
    """
    treino = dados.copy().sort_values("data_semana").reset_index(drop=True)

    for lag in [1, 2, 3]:
        treino[f"casos_lag_{lag}"] = treino["casos_confirmados"].shift(lag)
        treino[f"temp_min_lag_{lag}"] = treino["temperatura_minima"].shift(lag)
        treino[f"umidade_lag_{lag}"] = treino["umidade_maxima"].shift(lag)

    treino["casos_media_2s"] = (
        treino["casos_confirmados"]
        .shift(1)
        .rolling(window=2)
        .mean()
    )

    treino["casos_media_4s"] = (
        treino["casos_confirmados"]
        .shift(1)
        .rolling(window=4)
        .mean()
    )

    treino["variacao_casos_1s"] = (
        treino["casos_confirmados"].shift(1)
        - treino["casos_confirmados"].shift(2)
    )

    treino["semana_ano"] = (
        treino["data_semana"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    treino["semana_sin"] = np.sin(
        2 * np.pi * treino["semana_ano"] / 52
    )

    treino["semana_cos"] = np.cos(
        2 * np.pi * treino["semana_ano"] / 52
    )

    treino = treino.dropna(
        subset=["casos_confirmados"] + FEATURES
    ).reset_index(drop=True)

    return treino


# ==============================================================================
# 7. SIMULAR SEMANA A SEMANA E SALVAR NO SUPABASE
# ==============================================================================
resultados = []

for indice, linha_alvo in semanas_alvo.iterrows():
    data_alvo = linha_alvo["data_semana"]

    print("\n" + "-" * 80)
    print(
        f"Semana {indice + 1}/{len(semanas_alvo)} | "
        f"Previsão para: {data_alvo.strftime('%d/%m/%Y')}"
    )

    # Recorte rigoroso: somente dados antes da data-alvo.
    historico_bruto = df_completo[
        df_completo["data_semana"] < data_alvo
    ].copy()

    treino = criar_features_treino(historico_bruto)

    if len(treino) < 120:
        print(
            f"⚠️ Semana ignorada: treino insuficiente ({len(treino)} registros)."
        )
        continue

    X_treino = treino[FEATURES].astype(float)
    y_treino = treino["casos_confirmados"].astype(float)

    # Retreinamento do modelo para esta semana-alvo.
    modelo = xgb.XGBRegressor(**PARAMETROS_XGBOOST)
    modelo.fit(X_treino, y_treino)

    # Últimas observações realmente disponíveis antes da semana-alvo.
    historico = historico_bruto.sort_values(
        "data_semana"
    ).reset_index(drop=True)

    if len(historico) < 4:
        print("⚠️ Semana ignorada: histórico insuficiente para lags e médias.")
        continue

    semana_ano = int(data_alvo.isocalendar().week)

    # Linha da semana futura: valores estruturais/ADL da semana-alvo e
    # lags/tendência somente do histórico anterior.
    X_alvo = pd.DataFrame({
        "temp_min_lag_1": [
            float(historico["temperatura_minima"].iloc[-1])
        ],
        "umidade_lag_1": [
            float(historico["umidade_maxima"].iloc[-1])
        ],
        "temp_min_lag_2": [
            float(historico["temperatura_minima"].iloc[-2])
        ],
        "umidade_lag_2": [
            float(historico["umidade_maxima"].iloc[-2])
        ],
        "temp_min_lag_3": [
            float(historico["temperatura_minima"].iloc[-3])
        ],
        "umidade_lag_3": [
            float(historico["umidade_maxima"].iloc[-3])
        ],
        "densidade_populacional": [
            float(linha_alvo["densidade_populacional"])
        ],
        "taxa_coleta_residuos": [
            float(linha_alvo["taxa_coleta_residuos"])
        ],
        "indice_breteu_adl": [
            float(linha_alvo["indice_breteu_adl"])
        ],
        "casos_lag_1": [
            float(historico["casos_confirmados"].iloc[-1])
        ],
        "casos_lag_2": [
            float(historico["casos_confirmados"].iloc[-2])
        ],
        "casos_lag_3": [
            float(historico["casos_confirmados"].iloc[-3])
        ],
        "casos_media_2s": [
            float(historico["casos_confirmados"].iloc[-2:].mean())
        ],
        "casos_media_4s": [
            float(historico["casos_confirmados"].iloc[-4:].mean())
        ],
        "variacao_casos_1s": [
            float(
                historico["casos_confirmados"].iloc[-1]
                - historico["casos_confirmados"].iloc[-2]
            )
        ],
        "semana_sin": [
            float(np.sin(2 * np.pi * semana_ano / 52))
        ],
        "semana_cos": [
            float(np.cos(2 * np.pi * semana_ano / 52))
        ]
    })[FEATURES].astype(float)

    previsao = float(modelo.predict(X_alvo)[0])
    previsao = max(0, previsao)

    caso_real = float(linha_alvo["casos_confirmados"])
    erro_absoluto = abs(caso_real - previsao)

    registro = {
        "semana_predita": int(data_alvo.strftime("%Y%W")),
        "data_predicao": data_alvo.strftime("%Y-%m-%d"),
        "casos_previstos": int(round(previsao)),
        "modelo_usado": MODELO_USADO
    }

    # Upsert: insere se a semana não existe; atualiza se já existe.
    (
        supabase
        .table("predicoes_dengue")
        .upsert(
            registro,
            on_conflict="data_predicao"
        )
        .execute()
    )

    resultados.append({
        "data_predicao": data_alvo.strftime("%Y-%m-%d"),
        "casos_reais": round(caso_real, 2),
        "casos_previstos": round(previsao, 2),
        "erro_absoluto": round(erro_absoluto, 2)
    })

    print(
        f"✅ Previsto: {previsao:.2f} | "
        f"Real: {caso_real:.0f} | "
        f"Erro absoluto: {erro_absoluto:.2f}"
    )

# ==============================================================================
# 8. RESUMO FINAL
# ==============================================================================
df_resultados = pd.DataFrame(resultados)

print("\n" + "=" * 80)
print("RESUMO FINAL — SIMULAÇÃO HISTÓRICA")
print("=" * 80)

if df_resultados.empty:
    print("Nenhuma previsão foi gerada. Verifique as mensagens anteriores.")
else:
    y_real = df_resultados["casos_reais"].values
    y_pred = df_resultados["casos_previstos"].values

    mae = np.mean(np.abs(y_real - y_pred))
    rmse = np.sqrt(np.mean((y_real - y_pred) ** 2))

    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    print(f"Semanas previstas e gravadas: {len(df_resultados)}")
    print(f"R²:   {r2:.4f}")
    print(f"MAE:  {mae:.2f} casos")
    print(f"RMSE: {rmse:.2f} casos")
    print("-" * 80)
    print(df_resultados.to_string(index=False))
    print("=" * 80)

print("🎉 Simulação histórica finalizada.")
