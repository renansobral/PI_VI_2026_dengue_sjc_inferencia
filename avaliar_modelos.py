# ==============================================================================
# AVALIAÇÃO DE MODELOS — PI IV UNIVESP
# Executa backtesting temporal; não escreve dados no Supabase.
# ==============================================================================
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from supabase import create_client
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings = __import__("warnings")
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONEXÃO
# ==============================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERRO: SUPABASE_URL ou SUPABASE_KEY não foram configuradas.")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==============================================================================
# 2. LEITURA E PREPARAÇÃO DOS DADOS
# ==============================================================================
print("Lendo dados semanais...")
resposta_casos = supabase.table("casos_dengue_sjc") \
    .select("*") \
    .order("data_semana") \
    .execute()

df = pd.DataFrame(resposta_casos.data)

print("Lendo dados de ADL...")
resposta_adl = supabase.table("adl_indice_sjc") \
    .select("data_referencia, indice_adl") \
    .order("data_referencia") \
    .execute()

df_adl = pd.DataFrame(resposta_adl.data)

if df.empty or df_adl.empty:
    raise RuntimeError("A tabela de casos ou a tabela de ADL está vazia.")

df["data_semana"] = pd.to_datetime(df["data_semana"])
df_adl["data_referencia"] = pd.to_datetime(df_adl["data_referencia"])

df = df.sort_values("data_semana").reset_index(drop=True)
df_adl = df_adl.sort_values("data_referencia").reset_index(drop=True)

df = pd.merge_asof(
    df,
    df_adl.rename(columns={"indice_adl": "indice_breteu_adl"}),
    left_on="data_semana",
    right_on="data_referencia",
    direction="backward"
)

df["indice_breteu_adl"] = df["indice_breteu_adl"].bfill()
df = df.drop(columns=["data_referencia"])

# ==============================================================================
# 3. LAGS E FEATURES DO BASELINE
# ==============================================================================
for lag in [1, 2, 3]:
    df[f"casos_lag_{lag}"] = df["casos_confirmados"].shift(lag)
    df[f"temp_min_lag_{lag}"] = df["temperatura_minima"].shift(lag)
    df[f"umidade_lag_{lag}"] = df["umidade_maxima"].shift(lag)

# ==============================================================================
# FEATURES EXPERIMENTAIS: TENDÊNCIA E SAZONALIDADE
# Todas usam exclusivamente informações anteriores à semana prevista.
# ==============================================================================
df["casos_media_2s"] = (
    df["casos_confirmados"]
    .shift(1)
    .rolling(window=2)
    .mean()
)

df["casos_media_4s"] = (
    df["casos_confirmados"]
    .shift(1)
    .rolling(window=4)
    .mean()
)

df["variacao_casos_1s"] = (
    df["casos_confirmados"].shift(1)
    - df["casos_confirmados"].shift(2)
)

df["semana_ano"] = df["data_semana"].dt.isocalendar().week.astype(int)

df["semana_sin"] = np.sin(
    2 * np.pi * df["semana_ano"] / 52
)

df["semana_cos"] = np.cos(
    2 * np.pi * df["semana_ano"] / 52
)

features_baseline = [
    "temp_min_lag_1",
    "umidade_lag_1",
    "temp_min_lag_2",
    "umidade_lag_2",
    "temp_min_lag_3",
    "umidade_lag_3",
    "densidade_populacional",
    "taxa_coleta_residuos",
    "indice_breteu_adl",
    "casos_lag_1",
    "casos_lag_2",
    "casos_lag_3"
]

features_enriquecidas = features_baseline + [
    "casos_media_2s",
    "casos_media_4s",
    "variacao_casos_1s",
    "semana_sin",
    "semana_cos"
]


colunas_obrigatorias = ["casos_confirmados"] + features_enriquecidas
df = df.dropna(subset=colunas_obrigatorias).reset_index(drop=True)

if len(df) < 80:
    raise RuntimeError(
        f"Dados insuficientes após limpeza: {len(df)} linhas. "
        "São recomendadas ao menos 80 semanas para este teste."
    )

print(f"Semanas válidas para avaliação: {len(df)}")

# ==============================================================================
# 4. FUNÇÃO DE WALK-FORWARD BACKTEST
# ==============================================================================
def walk_forward_backtest(
    dados,
    features,
    parametros,
    horizonte=1,
    semanas_teste=52,
    minimo_treino=120
):
    """
    Para cada ponto da janela de teste:
    - treina usando somente semanas passadas;
    - prevê horizonte 1 ou 2;
    - compara previsão com ocorrência real.
    """

    resultados = []

    primeiro_corte = max(minimo_treino, len(dados) - semanas_teste)

    for corte in range(primeiro_corte, len(dados) - horizonte):
        treino = dados.iloc[:corte].copy()
        alvo = dados.iloc[corte + horizonte].copy()

        modelo = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            **parametros
        )

        modelo.fit(
            treino[features],
            treino["casos_confirmados"]
        )

        X_futuro = dados.iloc[[corte]][features].copy()

        previsao = float(modelo.predict(X_futuro)[0])
        previsao = max(0, previsao)

        real = float(alvo["casos_confirmados"])

        resultados.append({
            "data_corte": dados.iloc[corte]["data_semana"],
            "data_prevista": alvo["data_semana"],
            "horizonte": horizonte,
            "casos_reais": real,
            "casos_previstos": previsao,
            "erro_absoluto": abs(real - previsao),
            "erro_assinado": previsao - real
        })

    return pd.DataFrame(resultados)

# ==============================================================================
# 5. CONFIGURAÇÕES E CONJUNTOS DE FEATURES A COMPARAR
# ==============================================================================

# Duas versões da matriz de entrada:
# - Baseline: 12 variáveis já utilizadas no modelo atual.
# - Enriquecida: baseline + tendência recente + sazonalidade anual.
conjuntos_features = {
    "baseline_12_features": features_baseline,
    "enriquecido_17_features": features_enriquecidas
}

# Dois conjuntos de hiperparâmetros do XGBoost.
configuracoes = {
    "XGBoost_baseline_colab": {
        "learning_rate": 0.2,
        "max_depth": 3,
        "n_estimators": 200
    },
    "XGBoost_temporal_grid": {
        "learning_rate": 0.05,
        "max_depth": 7,
        "n_estimators": 50
    }
}

# ==============================================================================
# 6. EXECUTAR E EXIBIR MÉTRICAS
# ==============================================================================
resumo = []

for nome_features, features_atuais in conjuntos_features.items():
    for nome_modelo, parametros in configuracoes.items():
        print("\n" + "=" * 75)
        print(f"CONJUNTO: {nome_features}")
        print(f"MODELO: {nome_modelo}")
        print("=" * 75)

        for horizonte in [1, 2]:
            resultado = walk_forward_backtest(
                dados=df,
                features=features_atuais,
                parametros=parametros,
                horizonte=horizonte,
                semanas_teste=52,
                minimo_treino=120
            )

            mae = mean_absolute_error(
                resultado["casos_reais"],
                resultado["casos_previstos"]
            )

            rmse = np.sqrt(
                mean_squared_error(
                    resultado["casos_reais"],
                    resultado["casos_previstos"]
                )
            )

            r2 = r2_score(
                resultado["casos_reais"],
                resultado["casos_previstos"]
            )

            vies = resultado["erro_assinado"].mean()

            resumo.append({
                "features": nome_features,
                "modelo": nome_modelo,
                "horizonte_semanas": horizonte,
                "amostras": len(resultado),
                "r2": round(r2, 4),
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "vies_medio": round(vies, 2)
            })

            print(
                f"Horizonte {horizonte} semana(s) | "
                f"amostras={len(resultado)} | "
                f"R²={r2:.4f} | "
                f"MAE={mae:.2f} | "
                f"RMSE={rmse:.2f} | "
                f"viés={vies:.2f}"
            )

df_resumo = pd.DataFrame(resumo).sort_values(
    ["horizonte_semanas", "rmse", "mae"]
)

print("\n" + "=" * 90)
print("RESUMO FINAL — WALK-FORWARD BACKTEST: BASELINE VS TENDÊNCIA/SAZONALIDADE")
print("=" * 90)
print(df_resumo.to_string(index=False))
