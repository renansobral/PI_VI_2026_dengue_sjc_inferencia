# ==============================================================================
# PROJETO INTEGRADOR IV - UNIVESP
# MOTOR DE INFERÊNCIA E INGESTÃO DE DADOS (VERSÃO FINAL - 17 FEATURES)
# ==============================================================================
import pandas as pd
import numpy as np
import xgboost as xgb
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
    f"{df_adl['data_referencia'].max().strftime('%d/%m/%Y')}")
    
# ==============================================================================
# 4. ENGENHARIA DE RECURSOS (LAG FEATURES + TENDÊNCIA + SAZONALIDADE)
# ==============================================================================
for lag in [1, 2, 3]:
    df[f"casos_lag_{lag}"] = df["casos_confirmados"].shift(lag)
    df[f"temp_min_lag_{lag}"] = df["temperatura_minima"].shift(lag)
    df[f"umidade_lag_{lag}"] = df["umidade_maxima"].shift(lag)

# Features de tendência e sazonalidade
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

# Remover NaNs (primeiras semanas sem lags e sem médias móveis)
df = df.dropna().reset_index(drop=True)

# ==============================================================================
# 5. DEFINIÇÃO DAS 17 FEATURES
# ==============================================================================
features = [
    # Lags de clima
    'temp_min_lag_1', 'umidade_lag_1',
    'temp_min_lag_2', 'umidade_lag_2',
    'temp_min_lag_3', 'umidade_lag_3',
    # Infraestrutura e ADL
    'densidade_populacional', 'taxa_coleta_residuos', 'indice_breteu_adl',
    # Lags de casos
    'casos_lag_1', 'casos_lag_2', 'casos_lag_3',
    # Tendência recente
    'casos_media_2s', 'casos_media_4s', 'variacao_casos_1s',
    # Sazonalidade anual
    'semana_sin', 'semana_cos'
]

# ==============================================================================
# 6. BACKTESTING WALK-FORWARD (ÚLTIMAS 52 SEMANAS)
# ==============================================================================
print("Iniciando backtesting walk-forward (últimas 52 semanas)...")

# Garantir que há pelo menos 52 semanas após o início da série
if len(df) < 104:
    raise RuntimeError(
        "Série temporal muito curta para backtest de 52 semanas. "
        f"Atualmente há {len(df)} semanas válidas."
    )

# Definir janela de backtest
janela_backtest = 52
df_backtest = df.iloc[-janela_backtest:].copy().reset_index(drop=True)

# Histórico inicial: todos os dados antes da janela de backtest
historico = df.iloc[:-janela_backtest].copy().reset_index(drop=True)

resultados_backtest = []

for i in range(len(df_backtest)):
    linha_alvo = df_backtest.iloc[i]
    data_alvo = linha_alvo["data_semana"]

    # Construir features para a semana alvo usando apenas dados até i-1
    # Precisamos dos últimos 3 valores de casos, clima e das médias móveis
    if len(historico) < 4:
        # Não há histórico suficiente para calcular todas as features
        historico = pd.concat([historico, linha_alvo.to_frame().T], ignore_index=True)
        continue

    # Lags de clima
    temp_min_lag_1 = float(historico["temperatura_minima"].iloc[-1])
    umidade_lag_1 = float(historico["umidade_maxima"].iloc[-1])
    temp_min_lag_2 = float(historico["temperatura_minima"].iloc[-2])
    umidade_lag_2 = float(historico["umidade_maxima"].iloc[-2])
    temp_min_lag_3 = float(historico["temperatura_minima"].iloc[-3])
    umidade_lag_3 = float(historico["umidade_maxima"].iloc[-3])

    # Lags de casos
    casos_lag_1 = float(historico["casos_confirmados"].iloc[-1])
    casos_lag_2 = float(historico["casos_confirmados"].iloc[-2])
    casos_lag_3 = float(historico["casos_confirmados"].iloc[-3])

    # Tendência
    casos_com_shift = historico["casos_confirmados"].shift(1)
    casos_media_2s = float(casos_com_shift.iloc[-2:].mean())
    casos_media_4s = float(casos_com_shift.iloc[-4:].mean())
    variacao_casos_1s = float(
        historico["casos_confirmados"].iloc[-2] - historico["casos_confirmados"].iloc[-3]
    )

    # Sazonalidade
    semana_ano_alvo = int(linha_alvo["data_semana"].isocalendar().week)
    semana_sin = float(np.sin(2 * np.pi * semana_ano_alvo / 52))
    semana_cos = float(np.cos(2 * np.pi * semana_ano_alvo / 52))

    X_alvo = pd.DataFrame({
        'temp_min_lag_1': [temp_min_lag_1],
        'umidade_lag_1': [umidade_lag_1],
        'temp_min_lag_2': [temp_min_lag_2],
        'umidade_lag_2': [umidade_lag_2],
        'temp_min_lag_3': [temp_min_lag_3],
        'umidade_lag_3': [umidade_lag_3],
        'densidade_populacional': [float(linha_alvo["densidade_populacional"])],
        'taxa_coleta_residuos': [float(linha_alvo["taxa_coleta_residuos"])],
        'indice_breteu_adl': [float(linha_alvo["indice_breteu_adl"])],
        'casos_lag_1': [casos_lag_1],
        'casos_lag_2': [casos_lag_2],
        'casos_lag_3': [casos_lag_3],
        'casos_media_2s': [casos_media_2s],
        'casos_media_4s': [casos_media_4s],
        'variacao_casos_1s': [variacao_casos_1s],
        'semana_sin': [semana_sin],
        'semana_cos': [semana_cos]
    })

    # Garantir que todas as colunas são float
    X_alvo = X_alvo.astype(float)
    X_alvo = X_alvo[features]

    # Treinar modelo com todos os dados até i-1 (retreinamento a cada semana)
    X_treino = historico[features].astype(float)
    y_treino = historico["casos_confirmados"].astype(float)

    modelo_backtest = xgb.XGBRegressor(
        learning_rate=0.2,
        max_depth=3,
        n_estimators=200,
        random_state=42
    )
    modelo_backtest.fit(X_treino, y_treino)

    previsao = float(modelo_backtest.predict(X_alvo)[0])
    previsao = max(0, previsao)

    resultados_backtest.append({
        "data_semana": data_alvo.strftime("%Y-%m-%d"),
        "casos_reais": int(linha_alvo["casos_confirmados"]),
        "casos_previstos": round(previsao, 2),
        "erro_absoluto": round(abs(linha_alvo["casos_confirmados"] - previsao), 2)
    })

    # Adicionar semana real ao histórico para a próxima iteração
    
    historico = pd.concat([historico, linha_alvo.to_frame().T], ignore_index=True)

df_resultados_backtest = pd.DataFrame(resultados_backtest)

if len(df_resultados_backtest) > 0:
    y_real = df_resultados_backtest["casos_reais"].values
    y_pred = df_resultados_backtest["casos_previstos"].values

    mae_backtest = np.mean(np.abs(y_real - y_pred))
    rmse_backtest = np.sqrt(np.mean((y_real - y_pred) ** 2))

    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    r2_backtest = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    print("=" * 75)
    print("RESULTADO DO BACKTESTING WALK-FORWARD — XGBOOST (17 FEATURES)")
    print("=" * 75)
    print(f"Período de teste: {df_resultados_backtest['data_semana'].iloc[0]} "
          f"até {df_resultados_backtest['data_semana'].iloc[-1]}")
    print(f"Semanas avaliadas: {len(df_resultados_backtest)}")
    print(f"R²:   {r2_backtest:.4f}")
    print(f"MAE:  {mae_backtest:.2f} casos")
    print(f"RMSE: {rmse_backtest:.2f} casos")
    print("=" * 75)
    print(df_resultados_backtest.to_string(index=False))
    print("=" * 75)
else:
    print("Nenhuma semana válida para backtest walk-forward.")

# ==============================================================================
# 7. TREINAR MODELO FINAL COM TODOS OS DADOS
# ==============================================================================
print("Treinando modelo final com todos os dados disponíveis...")

X_final = df[features]
y_final = df["casos_confirmados"]

modelo_producao = xgb.XGBRegressor(
    learning_rate=0.2,
    max_depth=3,
    n_estimators=200,
    random_state=42
)
modelo_producao.fit(X_final, y_final)

print(f"✅ Modelo XGBoost treinado com {len(df)} semanas!")

# ==============================================================================
# 8. GERAR PREDIÇÃO PARA A PRÓXIMA SEMANA (QUE AINDA NÃO EXISTE NA BASE)
# ==============================================================================
print("Gerando predição para a próxima semana...")

# Última semana disponível na base
ultima_semana = df.iloc[-1]
data_ultima = ultima_semana["data_semana"]

# Próxima semana (ainda não existe na base)
proxima_semana = data_ultima + pd.Timedelta(days=7)

# Semana do ano da próxima semana (Timestamp direto, sem .dt)
semana_ano_proxima = int(proxima_semana.isocalendar().week)

# Construir features para a próxima semana
X_futuro = pd.DataFrame({
    # Lags de clima
    "temp_min_lag_1": [df["temperatura_minima"].iloc[-1]],
    "umidade_lag_1": [df["umidade_maxima"].iloc[-1]],
    "temp_min_lag_2": [df["temperatura_minima"].iloc[-2]],
    "umidade_lag_2": [df["umidade_maxima"].iloc[-2]],
    "temp_min_lag_3": [df["temperatura_minima"].iloc[-3]],
    "umidade_lag_3": [df["umidade_maxima"].iloc[-3]],
    # Infraestrutura e ADL
    "densidade_populacional": [df["densidade_populacional"].iloc[-1]],
    "taxa_coleta_residuos": [df["taxa_coleta_residuos"].iloc[-1]],
    "indice_breteu_adl": [df["indice_breteu_adl"].iloc[-1]],
    # Lags de casos
    "casos_lag_1": [df["casos_confirmados"].iloc[-1]],
    "casos_lag_2": [df["casos_confirmados"].iloc[-2]],
    "casos_lag_3": [df["casos_confirmados"].iloc[-3]],
    # Tendência recente
    "casos_media_2s": [
        df["casos_confirmados"].shift(1).iloc[-2:].mean()
    ],
    "casos_media_4s": [
        df["casos_confirmados"].shift(1).iloc[-4:].mean()
    ],
    "variacao_casos_1s": [
        df["casos_confirmados"].iloc[-2] - df["casos_confirmados"].iloc[-3]
    ],
    # Sazonalidade anual
    "semana_sin": [np.sin(2 * np.pi * semana_ano_proxima / 52)],
    "semana_cos": [np.cos(2 * np.pi * semana_ano_proxima / 52)]
})[features]

predicao_proxima_semana = modelo_producao.predict(X_futuro)[0]
predicao_proxima_semana = max(0, predicao_proxima_semana)

print(f"Predição para {proxima_semana.strftime('%d/%m/%Y')}: {predicao_proxima_semana:.0f} casos")

# ==============================================================================
# 9. SALVAR PREDIÇÃO NO SUPABASE (TABELA: predicoes_dengue)
# ==============================================================================
print("Salvando predição no Supabase...")

supabase.table("predicoes_dengue").insert({
    "semana_predita": int(proxima_semana.strftime('%Y%m')),
    "data_predicao": proxima_semana.strftime('%Y-%m-%d'),
    "casos_previstos": int(predicao_proxima_semana),
    "modelo_usado": "XGBoost_v2_17features_walkforward_52s"
}).execute()

print("✅ Predição salva no Supabase!")
print("🎉 Pipeline finalizado com sucesso!")
