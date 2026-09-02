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
# 4. ENGENHARIA DE RECURSOS (LAG FEATURES)
# ==============================================================================
for lag in [1, 2, 3]:
    df[f"casos_lag_{lag}"] = df["casos_confirmados"].shift(lag)
    df[f"temp_min_lag_{lag}"] = df["temperatura_minima"].shift(lag)
    df[f"umidade_lag_{lag}"] = df["umidade_maxima"].shift(lag)

# ==============================================================================
# 4.1 FEATURES EXPERIMENTAIS: TENDÊNCIA E SAZONALIDADE
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

# Remover NaNs (primeiras semanas sem lags e sem médias móveis)
df = df.dropna().reset_index(drop=True)

# ==============================================================================
# 5. PREPARAR FEATURES PARA TREINAMENTO (17 VARIÁVEIS)
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

# Recorte D-60 (esconder últimos 60 dias)
data_limite = df['data_semana'].max() - pd.Timedelta(days=60)
df_treino = df[df['data_semana'] <= data_limite]

X = df_treino[features]
y = df_treino['casos_confirmados']

print(f"IA treinada com dados até: {df_treino['data_semana'].max().strftime('%d/%m/%Y')}")

# ==============================================================================
# 6. TREINAR MODELO XGBOOST (HIPERPARÂMETROS VENCEDORES)
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
# 7. BACKTESTING TEMPORAL (AVALIAÇÃO EM DADOS FUTUROS JÁ CONHECIDOS)
# ==============================================================================
print("Iniciando backtesting temporal...")

df_teste = df[df["data_semana"] > data_limite].copy()

if df_teste.empty:
    raise RuntimeError(
        "Não existem semanas posteriores ao corte D-60 para avaliar o backtest."
    )

X_teste = df_teste[features]
y_teste = df_teste["casos_confirmados"]

predicoes_backtest = modelo_producao.predict(X_teste)
predicoes_backtest = np.maximum(predicoes_backtest, 0)

mae_backtest = np.mean(np.abs(y_teste - predicoes_backtest))
rmse_backtest = np.sqrt(np.mean((y_teste - predicoes_backtest) ** 2))

ss_res = np.sum((y_teste - predicoes_backtest) ** 2)
ss_tot = np.sum((y_teste - np.mean(y_teste)) ** 2)
r2_backtest = 1 - (ss_res / ss_tot)

print("=" * 65)
print("RESULTADO DO BACKTESTING TEMPORAL — XGBOOST (17 FEATURES)")
print("=" * 65)
print(f"Período de teste: {df_teste['data_semana'].min().strftime('%d/%m/%Y')} "
      f"até {df_teste['data_semana'].max().strftime('%d/%m/%Y')}")
print(f"Semanas avaliadas: {len(df_teste)}")
print(f"R²:   {r2_backtest:.4f}")
print(f"MAE:  {mae_backtest:.2f} casos")
print(f"RMSE: {rmse_backtest:.2f} casos")
print("=" * 65)

resultado_backtest = pd.DataFrame({
    "data_semana": df_teste["data_semana"].dt.strftime("%Y-%m-%d"),
    "casos_reais": y_teste.values,
    "casos_previstos": np.round(predicoes_backtest, 2),
    "erro_absoluto": np.round(np.abs(y_teste.values - predicoes_backtest), 2)
})

print(resultado_backtest.to_string(index=False))

# ==============================================================================
# 7.1 BACKTEST RECURSIVO DE 2 SEMANAS (17 FEATURES)
# ==============================================================================
print("Iniciando backtest recursivo de duas semanas...")

df_teste_recursivo = df[df["data_semana"] > data_limite].copy().reset_index(drop=True)

historico = df_treino.copy()
resultados_recursivos = []

for i in range(min(2, len(df_teste_recursivo))):
    linha_futura = df_teste_recursivo.iloc[i]

    # Semana do ano para a data futura ( Timestamp direto, sem .dt )
    semana_ano_futura = int(linha_futura["data_semana"].isocalendar().week)

    X_futuro = pd.DataFrame({
        "temp_min_lag_1": [historico["temperatura_minima"].iloc[-1]],
        "umidade_lag_1": [historico["umidade_maxima"].iloc[-1]],
        "temp_min_lag_2": [historico["temperatura_minima"].iloc[-2]],
        "umidade_lag_2": [historico["umidade_maxima"].iloc[-2]],
        "temp_min_lag_3": [historico["temperatura_minima"].iloc[-3]],
        "umidade_lag_3": [historico["umidade_maxima"].iloc[-3]],
        "densidade_populacional": [linha_futura["densidade_populacional"]],
        "taxa_coleta_residuos": [linha_futura["taxa_coleta_residuos"]],
        "indice_breteu_adl": [linha_futura["indice_breteu_adl"]],
        "casos_lag_1": [historico["casos_confirmados"].iloc[-1]],
        "casos_lag_2": [historico["casos_confirmados"].iloc[-2]],
        "casos_lag_3": [historico["casos_confirmados"].iloc[-3]],
        "casos_media_2s": [historico["casos_confirmados"].shift(1).iloc[-2:].mean()],
        "casos_media_4s": [historico["casos_confirmados"].shift(1).iloc[-4:].mean()],
        "variacao_casos_1s": [
            historico["casos_confirmados"].iloc[-2] - historico["casos_confirmados"].iloc[-3]
        ],
        "semana_sin": [np.sin(2 * np.pi * semana_ano_futura / 52)],
        "semana_cos": [np.cos(2 * np.pi * semana_ano_futura / 52)]
    })[features]

    previsao = float(modelo_producao.predict(X_futuro)[0])
    previsao = max(0, previsao)

    resultados_recursivos.append({
        "horizonte": i + 1,
        "data_semana": linha_futura["data_semana"].strftime("%Y-%m-%d"),
        "casos_reais": int(linha_futura["casos_confirmados"]),
        "casos_previstos": round(previsao, 2),
        "erro_absoluto": round(abs(linha_futura["casos_confirmados"] - previsao), 2)
    })

    # Para a segunda semana, a previsão substitui o caso real no histórico.
    nova_linha = linha_futura.to_frame().T
    nova_linha["casos_confirmados"] = previsao
    historico = pd.concat([historico, nova_linha], ignore_index=True)

df_backtest_recursivo = pd.DataFrame(resultados_recursivos)

print("=" * 65)
print("BACKTEST RECURSIVO — HORIZONTE DE DUAS SEMANAS (17 FEATURES)")
print("=" * 65)
print(df_backtest_recursivo.to_string(index=False))
print("=" * 65)

# ==============================================================================
# 8. GERAR PREDIÇÃO PARA PRÓXIMAS 2 SEMANAS (17 FEATURES)
# ==============================================================================
print("Gerando predições...")

# Pegar última semana disponível
ultima_semana = df.iloc[-1]

# Semana do ano da última semana (Timestamp direto, sem .dt)
semana_ano_ultima = int(ultima_semana["data_semana"].isocalendar().week)

# Preparar features para predição
X_future = pd.DataFrame({
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
    "semana_sin": [np.sin(2 * np.pi * semana_ano_ultima / 52)],
    "semana_cos": [np.cos(2 * np.pi * semana_ano_ultima / 52)]
})

# Garante, explicitamente, a mesma ordem usada no treinamento.
X_future = X_future[features]
predicao_semana_1 = modelo_producao.predict(X_future)[0]

# Semana 2: desloca a janela autorregressiva de casos e atualiza tendência.
X_future_2 = X_future.copy()

# Atualiza lags de casos
X_future_2["casos_lag_1"] = predicao_semana_1
X_future_2["casos_lag_2"] = df["casos_confirmados"].iloc[-1]
X_future_2["casos_lag_3"] = df["casos_confirmados"].iloc[-2]

# Atualiza médias móveis de casos (tendência)
casos_com_shift = df["casos_confirmados"].shift(1)
X_future_2["casos_media_2s"] = (
    pd.Series([casos_com_shift.iloc[-2], casos_com_shift.iloc[-1], predicao_semana_1])
    .iloc[-2:]
    .mean()
)

X_future_2["casos_media_4s"] = (
    pd.Series([
        casos_com_shift.iloc[-4],
        casos_com_shift.iloc[-3],
        casos_com_shift.iloc[-2],
        casos_com_shift.iloc[-1],
        predicao_semana_1
    ])
    .iloc[-4:]
    .mean()
)

# Atualiza variação de casos
X_future_2["variacao_casos_1s"] = (
    predicao_semana_1 - df["casos_confirmados"].iloc[-2]
)

# Atualiza sazonalidade para a semana seguinte
semana_ano_2 = int((ultima_semana["data_semana"] + pd.Timedelta(days=7)).isocalendar().week)
X_future_2["semana_sin"] = [np.sin(2 * np.pi * semana_ano_2 / 52)]
X_future_2["semana_cos"] = [np.cos(2 * np.pi * semana_ano_2 / 52)]

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
    "modelo_usado": "XGBoost_v2_17features_github_actions"
}).execute()

supabase.table("predicoes_dengue").insert({
    "semana_predita": int(data_semana_2.strftime('%Y%m')),
    "data_predicao": data_semana_2.strftime('%Y-%m-%d'),
    "casos_previstos": int(predicao_semana_2),
    "modelo_usado": "XGBoost_v2_17features_github_actions"
}).execute()

print("✅ Predições salvas no Supabase!")
print("🎉 Pipeline finalizado com sucesso!")
