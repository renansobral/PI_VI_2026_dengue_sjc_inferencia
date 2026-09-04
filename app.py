import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from supabase import create_client


st.set_page_config(
    page_title="Predição de Dengue — SJC",
    page_icon="🦟",
    layout="wide"
)


def obter_credencial(nome):
    valor = os.getenv(nome)

    if valor:
        return valor

    try:
        return st.secrets[nome]
    except Exception:
        return None


@st.cache_resource
def conectar_supabase():
    supabase_url = obter_credencial("SUPABASE_URL")
    supabase_key = obter_credencial("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        st.error(
            "As credenciais SUPABASE_URL e SUPABASE_KEY "
            "não foram configuradas."
        )
        st.stop()

    return create_client(supabase_url, supabase_key)


@st.cache_data(ttl=300)
def carregar_casos():
    supabase = conectar_supabase()

    resposta = (
        supabase
        .table("casos_dengue_sjc")
        .select("data_semana, casos_confirmados")
        .order("data_semana", desc=False)
        .execute()
    )

    dados = pd.DataFrame(resposta.data)

    if dados.empty:
        return dados

    dados["data_semana"] = pd.to_datetime(
        dados["data_semana"],
        errors="coerce"
    )

    dados["casos_confirmados"] = pd.to_numeric(
        dados["casos_confirmados"],
        errors="coerce"
    )

    return (
        dados
        .dropna(subset=["data_semana", "casos_confirmados"])
        .sort_values("data_semana")
        .drop_duplicates(subset=["data_semana"], keep="last")
        .reset_index(drop=True)
    )


@st.cache_data(ttl=300)
def carregar_predicoes():
    supabase = conectar_supabase()

    resposta = (
        supabase
        .table("predicoes_dengue")
        .select(
            "id, semana_predita, data_predicao, "
            "casos_previstos, modelo_usado, created_at"
        )
        .order("data_predicao", desc=False)
        .execute()
    )

    dados = pd.DataFrame(resposta.data)

    if dados.empty:
        return dados

    dados["data_predicao"] = pd.to_datetime(
        dados["data_predicao"],
        errors="coerce"
    )

    dados["created_at"] = pd.to_datetime(
        dados["created_at"],
        errors="coerce"
    )

    dados["casos_previstos"] = pd.to_numeric(
        dados["casos_previstos"],
        errors="coerce"
    )

    return (
        dados
        .dropna(
            subset=["data_predicao", "casos_previstos"]
        )
        .sort_values(
            ["data_predicao", "created_at", "id"]
        )
        .drop_duplicates(
            subset=["data_predicao"],
            keep="last"
        )
        .reset_index(drop=True)
    )


def formatar_data(data):
    if pd.isna(data):
        return "não informada"

    return pd.Timestamp(data).strftime("%d/%m/%Y")


def classificar_tendencia(previsao, ultimo_real):
    if ultimo_real == 0:
        return "indeterminada"

    variacao = (
        (previsao - ultimo_real)
        / ultimo_real
    ) * 100

    if variacao <= -10:
        return "queda"
    if variacao >= 10:
        return "aumento"

    return "estabilidade"


def calcular_metricas(dados_comparacao):
    if dados_comparacao.empty:
        return None

    y_real = dados_comparacao["casos_confirmados"].astype(float)
    y_prev = dados_comparacao["casos_previstos"].astype(float)

    mae = np.mean(np.abs(y_real - y_prev))
    rmse = np.sqrt(np.mean((y_real - y_prev) ** 2))

    ss_res = np.sum((y_real - y_prev) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)

    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "amostras": len(dados_comparacao)
    }


st.title("Motor de Inferência Preditiva para Casos de Dengue")
st.subheader("São José dos Campos — SP")

st.caption(
    "Ferramenta experimental de apoio à vigilância epidemiológica. "
    "As estimativas não substituem os boletins oficiais."
)

try:
    with st.spinner("Consultando dados epidemiológicos no Supabase..."):
        casos = carregar_casos()

    with st.spinner("Consultando previsões no Supabase..."):
        predicoes = carregar_predicoes()

except Exception as erro:
    st.error("Não foi possível consultar o Supabase.")
    st.exception(erro)
    st.stop()

if casos.empty:
    st.warning("Não foram encontrados dados epidemiológicos.")
    st.stop()

if predicoes.empty:
    st.warning("Ainda não foram encontradas previsões.")
    st.stop()

ultima_semana = casos.iloc[-1]
data_ultima_real = ultima_semana["data_semana"]
casos_ultima_semana = float(ultima_semana["casos_confirmados"])

predicoes_historicas = predicoes[
    predicoes["data_predicao"] <= data_ultima_real
].copy()

predicoes_futuras = predicoes[
    predicoes["data_predicao"] > data_ultima_real
].copy()

if predicoes_futuras.empty:
    previsao_operacional = None
else:
    previsao_operacional = (
        predicoes_futuras
        .sort_values("data_predicao")
        .iloc[0]
    )

comparacao = pd.merge(
    casos[
        ["data_semana", "casos_confirmados"]
    ],
    predicoes_historicas[
        [
            "data_predicao",
            "casos_previstos",
            "modelo_usado",
            "created_at"
        ]
    ],
    how="inner",
    left_on="data_semana",
    right_on="data_predicao"
)

if not comparacao.empty:
    comparacao["erro_absoluto"] = np.abs(
        comparacao["casos_confirmados"]
        - comparacao["casos_previstos"]
    )

metricas_historicas = calcular_metricas(comparacao)

st.markdown("---")

if previsao_operacional is not None:
    data_previsao = previsao_operacional["data_predicao"]
    casos_previstos = float(
        previsao_operacional["casos_previstos"]
    )

    variacao = (
        (
            casos_previstos
            - casos_ultima_semana
        )
        / casos_ultima_semana
    ) * 100 if casos_ultima_semana != 0 else 0.0

    tendencia = classificar_tendencia(
        casos_previstos,
        casos_ultima_semana
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Próxima semana",
            formatar_data(data_previsao)
        )

    with col2:
        st.metric(
            "Casos previstos",
            f"{casos_previstos:.0f}"
        )

    with col3:
        st.metric(
            "Última semana observada",
            f"{casos_ultima_semana:.0f}"
        )

    with col4:
        st.metric(
            "Variação estimada",
            f"{variacao:+.1f}%"
        )

    st.info(
        f"Tendência estimada: **{tendencia}**. "
        f"Última semana observada: {formatar_data(data_ultima_real)}."
    )

else:
    st.warning(
        "Não há previsão futura registrada. "
        "Execute o workflow de inferência para gerar a previsão "
        "da próxima semana."
    )

st.markdown("## Casos reais e previsões históricas")

maximo_semanas = min(104, len(casos))
valor_padrao = min(52, maximo_semanas)

quantidade_semanas = st.slider(
    "Período exibido no gráfico",
    min_value=12,
    max_value=maximo_semanas,
    value=valor_padrao,
    step=1
)

data_inicio_grafico = (
    casos
    .tail(quantidade_semanas)["data_semana"]
    .min()
)

casos_grafico = casos[
    casos["data_semana"] >= data_inicio_grafico
].copy()

predicoes_historicas_grafico = predicoes_historicas[
    predicoes_historicas["data_predicao"] >= data_inicio_grafico
].copy()

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=casos_grafico["data_semana"],
        y=casos_grafico["casos_confirmados"],
        mode="lines+markers",
        name="Casos reais",
        line=dict(color="#1f1f1f", width=3),
        marker=dict(size=6)
    )
)

if not predicoes_historicas_grafico.empty:
    fig.add_trace(
        go.Scatter(
            x=predicoes_historicas_grafico["data_predicao"],
            y=predicoes_historicas_grafico["casos_previstos"],
            mode="lines+markers",
            name="Previsões históricas",
            line=dict(
                color="#d62728",
                width=2,
                dash="dash"
            ),
            marker=dict(size=6)
        )
    )

if previsao_operacional is not None:
    fig.add_trace(
        go.Scatter(
            x=[previsao_operacional["data_predicao"]],
            y=[previsao_operacional["casos_previstos"]],
            mode="markers",
            name="Próxima previsão",
            marker=dict(
                color="#1f77b4",
                size=15,
                symbol="diamond"
            )
        )
    )

fig.update_layout(
    height=520,
    hovermode="x unified",
    xaxis_title="Semana epidemiológica",
    yaxis_title="Quantidade de casos",
    legend_title="Séries",
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("## Avaliação das previsões históricas")

if metricas_historicas is None:
    st.warning(
        "Ainda não há semanas com previsão e caso real correspondentes."
    )
else:
    metrica1, metrica2, metrica3, metrica4 = st.columns(4)

    with metrica1:
        st.metric("R² histórico", f"{metricas_historicas['r2']:.4f}")

    with metrica2:
        st.metric(
            "MAE histórico",
            f"{metricas_historicas['mae']:.2f} casos"
        )

    with metrica3:
        st.metric(
            "RMSE histórico",
            f"{metricas_historicas['rmse']:.2f} casos"
        )

    with metrica4:
        st.metric(
            "Semanas comparadas",
            metricas_historicas["amostras"]
        )

    st.caption(
        "Métricas calculadas a partir das previsões históricas "
        "simuladas e dos respectivos casos posteriormente observados."
    )

st.markdown("## Tabela de acompanhamento")

if comparacao.empty:
    st.info(
        "Não há previsões históricas para comparar com os casos reais."
    )
else:
    tabela = comparacao[
        [
            "data_semana",
            "casos_confirmados",
            "casos_previstos",
            "erro_absoluto",
            "modelo_usado"
        ]
    ].copy()

    tabela = tabela.rename(
        columns={
            "data_semana": "Data da semana",
            "casos_confirmados": "Casos reais",
            "casos_previstos": "Casos previstos",
            "erro_absoluto": "Erro absoluto",
            "modelo_usado": "Modelo"
        }
    )

    tabela["Data da semana"] = (
        tabela["Data da semana"]
        .dt.strftime("%d/%m/%Y")
    )

    for coluna in [
        "Casos reais",
        "Casos previstos",
        "Erro absoluto"
    ]:
        tabela[coluna] = tabela[coluna].round(0).astype("Int64")

    tabela = tabela.sort_values(
        "Data da semana",
        ascending=False
    )

    st.dataframe(
        tabela,
        use_container_width=True,
        hide_index=True
    )

with st.expander("Metodologia e variáveis do modelo"):
    st.write(
        """
        As previsões históricas foram produzidas por simulação
        walk-forward. Para cada semana prevista, o modelo XGBoost
        foi retreinado utilizando somente os registros anteriores
        à semana-alvo. Foram utilizadas 17 variáveis: lags de casos,
        temperatura mínima e umidade máxima; densidade populacional;
        taxa de coleta de resíduos; índice de Breteau; médias móveis;
        variação semanal de casos; e componentes de sazonalidade.
        """
    )

st.markdown("---")

st.caption(
    "Projeto Integrador IV — UNIVESP | "
    "Fonte epidemiológica: InfoDengue"
)

st.caption(
    f"Página consultada em: "
    f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
)
