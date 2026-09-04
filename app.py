import os
from datetime import datetime

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
        .select(
            "data_semana, casos_confirmados, "
            "temperatura_minima, umidade_maxima"
        )
        .order("data_semana", desc=False)
        .execute()
    )

    dados = pd.DataFrame(resposta.data)

    if dados.empty:
        return dados

    dados["data_semana"] = pd.to_datetime(dados["data_semana"])
    dados["casos_confirmados"] = pd.to_numeric(
        dados["casos_confirmados"],
        errors="coerce"
    )

    return dados.dropna(subset=["data_semana", "casos_confirmados"])


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

    return dados.dropna(
        subset=["data_predicao", "casos_previstos"]
    )


def formatar_data(data):
    if pd.isna(data):
        return "não informada"

    return pd.Timestamp(data).strftime("%d/%m/%Y")


def classificar_tendencia(previsao, ultimo_real):
    if ultimo_real == 0:
        return "indeterminada"

    variacao = ((previsao - ultimo_real) / ultimo_real) * 100

    if variacao <= -10:
        return "queda"
    if variacao >= 10:
        return "aumento"

    return "estabilidade"


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
ultima_previsao = predicoes.sort_values(
    ["data_predicao", "created_at", "id"]
).iloc[-1]

data_ultima_semana = ultima_semana["data_semana"]
data_previsao = ultima_previsao["data_predicao"]

casos_ultima_semana = float(ultima_semana["casos_confirmados"])
casos_previstos = float(ultima_previsao["casos_previstos"])

variacao = 0.0

if casos_ultima_semana != 0:
    variacao = (
        (casos_previstos - casos_ultima_semana)
        / casos_ultima_semana
    ) * 100

tendencia = classificar_tendencia(
    casos_previstos,
    casos_ultima_semana
)

st.markdown("---")

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
    f"Tendência estimada pelo modelo: **{tendencia}**. "
    f"A última semana disponível foi {formatar_data(data_ultima_semana)}."
)

st.markdown("## Histórico recente")

quantidade_semanas = st.slider(
    "Semanas exibidas",
    min_value=12,
    max_value=min(52, len(casos)),
    value=min(26, len(casos))
)

casos_grafico = casos.tail(quantidade_semanas).copy()

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=casos_grafico["data_semana"],
        y=casos_grafico["casos_confirmados"],
        mode="lines+markers",
        name="Casos reais",
        line=dict(color="#222222", width=3),
        marker=dict(size=7)
    )
)

predicoes_grafico = predicoes[
    predicoes["data_predicao"].isin(
        casos_grafico["data_semana"]
    )
].copy()

if not predicoes_grafico.empty:
    fig.add_trace(
        go.Scatter(
            x=predicoes_grafico["data_predicao"],
            y=predicoes_grafico["casos_previstos"],
            mode="lines+markers",
            name="Previsões registradas",
            line=dict(
                color="#d62728",
                width=2,
                dash="dash"
            ),
            marker=dict(size=6)
        )
    )

fig.add_trace(
    go.Scatter(
        x=[data_previsao],
        y=[casos_previstos],
        mode="markers",
        name="Próxima previsão",
        marker=dict(
            color="#1f77b4",
            size=14,
            symbol="diamond"
        )
    )
)

fig.update_layout(
    height=500,
    hovermode="x unified",
    xaxis_title="Semana epidemiológica",
    yaxis_title="Quantidade de casos",
    legend_title="Série",
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("## Previsões registradas")

tabela_predicoes = predicoes.sort_values(
    "data_predicao",
    ascending=False
).copy()

tabela_predicoes["data_predicao"] = (
    tabela_predicoes["data_predicao"]
    .dt.strftime("%d/%m/%Y")
)

tabela_predicoes["casos_previstos"] = (
    tabela_predicoes["casos_previstos"]
    .round(0)
    .astype("Int64")
)

st.dataframe(
    tabela_predicoes[
        [
            "data_predicao",
            "casos_previstos",
            "modelo_usado",
            "created_at"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

st.markdown("## Desempenho do modelo")

metrica1, metrica2, metrica3, metrica4 = st.columns(4)

with metrica1:
    st.metric("R²", "0,6947")

with metrica2:
    st.metric("MAE", "64,80 casos")

with metrica3:
    st.metric("RMSE", "82,01 casos")

with metrica4:
    st.metric("Features", "17")

st.caption(
    "Métricas obtidas no backtesting walk-forward das últimas "
    "52 semanas, com retreinamento semanal."
)

with st.expander("Variáveis utilizadas pelo modelo"):
    st.write(
        """
        O modelo utiliza lags de temperatura mínima, umidade máxima
        e casos confirmados, além de densidade populacional, taxa de
        coleta de resíduos, índice de Breteau, médias móveis de casos,
        variação semanal e codificação sazonal.
        """
    )

st.markdown("---")

st.caption(
    "Projeto Integrador IV — UNIVESP | "
    "Fonte epidemiológica: InfoDengue"
)

st.caption(
    f"Atualização da página: "
    f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
)
