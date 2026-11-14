"""
Detector de Spam em E-mails - Streamlit + TensorFlow
----------------------------------------------------
Aplicacao de arquivo unico.

Como funciona (visao geral):
1. Um conjunto de exemplos rotulados (spam / nao spam) e carregado.
2. Um modelo TensorFlow e treinado com esses exemplos (1x por sessao do servidor).
3. O usuario cola uma mensagem, e o modelo devolve a probabilidade de ser spam.

Executar localmente:  streamlit run app.py
"""

import os

# Reduz logs verbosos do TensorFlow (precisa vir ANTES do import do tensorflow).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------------------------------------------------------
# Configuracoes gerais
# --------------------------------------------------------------------------
SEED = 42                 # garante resultados reproduziveis
MAX_VOCAB = 5000          # tamanho maximo do vocabulario
SEQ_LEN = 60              # numero maximo de palavras consideradas por mensagem
EPOCHS = 40               # passadas completas pelos dados de treino
LIMITE_PADRAO = 0.50      # probabilidade a partir da qual a mensagem e spam

# --------------------------------------------------------------------------
# Base de exemplo (demonstracao). Para uso real, envie um CSV maior na
# barra lateral: quanto mais exemplos reais da empresa, melhor o modelo.
# --------------------------------------------------------------------------
SPAM_EXEMPLOS = [
    "Parabens! Voce ganhou um premio de R$ 10.000. Clique aqui para resgatar agora",
    "URGENTE: sua conta sera bloqueada. Confirme seus dados neste link imediatamente",
    "Ganhe dinheiro rapido trabalhando de casa, sem experiencia. Cadastre-se ja",
    "Oferta imperdivel! 90% de desconto somente hoje, compre agora e aproveite",
    "Voce foi selecionado para receber um cartao de credito sem consulta ao SPC",
    "Emprestimo aprovado sem burocracia, dinheiro na conta em 5 minutos",
    "Clique no link para receber seu bonus gratis de ate R$ 500 sem custo",
    "Remedios sem receita com preco baixo, entrega discreta em todo o Brasil",
    "Seu pacote esta retido. Pague a taxa de R$ 9,90 neste link para liberar",
    "Ultima chance: sorteio de um iPhone gratis, participe agora mesmo",
    "Aumente seus seguidores e curtidas em minutos, pacote promocional exclusivo",
    "Investimento com lucro garantido de 300% ao mes, vagas limitadas",
    "Atencao: atividade suspeita na sua conta bancaria, acesse o link e atualize a senha",
    "Voce e o visitante numero 1000000! Resgate seu presente exclusivo agora",
    "Dinheiro facil e sem risco, ganhe renda extra hoje, responda este e-mail",
    "Comprar seguidores, viagra e cassino online, bonus de boas vindas gratis",
    "Congratulations! You have won a free gift card. Click here to claim now",
    "URGENT: verify your account now or it will be suspended permanently",
    "Make money fast from home, no experience needed, limited offer act now",
    "Free entry in our weekly prize draw, text WIN to 80085 to receive your reward",
    "You have been pre-approved for a loan with no credit check, apply today",
    "Cheap meds online no prescription needed, best price guaranteed",
    "Exclusive deal just for you, 90% off, buy now before it expires tonight",
    "Your package could not be delivered, pay the fee at this link to reschedule",
    "Earn 5000 dollars per week guaranteed, click the link to join our program",
    "Claim your free vacation to the Bahamas, winner selected today only",
    "Dear friend, I am a prince and need your help to transfer funds, you will be rewarded",
    "Hot singles in your area want to meet you, click here for free access",
]

HAM_EXEMPLOS = [
    "Bom dia, segue em anexo a ata da reuniao de ontem para sua revisao",
    "Podemos remarcar nossa call de alinhamento para quinta-feira as 15h?",
    "Oi Carlos, preciso do relatorio de vendas do mes ate sexta-feira",
    "Lembrete: a campanha de e-mail marketing sera enviada amanha as 9h",
    "Obrigado pelo retorno. Vou revisar o contrato e te respondo ate amanha",
    "Segue a proposta comercial conforme conversamos por telefone",
    "A reuniao de planejamento trimestral sera na sala 3, as 10h",
    "Pode me enviar as metricas de abertura e cliques da ultima campanha?",
    "Ola equipe, o briefing do novo cliente ja esta disponivel na pasta compartilhada",
    "Confirmo minha presenca no evento de sexta-feira. Ate la",
    "Preciso de ajuda com o acesso ao sistema de CRM, poderia verificar?",
    "Parabens pelo excelente resultado do time neste trimestre",
    "Estou saindo para o almoco e volto as 13h30, qualquer coisa me ligue",
    "Anexei a nota fiscal referente ao servico prestado em outubro",
    "Vamos revisar o cronograma do projeto na proxima segunda-feira?",
    "O fornecedor confirmou a entrega do material grafico para quarta-feira",
    "Poderia aprovar o orcamento do evento? Deixei o documento no drive",
    "Bom dia, gostaria de agendar uma demonstracao do produto para nossa equipe",
    "Hi John, please find attached the agenda for tomorrow's meeting",
    "Can we reschedule our one on one to Thursday afternoon?",
    "Thanks for sending the report, I will review it and get back to you today",
    "Reminder: the quarterly review is scheduled for Monday at 10 am in room 4",
    "Could you share the latest campaign metrics before the end of the week?",
    "I have attached the invoice for the services delivered last month",
    "Great job on the presentation, the client was very happy with the results",
    "Let's catch up tomorrow about the new project timeline and deliverables",
    "The design team uploaded the final banner files to the shared folder",
    "Please confirm your attendance at the team workshop next Friday",
]


# --------------------------------------------------------------------------
# Funcoes de dados
# --------------------------------------------------------------------------
def carregar_base_padrao():
    """Monta a base embutida: lista de textos e lista de rotulos (1=spam, 0=nao spam)."""
    textos = SPAM_EXEMPLOS + HAM_EXEMPLOS
    rotulos = [1] * len(SPAM_EXEMPLOS) + [0] * len(HAM_EXEMPLOS)
    return textos, rotulos


def carregar_base_csv(arquivo):
    """
    Le um CSV enviado pelo usuario. Formato esperado: colunas 'label' e 'message'.
    Valores aceitos em label: spam/ham, 1/0.
    Retorna (textos, rotulos). Lanca ValueError se o formato estiver incorreto.
    """
    try:
        df = pd.read_csv(arquivo, sep=None, engine="python", encoding="utf-8")
    except UnicodeDecodeError:
        arquivo.seek(0)  # volta ao inicio do arquivo e tenta outra codificacao
        df = pd.read_csv(arquivo, sep=None, engine="python", encoding="latin-1")

    df.columns = [c.strip().lower() for c in df.columns]
    if not {"label", "message"}.issubset(df.columns):
        raise ValueError("O CSV precisa ter as colunas 'label' e 'message'.")

    mapa = {"spam": 1, "1": 1, "ham": 0, "0": 0}
    rotulos = df["label"].astype(str).str.strip().str.lower().map(mapa)
    df = df.assign(label_num=rotulos).dropna(subset=["label_num", "message"])

    if df["label_num"].nunique() < 2 or len(df) < 20:
        raise ValueError("Envie ao menos 20 linhas contendo spam e nao spam.")

    return df["message"].astype(str).tolist(), df["label_num"].astype(int).tolist()


# --------------------------------------------------------------------------
# Modelo TensorFlow
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Treinando o modelo, aguarde...")
def treinar_modelo(textos: tuple, rotulos: tuple):
    """
    Treina o classificador e devolve (modelo, acuracia_de_validacao).
    O cache evita retreinar a cada interacao: so treina de novo se a base mudar.
    """
    keras.utils.set_random_seed(SEED)

    x = np.array(textos, dtype=object)
    y = np.array(rotulos, dtype="float32")

    # Embaralha e separa 80% treino / 20% validacao
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x))
    corte = int(len(x) * 0.8)
    x_tr, y_tr = x[idx[:corte]], y[idx[:corte]]
    x_va, y_va = x[idx[corte:]], y[idx[corte:]]

    # Camada que converte texto em numeros (cada palavra vira um indice)
    vetorizador = layers.TextVectorization(
        max_tokens=MAX_VOCAB,
        output_sequence_length=SEQ_LEN,
        standardize="lower_and_strip_punctuation",
    )
    vetorizador.adapt(x_tr)  # aprende o vocabulario apenas com dados de treino

    modelo = keras.Sequential(
        [
            keras.Input(shape=(), dtype=tf.string),   # entrada: uma string
            vetorizador,                              # texto -> sequencia de inteiros
            layers.Embedding(vetorizador.vocabulary_size(), 16),  # inteiro -> vetor
            layers.GlobalAveragePooling1D(),          # resume a mensagem em 1 vetor
            layers.Dropout(0.3),                      # evita decorar os exemplos
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),    # saida: probabilidade de spam
        ]
    )
    modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    modelo.fit(x_tr, y_tr, epochs=EPOCHS, batch_size=16, verbose=0)

    _, acuracia = modelo.evaluate(x_va, y_va, verbose=0)
    return modelo, float(acuracia)


def prever(modelo, mensagem: str) -> float:
    """Devolve a probabilidade (0 a 1) de a mensagem ser spam."""
    entrada = np.array([mensagem], dtype=object)
    return float(modelo.predict(entrada, verbose=0)[0][0])


# --------------------------------------------------------------------------
# Interface Streamlit
# --------------------------------------------------------------------------
st.set_page_config(page_title="Detector de Spam", page_icon=None, layout="centered")

st.title("Detector de Spam em E-mails")
st.caption("Ferramenta interna de apoio as campanhas de marketing por e-mail.")

# ---- Barra lateral: base de treino e ajustes ----
with st.sidebar:
    st.header("Configuracoes")
    arquivo = st.file_uploader(
        "Base de treino (CSV opcional)",
        type=["csv"],
        help="Colunas obrigatorias: label (spam/ham) e message.",
    )
    limite = st.slider(
        "Sensibilidade (limite de spam)",
        min_value=0.10, max_value=0.90, value=LIMITE_PADRAO, step=0.05,
        help="Valores menores marcam mais mensagens como spam.",
    )

# ---- Carrega a base (CSV enviado ou base embutida) ----
try:
    if arquivo is not None:
        textos, rotulos = carregar_base_csv(arquivo)
        origem = f"CSV enviado ({len(textos)} mensagens)"
    else:
        textos, rotulos = carregar_base_padrao()
        origem = f"base de demonstracao ({len(textos)} mensagens)"
except ValueError as erro:
    st.sidebar.error(str(erro))
    st.stop()  # interrompe a execucao do script ate o usuario corrigir

modelo, acuracia = treinar_modelo(tuple(textos), tuple(rotulos))

with st.sidebar:
    st.metric("Acuracia de validacao", f"{acuracia:.0%}")
    st.caption(f"Treinado com: {origem}")
    if arquivo is None:
        st.info("A base de demonstracao e pequena. Para resultados confiaveis, "
                "envie um CSV com e-mails reais da empresa.")

# ---- Area principal: analise da mensagem ----
mensagem = st.text_area(
    "Mensagem do e-mail",
    height=220,
    placeholder="Cole aqui o assunto e o corpo do e-mail para analise...",
)

if st.button("Analisar mensagem", type="primary"):
    if not mensagem.strip():
        st.warning("Informe o conteudo da mensagem para realizar a analise.")
    else:
        prob = prever(modelo, mensagem)
        eh_spam = prob >= limite

        if eh_spam:
            st.error("Resultado: SPAM")
        else:
            st.success("Resultado: NAO E SPAM")

        col1, col2 = st.columns(2)
        col1.metric("Probabilidade de spam", f"{prob:.1%}")
        col2.metric("Limite configurado", f"{limite:.0%}")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption("O resultado e uma estimativa estatistica e deve apoiar, "
                   "nao substituir, a revisao humana.")
