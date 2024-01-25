import pdfplumber
import spacy
import re

# Carregar modelo em português do SpaCy
nlp = spacy.load('pt_core_news_sm')

# Função para extrair texto de um arquivo PDF
def extrair_texto_do_pdf(caminho_do_arquivo):
    with pdfplumber.open(caminho_do_arquivo) as pdf:
        texto = ''
        for pagina in pdf.pages:
            texto += pagina.extract_text()
    return texto

# Função para processar o texto
def processar_texto(texto):
    # Converter para minúsculas e remover caracteres não alfabéticos
    texto = re.sub(r'[^a-záéíóúâêîôûãõç]', ' ', texto.lower())
    return texto

# Função para identificar termos úteis e stopwords
def identificar_termos_e_stopwords(texto):
    doc = nlp(texto)
    termos_uteis = {'verbos': [], 'adjetivos': [], 'substantivos': []}
    stopwords = []

    for token in doc:
        if token.is_stop:
            stopwords.append(token.text)
        elif token.pos_ == 'VERB':
            termos_uteis['verbos'].append(token.text)
        elif token.pos_ == 'ADJ':
            termos_uteis['adjetivos'].append(token.text)
        elif token.pos_ == 'NOUN':
            termos_uteis['substantivos'].append(token.text)

    return stopwords, termos_uteis

# Exemplo de uso
caminho_do_pdf = 'A_Canção_dos_tamanquinhos_Cecília_Meireles.pdf'
texto = extrair_texto_do_pdf(caminho_do_pdf)
texto_processado = processar_texto(texto)
stopwords_encontradas, termos_uteis = identificar_termos_e_stopwords(texto_processado)

print("Stopwords encontradas:", set(stopwords_encontradas))
print("Termos úteis:", termos_uteis)
