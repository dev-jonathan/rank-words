import pdfplumber
import spacy
import re
from collections import defaultdict

# Carregar modelo em português do SpaCy
nlp = spacy.load('pt_core_news_sm')

# Função para extrair texto de um arquivo PDF
def extrair_texto_do_pdf(caminho_do_arquivo):
    with pdfplumber.open(caminho_do_arquivo) as pdf:
        texto = ''
        for pagina in pdf.pages:
            texto_pagina = pagina.extract_text()
            if texto_pagina:  # Garantindo que a página tenha texto
                texto += texto_pagina + '\n'  # Adicionando quebra de linha entre as páginas
    return texto

# Função para processar o texto e extrair informações
def processar_texto(texto, doc_id, dados_por_documento):
    # Preprocessamento básico para garantir a separação correta das palavras
    texto = re.sub(r'[^a-záéíóúâêîôûãõç ]', ' ', texto.lower())
    texto = re.sub(r'\s+', ' ', texto).strip()

    doc = nlp(texto)
    for token in doc:
        if not token.is_stop and token.is_alpha:
            lemma = token.lemma_
            dados_por_documento[doc_id][lemma] += 1
            if token.text != lemma:
                print(f"Documento: {doc_id}, Palavra original: {token.text}, Lematizada: {lemma}")

# Função para analisar um documento PDF
def analisar_documento(caminho_do_arquivo, doc_id, dados_por_documento):
    texto = extrair_texto_do_pdf(caminho_do_arquivo)
    processar_texto(texto, doc_id, dados_por_documento)

# Inicializando o dicionário para armazenar dados por documento
dados_por_documento = defaultdict(lambda: defaultdict(int))

# Lista dos caminhos dos arquivos PDF
caminhos_dos_pdfs = ['A_Canção_dos_tamanquinhos_Cecília_Meireles.pdf', 'A_Centopeia_Marina_Colasanti.pdf', 'A_porta_Vinicius_de_Moraes.pdf', 
                     'Ao_pé_de_sua_criança_Pablo_Neruda.pdf', 'As_borboletas_Vinicius_de_Moraes.pdf', 'Convite_José_Paulo_Paes.pdf', 
                     'Pontinho_de_Vista_Pedro_Bandeira.pdf']

# Processar cada documento
for i, caminho in enumerate(caminhos_dos_pdfs):
    analisar_documento(caminho, f'documento{i+1}', dados_por_documento)

# Exibir dados por documento
for doc_id, termos in dados_por_documento.items():
    print(f"---- {doc_id} ----")
    for termo, freq in termos.items():
        print(f"Termo: {termo}, Frequência: {freq}")
    print("\n")
