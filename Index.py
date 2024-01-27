import pdfplumber
import spacy
import re
import nltk
from collections import defaultdict
from nltk.stem import RSLPStemmer

# Baixar dados do stemmer RSLP para português e carregar modelo SpaCy
nltk.download('rslp')
nlp = spacy.load('pt_core_news_sm')
stemmer = RSLPStemmer()

# Definir stopwords personalizadas e atualizar no SpaCy
stopwords = ["troc", "oh", "visse", "ir", "vão", "vamos", "ter", "ficam", "fico", "ia", "ser", "será", "há", "cem", "fazer", "feita", "haver", "pra", "saber", "querer", "poder", "algum"]
nlp.Defaults.stop_words.update(set(stopwords))

# Função para estematizar uma palavra (reduzi-la à sua raiz)
def estematizar_palavra(palavra):
    return stemmer.stem(palavra)

# Correções manuais para lematizações
lematizadasCorrigidas = {
    "ligeirinhos": "ligeiro",
    "tamanquinhos": "tamanco",
    "chove": "chover",
    "caminhos": "caminhar",
    "manca": "mancar",
    "madeira": "madeira",
    "abro": "abrir",
    "devagarinho": "devagar",
    "menininho": "menino",
    "prazenteira": "prazenteiro",
    "cozinheira": "cozinheiro",
    "fecho": "fechar",
    "redonda": "redondar",
    "brancas": "branco",
    "amarelas": "amarelo",
    "borboletas": "borboleta",
    "alegres": "alegre",
    "amarelinhas": "amarelo",
    "bonitinhas": "bonito",
    "pretas": "preto",
    "fico": "ficar",
    "formiga": "formiga",
    "falasse": "falar",
    "grandão": "grande"
}

# Lista dos caminhos dos arquivos PDF
caminhos_dos_pdfs = ['A_Canção_dos_tamanquinhos_Cecília_Meireles.pdf', 'A_Centopeia_Marina_Colasanti.pdf', 'A_porta_Vinicius_de_Moraes.pdf', 
                     'Ao_pé_de_sua_criança_Pablo_Neruda.pdf', 'As_borboletas_Vinicius_de_Moraes.pdf', 'Convite_José_Paulo_Paes.pdf', 
                     'Pontinho_de_Vista_Pedro_Bandeira.pdf']

# Função para extrair texto de um arquivo PDF
def extrair_texto_do_pdf(caminho_do_arquivo):
    with pdfplumber.open(caminho_do_arquivo) as pdf:
        texto = ''
        for pagina in pdf.pages:
            texto_pagina = pagina.extract_text()
            if texto_pagina:
                texto += texto_pagina + '\n'
    return texto
    
# Função para processar texto, lematizar e estematizar palavras
def processar_texto(texto, doc_id, dados_por_documento, stopwords_encontradas, dados_estematizados):
    texto = re.sub(r'[^a-záéíóúâêîôûãõç ]', ' ', texto.lower())
    texto = re.sub(r'\s+', ' ', texto).strip()

    doc = nlp(texto)
    for token in doc:
        # Estematização da palavra
        stem = estematizar_palavra(token.text)

        # Verificar se a palavra está no dicionário de lematizações corrigidas
        if token.text in lematizadasCorrigidas:
            lemma = lematizadasCorrigidas[token.text]
        else:
            lemma = token.lemma_

        # Processar tokens que não são stopwords
        if token.is_alpha and (not token.is_stop and token.text not in stopwords):
            dados_por_documento[doc_id][lemma] += 1
            dados_estematizados[doc_id][stem] += 1

            # Opcional: Remover a impressão para reduzir a saída do console
            if token.text != lemma:
                print(f"Documento: {doc_id}, Palavra original: {token.text}, Lematizada: {lemma}, Estematizada: {stem}")

        # Adicionar à lista de stopwords encontradas
        elif token.is_stop or token.text in stopwords:
            stopwords_encontradas.add(token.text)

# Função para analisar um documento PDF
def analisar_documento(caminho_do_arquivo, doc_id, dados_por_documento, stopwords_encontradas, dados_estematizados):
    texto = extrair_texto_do_pdf(caminho_do_arquivo)
    processar_texto(texto, doc_id, dados_por_documento, stopwords_encontradas, dados_estematizados)

# Inicializando dicionários para armazenar dados processados
dados_por_documento = defaultdict(lambda: defaultdict(int))
stopwords_encontradas = set()
dados_estematizados = defaultdict(lambda: defaultdict(int))

# Processar cada documento PDF
for i, caminho in enumerate(caminhos_dos_pdfs):
    analisar_documento(caminho, f'documento{i+1}', dados_por_documento, stopwords_encontradas, dados_estematizados)

# Exibir stopwords encontradas
print(f"\nStopwords encontradas: {stopwords_encontradas} \n")

# Calcular frequências totais e detalhadas por documento
frequencias_totais = defaultdict(int)
for doc_id, termos in dados_por_documento.items():
    for termo, freq in termos.items():
        frequencias_totais[termo] += freq

# Exibir frequências no formato Termo/frequência_no_Corpus -> código_documento/aparições_no_doc
for termo, freq_total in frequencias_totais.items():
    detalhes_por_documento = ' '.join([f"{doc_id}/{dados_por_documento[doc_id][termo]}" for doc_id in dados_por_documento if termo in dados_por_documento[doc_id]])
    print(f"{termo}/{freq_total} -> {detalhes_por_documento}")


#============ parte de estimizar com print seguindo Termo/frequência_no_Corpus -> código_documento/aparições_no_doc ==========
    
# Inicializando dicionário para armazenar dados estematizados
# dados_estematizados = defaultdict(lambda: defaultdict(int))

# Processar cada documento

# for i, caminho in enumerate(caminhos_dos_pdfs):
#     analisar_documento(caminho, f'documento{i+1}', dados_por_documento, stopwords_encontradas, dados_estematizados)

# Função para imprimir as frequências das palavras estematizadas
# def imprimir_frequencias_estematizadas(dados_estematizados):
#     frequencias_totais_estematizadas = defaultdict(int)
#     for doc_id, termos in dados_estematizados.items():
#         for termo, freq in termos.items():
#             frequencias_totais_estematizadas[termo] += freq

#     for termo, freq_total in frequencias_totais_estematizadas.items():
#         detalhes_por_documento = ' '.join([f"{doc_id}/{dados_estematizados[doc_id][termo]}" for doc_id in dados_estematizados if termo in dados_estematizados[doc_id]])
#         print(f"{termo}/{freq_total} -> {detalhes_por_documento}")

# Chamada à função de impressão das frequências estematizadas
## print("\nFrequências com Estematizações:")
## imprimir_frequencias_estematizadas(dados_estematizados)