import pdfplumber
import spacy
import re
import nltk
from collections import defaultdict
from nltk.stem import RSLPStemmer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from math import log10
import json

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
with open('lematizadasCorrigidas.json', 'r') as file:
    lematizadasCorrigidas = json.load(file)

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
    
# ==================== MODELO VETORIAL ========================================
    
# Processar cada documento PDF e calcular TF
for i, caminho in enumerate(caminhos_dos_pdfs):
    analisar_documento(caminho, f'documento{i+1}', dados_por_documento, stopwords_encontradas, dados_estematizados)

N = len(caminhos_dos_pdfs)  # Número total de documentos

# Calcular DF para cada termo
df = defaultdict(int)
for termos in dados_por_documento.values():
    for termo in termos.keys():
        df[termo] += 1

# Calcular IDF para cada termo
idf = {termo: log10(N / df[termo]) for termo in df}

# Calcular TF-IDF para cada termo em cada documento
tf_idf = defaultdict(dict)
for doc_id, termos in dados_por_documento.items():
    for termo, freq in termos.items():
        tf_idf[doc_id][termo] = (1 + log10(freq)) * idf[termo] if freq > 0 else 0

# Processar termos da consulta
def processar_termos_consulta(termos):
    termos_processados = []
    for termo in termos:
        termo = termo.lower()
        doc = nlp(termo)
        for token in doc:
            if token.lemma_ in lematizadasCorrigidas:  # Usar correções manuais, se disponível
                termo_processado = lematizadasCorrigidas[token.lemma_]
            else:
                termo_processado = token.lemma_  # Usar lema padrão
            termos_processados.append(termo_processado)
    return termos_processados

consulta_termos = ["cantar", "formiga", "bicho"]  # Exemplo de termo de consulta
consulta_termos = processar_termos_consulta(consulta_termos)

vetor_consulta = np.array([idf[termo] for termo in consulta_termos if termo in idf])
palavras_nao_encontradas = [termo for termo in consulta_termos if termo not in idf]

# Informar ao usuário se houver termos da consulta não encontrados
if palavras_nao_encontradas:
    for termo in palavras_nao_encontradas:
        print(f"Aviso: O termo '{termo}' não foi encontrado no conjunto de dados.")

# Vetores dos documentos
vetores_documentos = {doc_id: np.array([tf_idf[doc_id].get(termo, 0) for termo in consulta_termos])
                      for doc_id in tf_idf}

# Normalizar vetores dos documentos
for doc_id, vetor in vetores_documentos.items():
    norma = np.linalg.norm(vetor)
    vetores_documentos[doc_id] = vetor / norma if norma > 0 else vetor

# Normalizar vetor da consulta
norma_consulta = np.linalg.norm(vetor_consulta)
vetor_consulta_norm = vetor_consulta / norma_consulta if norma_consulta > 0 else vetor_consulta

# Calcular a similaridade do cosseno
if vetor_consulta_norm.size > 0:
    similaridade_cosseno = {doc_id: np.dot(vetor, vetor_consulta_norm) 
                            for doc_id, vetor in vetores_documentos.items() if vetor.size == vetor_consulta_norm.size}
else:
    print("ERRO, O vetor de consulta está vazio. Verifique os termos da consulta.")

# Ordenar documentos pela similaridade
documentos_ordenados = sorted(similaridade_cosseno.items(), key=lambda x: x[1], reverse=True)

# Exibir resultados do modelo vetorial
print(f"\n ================ MODELO VETORIAL ================")
print(f"\n consulta lematizada {consulta_termos} \nDocumentos ordenados por similaridade à consulta (ranqueamento):")
print("============== RANK ==============")
for doc_id, sim in documentos_ordenados:
    print(f"{doc_id}: Similaridade = {sim:.4f}")

# parte de stemizacao para achar trechos de textos que contenham as palavras buscadas independe ou nao de lematizacao
def buscar_trecho(texto, termos):
    for termo in termos:
        # Busca por uma ocorrência do termo no texto, considerando palavras completas
        match = re.search(r'\b' + re.escape(termo) + r'\b', texto, re.IGNORECASE)
        if match:
            start = max(match.start() - 50, 0)  # Tenta pegar 50 caracteres antes
            end = min(match.end() + 50, len(texto))  # E 50 caracteres depois
            return texto[start:end]  # Retorna o trecho encontrado
    return "Trecho relevante não encontrado."  # Se nenhum termo for encontrado

textos_documentos = {}  # Inicializa o dicionário vazio

# Carrega os textos dos documentos PDF para a análise posterior
for i, caminho in enumerate(caminhos_dos_pdfs):
    doc_id = f'documento{i+1}'
    texto = extrair_texto_do_pdf(caminho)
    textos_documentos[doc_id] = texto

# Função para buscar trechos nos textos dos documentos com base na estematização
def buscar_trechos_por_estematizacao(doc_id, termos_consulta, textos_documentos, stemmer):
    texto = textos_documentos[doc_id]  # Obtém o texto do documento
    trechos_encontrados = []

    # Processa cada termo de consulta, estematizando e buscando no texto
    for termo in termos_consulta:
        termo_estematizado = stemmer.stem(termo)  # Estematiza o termo de consulta
        # Cria um padrão de regex para encontrar todas as variantes do termo estematizado
        padrao = re.compile(r'\b' + re.escape(termo_estematizado) + r'\w*', re.IGNORECASE)
        ocorrencias = padrao.findall(texto)  # Encontra todas as ocorrências no texto

        if ocorrencias:
            indice = texto.find(ocorrencias[0])  # Pega o índice da primeira ocorrência
            # Extrai um trecho do texto ao redor da ocorrência encontrada
            trecho = texto[max(0, indice-50):indice+50]
            trechos_encontrados.append((termo, trecho))  # Guarda o trecho encontrado
    
    # Imprime os trechos encontrados ou informa que não foram encontradas ocorrências
    if trechos_encontrados:
        for termo, trecho in trechos_encontrados:
            print(f"\nTrecho encontrado para '{termo}' em {doc_id}:\n\"...{trecho}...\"")
    else:
        print(f"Nenhuma ocorrência encontrada para a busca em {doc_id}.")


# Após calcular a similaridade e ordenar os documentos
for doc_id, sim in documentos_ordenados:
    if sim > 0:  # Pula documentos com similaridade igual a zero
        buscar_trechos_por_estematizacao(doc_id, consulta_termos, textos_documentos, stemmer)
