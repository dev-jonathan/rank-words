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
from rank_bm25 import BM25Okapi

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
    
# ==================== parte do MODELO VETORIAL ========================
    
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

## ====================== parte do modelo probabilístico BM25 ===============================
        
print(f"\n ================ MODELO PROBABILÍSTICO BM25 ================")

# Consultas de exemplo
consultas = [
    ["borboletas", "formiga", "bicho"],  # Aparece em 2 documentos "borboleta"
    ["pé", "porta"],  # Aparece em 2 documentos tanto "pé" como "porta"
    ["manca", "papagaio"]
]

# Preprocessa os textos dos PDFs para extrair e limpar o texto
tamanhos_dos_textos = []
for caminho in caminhos_dos_pdfs:
    texto = extrair_texto_do_pdf(caminho)  # Extrai o texto do PDF
    # Remove caracteres especiais e converte para minúsculas
    texto_preprocessado = re.sub(r'[^a-záéíóúâêîôûãõç ]', ' ', texto.lower())
    texto_preprocessado = re.sub(r'\s+', ' ', texto_preprocessado).strip()
    tamanhos_dos_textos.append(len(texto_preprocessado.split()))

# Calcula a média dos tamanhos dos textos para análise posterior, se necessário
media_tamanhos = sum(tamanhos_dos_textos) / len(tamanhos_dos_textos)

# Prepara o corpus para o BM25, processando cada texto para o singular
corpus = []
for caminho in caminhos_dos_pdfs:
    texto = extrair_texto_do_pdf(caminho)  # Repete a extração e limpeza para cada PDF
    texto_preprocessado = re.sub(r'[^a-záéíóúâêîôûãõç ]', ' ', texto.lower()).strip()
    corpus.append(texto_preprocessado.split())

# Função para ajustar palavras para o singular, removendo 's' no final
def ajustar_para_singular(palavras):
    # Remove 's' no final das palavras, assumindo que são plurais
    return [palavra[:-1] if palavra.endswith('s') else palavra for palavra in palavras]

# Ajusta o corpus inteiro para singular antes de inicializar o BM25
corpus_singular = [ajustar_para_singular(texto) for texto in corpus]

# Inicializa o BM25 com o corpus ajustado para o singular
bm25 = BM25Okapi(corpus_singular, k1=2, b=0.75)

# Define uma função para buscar trechos relevantes nos textos
def buscar_trecho(texto, termos):
    trechos_encontrados = []
    for termo in termos:
        # Considera termos no singular e no plural na busca
        termo_singular = termo[:-1] if termo.endswith('s') else termo
        termo_plural = termo + 's' if not termo.endswith('s') else termo
        
        padrao = re.compile(r'\b({}|{})\b'.format(termo_singular, termo_plural), re.IGNORECASE)
        
        # Encontra todas as ocorrências do termo no texto
        matches = padrao.findall(texto)
        if matches:
            inicio = texto.lower().find(matches[0].lower())
            inicio = max(inicio - 30, 0)
            fim = min(inicio + 60, len(texto))
            trecho = texto[inicio:fim]
            trechos_encontrados.append(trecho + "...")
    
    # Retorna os trechos encontrados ou uma mensagem padrão
    return " ... ".join(trechos_encontrados) if trechos_encontrados else "Trecho relevante não encontrado."

# Processa cada consulta, ajustando para singular, calculando scores e exibindo os resultados
for consulta in consultas:
    consulta_singular = ajustar_para_singular(consulta)
    print(f"\nConsulta: [{', '.join(consulta_singular)}]")
    scores = bm25.get_scores(consulta_singular)

    # Filtra e ordena os documentos com base nos scores
    scores_e_ids = [(doc_id, score) for doc_id, score in enumerate(scores) if score > 0]
    scores_e_ids.sort(key=lambda x: x[1], reverse=True)

    # Exibe o ranqueamento dos documentos
    ranqueamento_str = ', '.join([f"doc{doc_id+1}: {score:.4f}" for doc_id, score in scores_e_ids])
    print(f"|RANK|: {ranqueamento_str}")

    # Exibe trechos relevantes para cada documento ranqueado
    for doc_id, score in scores_e_ids:
        trecho = buscar_trecho(textos_documentos[f'documento{doc_id+1}'], consulta_singular)
        print(f"Trecho de doc{doc_id+1}: \"...{trecho}\"\n")