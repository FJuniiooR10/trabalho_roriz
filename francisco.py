import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.corpus import stopwords
from collections import Counter
import PyPDF2
import docx2txt
from textblob import TextBlob  # Adicionando TextBlob para análise de sentimentos
from PIL import Image  # Para usar imagens na nuvem de palavras
from io import BytesIO
import numpy as np


# Baixando recursos do NLTK
nltk.download('stopwords')
nltk.download('punkt') 

def extract_text_from_pdf(file):
    buffer = BytesIO(file.read())
    reader = PyPDF2.PdfReader(buffer)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    text = docx2txt.process(file_path)
    return text

def get_text_from_web(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('portuguese'))
    words = text.lower().split()
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return filtered_words

def generate_wordcloud(text):
    try:
        # Carregando uma imagem de máscara para a nuvem de palavras
        mask = np.array(Image.open("cloud_mask.png"))  
        wordcloud = WordCloud(width=800, height=400, background_color='white', mask=mask).generate(text)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erro ao gerar a nuvem de palavras: {e}")

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        return "positivo"
    elif sentiment_score < 0:
        return "negativo"
    else:
        return "neutro"

def main():
    text = ''
    st.title('Análise Estatística e de Sentimentos de Texto')

    input_option = st.radio('Selecione o tipo de entrada de dados:', ('PDF', 'Word', 'Link da Página', 'Texto Direto'))

    if input_option == 'PDF':
        file = st.file_uploader('Carregar PDF:', type=['pdf'])
        if file is not None:
            text = extract_text_from_pdf(file)
    elif input_option == 'Word':
        file = st.file_uploader('Carregar documento Word:', type=['docx'])
        if file is not None:
            text = extract_text_from_docx(file)
    elif input_option == 'Link da Página':
        url = st.text_input('Insira o URL da página:')
        if url:
            text = get_text_from_web(url)
    else:
        text = st.text_area('Insira o texto aqui:')

    if text:
        st.subheader('Texto Analisado:')
        st.write(text)

        # Análise de sentimentos
        sentiment = analyze_sentiment(text)
        st.subheader('Análise de Sentimentos:')
        st.write(f'O sentimento do texto é {sentiment}.')

        filtered_words = remove_stopwords(text)
        word_freq = Counter(filtered_words)
        most_common_words = word_freq.most_common(20)

        st.subheader('As 20 Palavras que mais se repetem (exceto stopwords):')
        for word, freq in most_common_words:
            st.write(f'{word}: {freq}')

        st.subheader('word cloud:')
        generate_wordcloud(' '.join(filtered_words))

if __name__ == "__main__":
    main()
