# File containing all the analysis functions for the web app

# Standard Libraries
import os 
import re 
import string 
import numpy as np
from collections import Counter

# Text Processing Library 
from wordcloud import WordCloud
from gensim import utils
import streamlit as st
import pprint
import gensim
import gensim.downloader as api
import warnings
import spacy
from spacy import displacy
from pathlib import Path
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span
import tempfile
warnings.filterwarnings(action='ignore')
from pathlib import Path

# Data Visualisation 
import matplotlib.pyplot as plt 
import seaborn as sns
import spacy_streamlit
from PIL import Image


# Create a word cloud function 
def create_wordcloud(text, image_path = None):
    '''
    Pass a string to the function and output a word cloud
    
    ARGS 
    text: The text for wordcloud
    image_path (optional): The image mask with a white background (default None)
    
    '''
    from janome.tokenizer import Tokenizer
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt 
    import collections
    
    # ヘッダ
    st.header("テキスト可視化 - Word Cloud生成")

    t = Tokenizer()
    tokens = t.tokenize(text)

    word_list=[]
    for token in tokens:
        word = token.surface
        partOfSpeech = token.part_of_speech.split(',')[0]
        partOfSpeech2 = token.part_of_speech.split(',')[1]
            
        if partOfSpeech == "名詞":
            if (partOfSpeech2 != "非自立") and (partOfSpeech2 != "代名詞") and (partOfSpeech2 != "数"):
                word_list.append(word)
        
    words_wakati=" ".join(word_list)

    c = collections.Counter(word_list)
    print(c)

    stop_words = ['さん','そう','くん','ちゃん','たち']  
#    fpath = "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"  # 日本語フォント指定
#    fpath = "/usr/local/lib/python3.7/dist-packages/japanize_matplotlib/fonts/ipaexg.ttf" #Google Colab用のpath
    fpath = "/home/appuser/venv/lib/python3.7/site-packages/japanize_matplotlib/fonts/ipaexg.ttf" #Streamlit sharing用のpath

    st.write(fpath)


    option1 = st.selectbox('背景を選んでください',("white","black"))

    option2 = st.selectbox('カラーマップを選んでください',( 
    'tab10','inferno', 'magma', 'plasma', 'viridis',
    'Blues', 'BuGn', 'BuPu', 'GnBu',
    'Greens', 'Greys', 'OrRd', 'Oranges',
    'PuBu', 'PuBuGn', 'PuRd', 'Purples',
    'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd',
    'binary', 'gist_yarg', 'gist_gray', 'gray',
    'bone', 'pink', 'spring', 'summer',
    'autumn', 'winter', 'cool', 'Wistia',
    'hot', 'afmhot', 'gist_heat', 'copper',
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
    'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
    'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
    'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c',
    'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot',
    'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv', 'gist_rainbow', 'rainbow',
    'jet', 'nipy_spectral', 'gist_ncar'))


    if st.button("WordCloud生成"):

        if image_path == None:
                    
            wordcloud = WordCloud(
                font_path=fpath,
                width=600, height=400,   # default width=400, height=200
                background_color=option1,   # default="white"
                colormap=option2,   # default="set1?"
                stopwords=set(stop_words),
                max_words=200,   # default=200
                min_font_size=4,   #default=4
                collocations = False   #default = True
                ).generate(words_wakati)

        else:
            mask = np.array(Image.open(image_path))
            wordcloud = WordCloud(
                font_path=fpath,
                width=600, height=400,   # default width=400, height=200
                background_color="white",   # default="black"
                stopwords=set(stop_words),
                mask=mask,
                max_words=200,   # default=200
                min_font_size=4,   #default=4
                collocations = False   #default = True
                ).generate(words_wakati)

        
        plt.figure(figsize=(15,12))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.savefig("word_cloud.png")
        plt.show()

        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()


