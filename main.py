import tkinter as tk
import re
import spacy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from gensim.models import Word2Vec

def clean_text_ML(text):
    text = str(text).lower()  
    text = re.sub(r"http\S+", "", text)  
    text = re.sub(r"[^\w\s]", "", text)
    return text

nlp = spacy.load("en_core_web_sm")

def clean_text_genreDL(text):
    text = str(text).lower()  
    text = re.sub(r"http\S+", "", text)  
    text = re.sub(r"[^\w\s]", "", text)

    text = nlp(text)
    text = " ".join([token.lemma_ for token in text if not token.is_stop])
    return text

stop_words = set(stopwords.words('english'))

def clean_text_resumeDL(text):  
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\w*\d\w*', '', text)    
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

def vectorize_sentence(sentence, model, max_len=300):
    words = sentence.split()
    vectorized = []
    for word in words[:max_len]:
        if word in model.wv:
            vectorized.append(model.wv[word])
        else:
            vectorized.append(np.zeros(model.vector_size))  
    while len(vectorized) < max_len:
        vectorized.append(np.zeros(model.vector_size))
    return np.array(vectorized, dtype=np.float32)

wrd2vec_resume = Word2Vec.load('./model/word2vec.model')
encoder_model = load_model('./model/encoder_model.keras')
decoder_model = load_model('./model/decoder_model.keras')

print(type(wrd2vec_resume))
print("--------------------------------------")

def decode_sequence(input_seq, max_len=30):

    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, 100))

    decoded_sentence = []

    for _ in range(max_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        word_vector = output_tokens[0, -1, :]
        similar_words = wrd2vec_resume.wv.similar_by_vector(word_vector, topn=8) #topn = 10 To fix the "server server server" problem
        sampled_word = np.random.choice([w for w, _ in similar_words])
        decoded_sentence.append(sampled_word)

        target_seq[0, 0, :] = wrd2vec_resume.wv[sampled_word]

        states_value = [h, c]

    return ' '.join(decoded_sentence)

with open('./model/tfidfVectorizer - TFIDF.pkl', 'rb') as f:
    vectorizer_ML = pickle.load(f)

with open('./model/modelPredictGenre - TFIDF.pkl', 'rb') as f:
    model_ML = pickle.load(f)

genres = ['Action', 'Adventure', 'Boys\' Love', 'Comedy', 'Crime', 'Drama', 'Fantasy', 'Girls\' Love', 'Historical', 'Horror', 'Isekai','Magical Girls', 'Mecha', 'Medical', 'Mystery', 'Philosophical','Psychological', 'Romance', 'Sci-Fi', 'Slice of Life', 'Sports','Superhero', 'Thriller', 'Tragedy', 'Wuxia']

with open('./model/tokenizer_Word2Vec_Genre.pkl', 'rb') as f:
    tokenizer_genreDL = pickle.load(f)
    
model_genre_DL = load_model('.:model/modelPredictGenre - Word2Vec.keras')

model_resume_DL = load_model('./model/Resume_Word2Vec_Model.keras')


def check_request(event=None):
    content = text_box.get("1.0", tk.END).strip()
    lines = content.split("\n")
    last_input = lines[-1]
    if last_input[:2] == '$r' :
        text_box.insert(tk.END, f"\n---------------------------------\n")
        generate_summary(text=last_input[3:])
    elif last_input[:2] == '$g' :
        text_box.insert(tk.END, f"\n---------------------------------\\n")
        guess_genre_ML(text=last_input[3:])
    elif last_input[:2] == '$G' :
        text_box.insert(tk.END, f"\n---------------------------------\\n")
        guess_genre_DL(text=last_input[3:])
    else :
        text_box.insert(tk.END, f"\nSorry I didn't get that\n")
        text_box.see(tk.END)

def guess_genre_ML(text):
    valVec = vectorizer_ML.transform([clean_text_ML(text)])
    pred = model_ML.predict(valVec)

    pred = ", ".join([genres for genres, pred in zip(genres, pred[0]) if pred == 1])

    text_box.insert(tk.END, f"\n{pred}\n")
    text_box.see(tk.END)
    return "break"

def guess_genre_DL(text):
    valVec = pad_sequences(tokenizer_genreDL.texts_to_sequences([clean_text_genreDL(text)]), maxlen = 200)

    pred = model_genre_DL.predict(valVec)
    pred = (pred > 0.45).astype(int)

    pred = ", ".join([genres for genres, pred in zip(genres, pred[0]) if pred == 1])

    text_box.insert(tk.END, f"\n{pred}\n")
    text_box.see(tk.END)
    return "break"

def generate_summary(text):

    text = clean_text_resumeDL(text)
    sample_input = vectorize_sentence(text, wrd2vec_resume).reshape(1, 300, 100)

    text = decode_sequence(sample_input)

    text_box.insert(tk.END, f"\n{text}\n")
    text_box.see(tk.END)
    return "break"

# Main window
window = tk.Tk()
window.title("Summarizer Terminal")
window.geometry("800x600")

# Frame to contain the terminal
terminal_frame = tk.Frame(window, padx=10, pady=10, bg="#f0f0f0", relief=tk.RIDGE, bd=2)
terminal_frame.pack(pady=40)

# Terminal Text box
text_box = tk.Text(terminal_frame, width=80, height=20, wrap=tk.WORD, font=("Courier", 12))
text_box.pack()

# Insert initial instruction or welcome message
initial_message = "Hello I'm Vivi and I'm here to help you.\n Give me the text you need me to analize and just type before it :\n \t$g if you want the main genre of the manga\n \t$G if you want all the genre of the manga\n \tAnd $r if you want me to resume the text\n"
text_box.insert(tk.END, initial_message)

# Focus and bind Enter key
text_box.bind("<Return>", check_request)
text_box.focus_set()

window.mainloop()
