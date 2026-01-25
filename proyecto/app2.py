import pandas as pd
import re
import os
import nltk
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Descargar stopwords si no las tienes
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

app = Flask(__name__)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# --- CARGA Y UNIÓN DE LOS DOS ARCHIVOS ---
print("Cargando datos...")
df_true = pd.read_csv('True.csv')
df_fake = pd.read_csv('Fake.csv')

# Creamos la columna 'label' (1 para real, 0 para falso)
df_true['label'] = 1
df_fake['label'] = 0

# Unimos ambos y mezclamos
df = pd.concat([df_true, df_fake]).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True) # Mezclar datos

# Entrenamos el modelo (usamos solo una parte para que cargue rápido)
print("Entrenando modelo...")
df['text'] = df['text'].apply(lambda x: clean_text(str(x)))

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)
print("¡Servidor listo!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        if not news_text.strip():
            return render_template('index.html', prediction_text="⚠️ Por favor, introduce texto.")
        
        cleaned = clean_text(news_text)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        
        result = "🟢 NOTICIA VERDADERA" if prediction == 1 else "🔴 NOTICIA FALSA"
        return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)