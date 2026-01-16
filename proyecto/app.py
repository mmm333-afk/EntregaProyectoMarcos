import pandas as pd
import re
import nltk
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from flask_cors import CORS 

# Inicialización
app = Flask(__name__)
CORS(app) 

# Configuración de NLTK
nltk.download('stopwords')
stop_words_en = set(stopwords.words("english"))
stop_words_es = set(stopwords.words("spanish"))
all_stop_words = stop_words_en.union(stop_words_es)

def clean_text(text):
    text = str(text).lower().replace('\n', ' ')
    # Limpieza de menciones, hashtags y URLs típicas de X
    text = re.sub(r'@\w+|#\w+|http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return " ".join([word for word in tokens if word not in all_stop_words])

try:
    # Cargamos una base de datos amplia
    df_true = pd.read_csv('True.csv').sample(10000)
    df_fake = pd.read_csv('Fake.csv').sample(10000)

    # Quitamos la firma de agencia para que no dependa de "Reuters"
    df_true['text'] = df_true['text'].apply(lambda x: re.sub(r'^.*?\(reuters\)\s*-', '', x, flags=re.IGNORECASE))
    
    df_true['label'] = 1
    df_fake['label'] = 0

    df = pd.concat([df_true, df_fake]).reset_index(drop=True)
    df['text'] = df['text'].apply(clean_text)

    # Vectorización
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,3))
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    # Modelo equilibrado
    model = LogisticRegression(C=0.1, solver='liblinear', class_weight='balanced')
    model.fit(X, y)
    print("--- SISTEMA ONLINE ---")

except FileNotFoundError:
    print("❌ Error: Archivos CSV no encontrados.")

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        
        if not news_text.strip() or len(news_text) < 10:
            return "⚠️ Texto muy corto"
        
        cleaned = clean_text(news_text)
        vector = vectorizer.transform([cleaned])
        
        # Obtenemos probabilidades: [Prob_Fake, Prob_Real]
        probs = model.predict_proba(vector)[0]
        prob_fake = probs[0]
        prob_real = probs[1]
        
        # --- LÓGICA DE DUDA POSITIVA ---
        # Solo diremos que es FALSA si el modelo está muy seguro (ej. > 70%)
        # De lo contrario, le daremos el beneficio de la duda como REAL
        if prob_fake > 0.70:
            res = f"🔴 FALSA ({round(prob_fake * 100, 2)}%)"
        else:
            # Aquí entra la "Duda": si es 50/50 o incluso un poco sospechosa, dirá REAL
            res = f"🟢 REAL ({round(prob_real * 100, 2)}%)"
            
        return res

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)