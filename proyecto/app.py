import pandas as pd
import re
import nltk
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from flask_cors import CORS 

app = Flask(__name__)
CORS(app) 

# Configuración de NLTK - Añadimos español por si acaso
nltk.download('stopwords')
stop_words_en = set(stopwords.words("english"))
stop_words_es = set(stopwords.words("spanish"))
all_stop_words = stop_words_en.union(stop_words_es)

def clean_text(text):
    # 1. Convertir a minúsculas y quitar saltos de línea
    text = str(text).lower().replace('\n', ' ')
    # 2. Quitar menciones de Twitter (@user) y URLs
    text = re.sub(r'@\w+|http\S+|www\S+', '', text)
    # 3. Quitar caracteres especiales pero mantener letras y números básicos
    text = re.sub(r'[^\w\s]', '', text)
    # 4. Eliminar palabras vacías (stopwords)
    tokens = text.split()
    return " ".join([word for word in tokens if word not in all_stop_words])

print("--- ENTRENANDO MODELO OPTIMIZADO ---")

# Cargamos los datos
df_true = pd.read_csv('True.csv')
df_fake = pd.read_csv('Fake.csv')

# --- MEJORA CLAVE: ELIMINACIÓN DE SESGOS ---
# Las noticias reales de este dataset casi todas dicen "Reuters". 
# Si el modelo ve que NO dice "Reuters", dirá que es falsa. Hay que limpiar eso:
df_true['text'] = df_true['text'].apply(lambda x: re.sub(r'^.*?\(reuters\)\s*-', '', x, flags=re.IGNORECASE))

df_true['label'] = 1
df_fake['label'] = 0

# Usamos una muestra más equilibrada
df = pd.concat([df_true.sample(5000), df_fake.sample(5000)]).reset_index(drop=True)
df['text'] = df['text'].apply(clean_text)

# --- MEJORA DE VECTORIZACIÓN ---
# Usamos un rango de 1 a 3 palabras (trigramas) para captar mejor las frases
vectorizer = TfidfVectorizer(
    max_features=15000, 
    ngram_range=(1, 3), 
    min_df=2, 
    max_df=0.8
)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Usamos un parámetro de regularización 'C' más equilibrado para evitar que sea tan drástico
model = LogisticRegression(C=0.5, solver='liblinear', class_weight='balanced')
model.fit(X, y)

print("--- MODELO LISTO ---")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        if not news_text.strip() or len(news_text) < 10:
            return "⚠️ Texto demasiado corto para analizar"
        
        cleaned = clean_text(news_text)
        vector = vectorizer.transform([cleaned])
        
        # Obtenemos la probabilidad
        prob = model.predict_proba(vector)[0]
        confianza = round(max(prob) * 100, 2)
        prediction = model.predict(vector)[0]
        
        # Umbral de duda: Si la confianza es baja, avisar
        if confianza < 55:
            return f"❓ INCONCLUSO (Duda al {confianza}%)"
        
        if prediction == 1:
            return f"🟢 REAL ({confianza}%)"
        else:
            return f"🔴 FALSA ({confianza}%)"

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)