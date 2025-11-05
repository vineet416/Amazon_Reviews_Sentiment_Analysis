from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
import pandas as pd
import pickle
import base64
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import nltk

# Download required NLTK data (for Streamlit Cloud deployment)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
    
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Initialize stopwords and remove 'not' to retain negation context
STOPWORDS = set(stopwords.words("english"))
if 'not' in STOPWORDS:
    STOPWORDS.remove('not')

# Load models once at startup
with open("Models/xgboost_model.pkl", "rb") as f:
    predictor = pickle.load(f)
with open("Models/tfidfVectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    """Convert nltk POS tag to wordnet POS tag"""
    if tag.startswith('J'):
        return 'a'  # adjective
    elif tag.startswith('V'):
        return 'v'  # verb
    elif tag.startswith('N'):
        return 'n'  # noun
    elif tag.startswith('R'):
        return 'r'  # adverb
    else:
        return 'n'  # default to noun

def preprocess_text(text):
    # Removing special characters and numbers
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()  # Converting to lowercase
    review = nltk.word_tokenize(review)  # Tokenization
    pos_tags = nltk.pos_tag(review)  # finding pos tags
    # Lemmatization and removing stopwords
    review = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags if word not in STOPWORDS]
    review = ' '.join(review)  # Joining the words back to form the cleaned review
    return review

@app.route("/")
def home():
    return render_template("landing.html")

@app.route("/predict")
def predict_page():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" in request.files:
            file = request.files["file"]
            data = pd.read_csv(file)
            data["Predicted"] = data["Sentence"].apply(lambda x: predict_sentiment(x))
            
            # Generate graph
            plt.figure(figsize=(5,5))
            data["Predicted"].value_counts().plot(kind="pie", autopct="%1.1f%%")
            img = BytesIO()
            plt.savefig(img, format="png")
            plt.close()
            img.seek(0)
            
            # Prepare response
            output = BytesIO()
            data.to_csv(output, index=False)
            output.seek(0)
            
            response = send_file(output, mimetype="text/csv", 
                               as_attachment=True, download_name="predictions.csv")
            response.headers["X-Graph"] = base64.b64encode(img.getvalue()).decode("utf-8")
            return response
        
        elif "text" in request.json:
            text = request.json["text"]
            return jsonify({"result": predict_sentiment(text)})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def predict_sentiment(text):
    processed = preprocess_text(text)
    vectorized = tfidf_vectorizer.transform([processed]).toarray()
    prediction = predictor.predict(vectorized)[0]
    return "Positive" if prediction == 1 else "Negative"

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)