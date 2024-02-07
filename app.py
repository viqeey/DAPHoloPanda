from flask import Flask, request, render_template, jsonify
import io
import fitz  # PyMuPDF for extracting text from PDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'pdf'}

JOB_DESCRIPTIONS_CSV_PATH = 'download_this_dataset\naukri_com-job_sample.csv'  # Update this path

# Load job descriptions from CSV
df = pd.read_csv(JOB_DESCRIPTIONS_CSV_PATH)
df['jobdescription'] = df['jobdescription'].fillna('')
job_descriptions = df['jobdescription'].tolist()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_data):
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords_set]
    return ' '.join(cleaned_tokens)

def get_job_recommendations(resume_text):
    documents = [preprocess_text(resume_text)] + [preprocess_text(desc) for desc in job_descriptions]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    recommendations = sorted(enumerate(cosine_similarities[0]), key=lambda x: x[1], reverse=True)
    top_five = recommendations[:5]
    
    top_recommendations = []
    for i, score in top_five:
        top_recommendations.append({
            'company': df.iloc[i]['company'],
            'jobtitle': df.iloc[i]['jobtitle'],
            'jobdescription': df.iloc[i]['jobdescription'][40:200],  # First 160 characters
            'similarity_score': round(score * 100, 2),  # Added similarity score, rounded and converted to percentage
        })
    
    return top_recommendations

@app.route('/', methods=['GET', 'POST'])
def upload_resume():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return 'No file part'

        if allowed_file(file.filename):
            resume_text = extract_text_from_pdf(file.read())
            recommendations = get_job_recommendations(resume_text)
            return render_template('display.html', recommendations=recommendations)
        else:
            return 'Invalid file extension'

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
