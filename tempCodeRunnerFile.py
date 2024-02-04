from flask import Flask, request, render_template, jsonify
import os
import fitz  # PyMuPDF for extracting text from PDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

app = Flask(__name__)

# Define the folder where uploaded resumes will be stored
UPLOAD_FOLDER = '/Users/jay/Desktop/advent_of_code/job_test/uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Specify the path to your CSV file containing job descriptions
JOB_DESCRIPTIONS_CSV_PATH = '/Users/jay/Downloads/naukri_com-job_sample.csv'  # Update this path

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
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

def load_job_descriptions(csv_path):
    df = pd.read_csv(csv_path)
    return df['jobdescription'].tolist()

def get_job_recommendations(resume_text, job_descriptions):
    documents = [resume_text] + job_descriptions
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    recommendations = sorted(enumerate(cosine_similarities[0]), key=lambda x: x[1], reverse=True)
    top_five = recommendations[:5]
    return [job_descriptions[i] for i, _ in top_five]

@app.route('/', methods=['GET', 'POST'])
def upload_resume():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if not allowed_file(file.filename):
            return 'Invalid file extension'
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        resume_text = extract_text_from_pdf(filename)
        preprocessed_resume_text = preprocess_text(resume_text)
        
        # Load job descriptions from CSV
        job_descriptions = load_job_descriptions(JOB_DESCRIPTIONS_CSV_PATH)
        preprocessed_job_descriptions = [preprocess_text(desc) for desc in job_descriptions]
        
        recommendations = get_job_recommendations(preprocessed_resume_text, preprocessed_job_descriptions)
        return render_template('display.html', recommendations=recommendations)
    return render_template('upload.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)


