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
# Fill NaN values with a placeholder string, e.g., an empty string
df = pd.read_csv(JOB_DESCRIPTIONS_CSV_PATH)
df['jobdescription'] = df['jobdescription'].fillna('')
job_descriptions = df['jobdescription'].tolist()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def preprocess_text(text):
    # Convert any non-string input to string and handle NaN values
    text = str(text) if text is not None else ''
    
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords_set]
    return ' '.join(cleaned_tokens)

def load_job_descriptions(csv_path):
    df = pd.read_csv(csv_path)
    df['jobdescription'] = df['jobdescription'].fillna('')
    return df


def get_job_recommendations(resume_text, df):
    job_descriptions = df['jobdescription'].tolist()
    documents = [preprocess_text(resume_text)] + [preprocess_text(desc) for desc in job_descriptions]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    recommendations = sorted(enumerate(cosine_similarities[0]), key=lambda x: x[1], reverse=True)
    top_five = recommendations[:5]
    
    top_recommendations = []
    for i, _ in top_five:
        top_recommendations.append({
            'company': df.iloc[i]['company'],
            'jobtitle': df.iloc[i]['jobtitle'],
            'jobdescription': df.iloc[i]['jobdescription'][:200]  # First 100 characters
        })
    
    return top_recommendations


def handle_resume_upload(file):
    if file.filename == '':
        return 'No selected file'
    if not allowed_file(file.filename):
        return 'Invalid file extension'
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    resume_text = extract_text_from_pdf(filename)
    
    df = load_job_descriptions(JOB_DESCRIPTIONS_CSV_PATH)  # Load as DataFrame
    recommendations = get_job_recommendations(resume_text, df)
    
    return recommendations


@app.route('/', methods=['GET', 'POST'])
@app.route('/upload', methods=['POST'])
def upload_resume():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return 'No file part'
        
        recommendations = handle_resume_upload(file)
        if isinstance(recommendations, str):  # Error message returned
            return recommendations
        return render_template('display.html', recommendations=recommendations)
    
    return render_template('upload.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)