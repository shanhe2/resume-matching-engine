import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the NLP model
nlp = spacy.load("en_core_web_sm")

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_text(text):
    """
    Uses spaCy to tokenize, remove stop words, and lemmatize the text 
    to extract the core meaning.
    """
    doc = nlp(text.lower())
    # Keep only alphabetic tokens that aren't stop words
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def calculate_match_score(resume_text, job_text):
    """
    Calculates the cosine similarity between the preprocessed texts using TF-IDF.
    """
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the texts into mathematical vectors
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
    
    # Calculate cosine similarity between the two vectors (index 0 and index 1)
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    # Extract the score and convert to a percentage
    match_score = similarity_matrix[0][0] * 100
    return round(match_score, 2)

if __name__ == "__main__":
    # Load raw data
    raw_resume = load_text("test_resume.txt")
    raw_job = load_text("job_desc.txt")
    
    # Preprocess text to extract core competencies
    clean_resume = preprocess_text(raw_resume)
    clean_job = preprocess_text(raw_job)
    
    # Generate the score
    score = calculate_match_score(clean_resume, clean_job)
    
    print(f"--- Competency Matching Engine ---")
    print(f"Resume-to-Job Match Score: {score}%")