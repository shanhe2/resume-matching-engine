from sentence_transformers import SentenceTransformer, util

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def calculate_semantic_match(resume_text, job_text):
    """
    Uses a pre-trained Sentence Transformer to generate embeddings 
    and calculate the semantic cosine similarity.
    """
    # Load a lightweight, highly efficient semantic model from Hugging Face
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Convert texts to high-dimensional embeddings (capturing meaning)
    embeddings = model.encode([resume_text, job_text])
    
    # Calculate cosine similarity between the two embeddings
    cosine_score = util.cos_sim(embeddings[0], embeddings[1])
    
    # Extract the score and convert to a percentage
    match_score = cosine_score.item() * 100
    return round(match_score, 2)

if __name__ == "__main__":
    # Load raw data
    raw_resume = load_text("test_resume.txt")
    raw_job = load_text("job_desc.txt")
    
    print("Loading AI model and calculating semantic match...")
    
    # Generate the score
    score = calculate_semantic_match(raw_resume, raw_job)
    
    print(f"\n--- Competency Matching Engine (Semantic Version) ---")
    print(f"Resume-to-Job Match Score: {score}%")