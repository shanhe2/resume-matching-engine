# resume-matching-engine

## Objective
This prototype was developed to automate the mapping of student competencies to specific project requirements using natural language processing (NLP) and machine learning. It ingests unstructured text data (resumes and job descriptions), extracts core technical skills, and calculates a data-driven match score. 

## V1 Tech Stack
* **Language:** Python
* **NLP Processing:** spaCy (`en_core_web_sm`) for tokenization, stop-word removal, and lemmatization.
* **Vectorization & Math:** scikit-learn (`TfidfVectorizer`, `cosine_similarity`) 

## How It Works
1. **Ingestion:** Loads unstructured text from `test_resume.txt` and `job_desc.txt`.
2. **Preprocessing:** Uses spaCy to clean the text, isolating alphabetic lemmas and removing grammatical noise.
3. **Vectorization:** Converts the cleaned text into mathematical vectors using Term Frequency-Inverse Document Frequency (TF-IDF).
4. **Scoring:** Calculates the cosine similarity between the two vectors to generate a quantitative match percentage.

## V1 Analysis & Limitations (The TF-IDF Problem)
This V1 baseline intentionally uses TF-IDF to demonstrate the limitations of exact-keyword matching in unstructured data. 

In initial testing, the engine produced a **11.98% match score** between a highly qualified candidate and the target job description. 
* **Why:** TF-IDF relies on exact lexical overlap. While the candidate possessed the required skills, they used phrasing like *"data engineering"* and *"software development,"* whereas the project description requested *"data processing"* and *"software tools."*
* **Conclusion:** To accurately triangulate student competencies, the engine must understand semantic meaning, not just exact term frequency.

## Future Roadmap (V2)
To resolve the lexical limitations of V1, the next iteration of this engine will swap scikit-learn's TF-IDF for a pre-trained **Sentence Transformer** (via Hugging Face). This will allow the engine to generate high-dimensional embeddings and calculate semantic cosine similarity, recognizing that "machine learning" and "predictive modeling" are conceptually linked.

## How to Run Locally
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment and install dependencies:
   ```bash
   pip install pandas scikit-learn spacy
   python -m spacy download en_core_web_sm
4. Run the engine: 
    ```bash
    python matcher.py