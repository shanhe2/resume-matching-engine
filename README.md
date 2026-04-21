# resume-matching-engine

## Objective
This prototype was developed to automate the mapping of student resume to specific project requirements using natural language processing (NLP) and machine learning. It ingests unstructured text data (resumes and job descriptions), extracts core technical skills, and calculates a data-driven match score. 

## Tech Stack (Current: V2)
* **Language:** Python
* **Semantic Engine:** Hugging Face `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Backend:** PyTorch
* **Data Processing:** pandas, spaCy

---

## The Iteration Process & Results

### V1 Baseline: The Lexical Limitation (TF-IDF)
The initial prototype utilized `scikit-learn`'s TF-IDF and spaCy to calculate cosine similarity based on exact lexical overlap. 
* **V1 Match Score:** `11.98%`
* **Analysis:** The baseline score was artificially low because TF-IDF cannot interpret semantic meaning. While the candidate possessed the required skills, they used phrasing like *"data engineering"* and *"software development,"* whereas the project description requested *"data processing"* and *"software tools."*

### V2 Upgrade: Semantic Matching (Sentence Transformers)
To resolve the lexical limitations of V1, the engine was upgraded to use a pre-trained neural network (`all-MiniLM-L6-v2`). This allows the engine to generate high-dimensional embeddings and calculate semantic cosine similarity, mathematically recognizing that "machine learning" and "predictive modeling" are conceptually linked.
* **V2 Match Score:** `36.56%`
* **Analysis:** By shifting from keyword counting to semantic understanding, the engine achieved a **3x improvement in accuracy**. A >30% cosine similarity between a brief resume and a concise job description correctly identifies a highly viable candidate-to-project match from purely unstructured data.

---

## How to Run Locally
1. Clone the repository.
2. Create and active a virtual environment: 
```bash
   python -m venv venv
   # Windows: .\venv\Scripts\activate
   # Mac/Linux: source venv/bin/activate
3. Install dependencies:
   ```bash
   pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
   pip install sentence-transformers pandas spacy
   python -m spacy download en_core_web_sm
4. Run the engine: 
    ```bash
    python matcher.py