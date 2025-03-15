# 🔍 AI-Powered LinkedIn Profile Search

This project is an **AI-powered search engine for LinkedIn profiles**, allowing users to find relevant candidates based on **natural language queries**. It uses **FAISS for similarity search** and **Hugging Face sentence-transformers** to rank profiles.

## **🚀 Features**
- 🔎 **Search LinkedIn profiles** using AI-powered embeddings.
- 🎯 **Feedback loop** to improve results over time.
- ⚡ **Fast similarity search** with FAISS.
- 🏆 **Customizable AI models** (easily switch transformer models).
- 🌐 **Web-based UI with Streamlit**.

---

## **📦 Installation**
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/your-username/linkedin-ai-search.git
cd linkedin-ai-search
```

### **2️⃣ Set Up a Virtual Environment (Recommended)**
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

---

## **📊 CSV Data Structure**
To use this app, you must provide a CSV file named **`linkedin_profiles.csv`** inside the **`data/`** directory.

### **✅ Required CSV Format**
| mobile | name | link | headline | about | group | field | pl | company | position | gender | tags | email | age | date | # | newsletter | source | volunteer | id |
|--------|------|------|----------|-------|-------|-------|----|---------|----------|--------|------|-------|-----|------|---|------------|--------|-----------|----|
| +123456 | John Doe | https://linkedin.com/in/johndoe | Data Scientist | AI expert with 10+ years in the field | AI Community | Data Science | USA | Google | Senior Data Scientist | Male | AI, ML | johndoe@email.com | 35 | 2023-04-01 | 1 | Yes | User Submission | Volunteer | 001 |

**📌 Notes:**
- The **`link`** column must contain valid LinkedIn profile URLs.
- The **`headline`**, **`about`**, **`company`**, **`position`**, and **`tags`** columns are used for search queries.
- The **file must be in `data/linkedin_profiles.csv`**.

---

## **⚙️ Configuration**
### **Change AI Model**
To use a different **sentence-transformer model**, edit `config.py`:
```python
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Change to another model if needed
```

### **Reset Feedback**
To clear previous feedback and restart learning, delete the feedback file:
```sh
echo '{"feedback": []}' > data/user_feedback.json
```

---

## **🔧 Running the App**
### **1️⃣ Generate Embeddings**
Before using the app, process the LinkedIn profiles:
```sh
python backend/process_data.py
```
This **creates FAISS embeddings** and saves metadata.

### **2️⃣ Run the Web App**
```sh
streamlit run frontend/app.py
```
The **app will launch in your browser** at `http://localhost:8501`.

---

## **🛠 Usage**
1️⃣ **Enter a search query** in natural language (e.g., *"I need a speaker on AI ethics"*).  
2️⃣ **Browse the top matches** with their LinkedIn links.  
3️⃣ **Select the best profiles** to train the AI for better future results.  
4️⃣ **Refine your search** and find better recommendations over time!  

---

## **📚 Technologies Used**
- **[Sentence Transformers](https://www.sbert.net/)** (Natural Language Processing)
- **[FAISS](https://faiss.ai/)** (Efficient Similarity Search)
- **[Streamlit](https://streamlit.io/)** (Interactive UI)
- **[Hugging Face](https://huggingface.co/)** (Model Hosting)
- **[Scikit-learn](https://scikit-learn.org/)** (Cosine Similarity)
- **[Pandas](https://pandas.pydata.org/)** (Data Processing)

---

## **💡 Updating the App**
To update the project with new profiles:
```sh
1️⃣ Add new profiles to `data/linkedin_profiles.csv`
2️⃣ Run: `python backend/process_data.py`
3️⃣ Restart: `streamlit run frontend/app.py`
```

---

## **💾 Troubleshooting**
### **Common Issues & Fixes**
| Problem | Solution |
|---------|----------|
| ❌ No results found | Ensure profiles exist in `data/linkedin_profiles.csv` and rerun `process_data.py`. |
| ⚠️ LinkedIn links not working | Check if the `"link"` column exists in the CSV and `process_data.py` saves `"linkedin_url"`. |
| 💾 Memory error on large datasets | Try `faiss-gpu` instead of `faiss-cpu` in `requirements.txt`. |
| ⚡ Slow search | Switch to `bge-small-en-v1.5` in `config.py` for **faster results**. |

---

## **📝 License**
This project is open-source under the **MIT License**.

---

## **👨‍💻 Contributors**
👤 **Pascal Guéra**  
📧 pascal.guera@gmail.com

👤 **Virginia Wenger**  
📧 wenger.virginia@gmail.com
🔐 GitHub: [your-username](https://github.com/your-username)  
---

🎯 **Now you're ready to use AI-powered LinkedIn search! 🚀🔥**