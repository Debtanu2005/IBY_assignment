# Question generator ai agent

## Author
- **Name**: Debtanu Das   
- **University**: IIT Bhubaneswar
- **Department**: Computer Science

---

## 📌 Project Overview
This project builds an **AI Agent** that can **reason, plan, and execute** to automate a manual academic task:  
**Summarization and Question Generation from Scientific Documents.**

The agent uses:
- A **fine-tuned large language model (Qwen3-Next-80B-Instruct with LoRA adapters)** for **scientific summarization**.  
- **Gemini Flash API** for generating and answering questions from the processed documents.  
- **LangChain + Chroma** for retrieval and vector database storage.  

---

## 🎯 Why This Fine-Tuning Target?
I chose **scientific summarization** as the fine-tuning task because:
- Summarizing research papers and long technical documents is **time-consuming** in daily university work.  
- Fine-tuning ensures **task specialization**: the model produces **concise, domain-adapted summaries**.  
- Improves **reliability**: generic LLMs often hallucinate; the fine-tuned model stays faithful to input text.  
- Enables **adapted style**: outputs match the academic abstract style expected in reports and research summaries.  

---

## 🏗️ Agent Architecture
**Components & Flow:**
1. **Document Loader** – Load PDF lecture notes/reports using `PyPDFLoader`.  
2. **Vector Store** – Store embeddings with Chroma for semantic search.  
3. **Fine-Tuned Model** – `finetune.py` trains Qwen3 with LoRA on arXiv summarization dataset.  
4. **Question Generator** – `question_generator.py` generates exam-style questions.  
5. **Answer Generator** – Retrieves context, summarizes it, and produces answers.  
6. **Evaluator** – Uses ROUGE/BLEU metrics to evaluate agent reliability.  

---

## 📂 Repository Contents
- `finetune.py` → Fine-tuning setup (Qwen + LoRA on arXiv dataset).  
- `question_generator.py` → AI Agent logic (question generation + answering).  
- `result.txt` → Generated Q&A outputs.  
- `Process_QA` used to evaluate the mertics

---

## 📊 Data Science Report

### Fine-Tuning Setup
- **Dataset**: [kaggle dataset](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)  
- **Base Model**: facebook/bart-base  
- **Quantization**: 4-bit (bitsandbytes) for GPU efficiency  
- **Training**: 3 epoch, learning rate 3e-5  

### Results
- Model outputs short, abstract-style summaries.  
- Trainable parameters greatly reduced using LoRA (~0.1% of base model).  

### Evaluation
We used **ROUGE metrics** to measure summarization quality:  

- **ROUGE-1**: Precision = 0.387, Recall = 0.778, F1 = 0.517  
- **ROUGE-2**: Precision = 0.184, Recall = 0.369, F1 = 0.245  
- **ROUGE-L**: Precision = 0.212, Recall = 0.427, F1 = 0.284  


---

## 🚀 Deliverables
- ✅ Source code of the prototype (`finetune.py`, `question_generator.py`)  
- ✅ AI agent architecture documentation (this README)  
- ✅ Data science report (fine-tuning setup, evaluation with ROUGE scores)  

---






