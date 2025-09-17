
import os
from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from typing_extensions import TypedDict
import re
from dotenv import load_dotenv
import google.generativeai as genai

from finetune import get_summary

load_dotenv()

geminai_key = os.getenv("geminai_key")
if not geminai_key:
    geminai_key = "AIzaSyBjvWwAFFAdT1KpVaz4VCEtAMXtb2rKkak"


genai.configure(api_key=geminai_key)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
)

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Process.pdf")
pages = loader.load()

type(pages)

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore.from_documents(pages , embeddings)

# type(vector_store)

# vector_store.similarity_search("What is the number of instruction in Risc v", k=3)




class graph_satate(TypedDict):
    file_path: str;
    imp_topics: str;
    question_set: str;
    pages: list;
    answers: dict;



def preprocess(text: str):
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"^\s+", "", text)
    return text

def generate_question(state:graph_satate):

    system_prompt = """### Instruction:
              You are a question generator. Read the given paragraph and create thoughtful, relevant questions based only on the information in the text.
              The questions should test understanding, highlight key points, and avoid introducing outside knowledge.generate each question in a
              new line generate only a few important question like 2-3 question per paragraph.after each question add a \n.

              ### Paragraph:
              {paragraph}

              ### Questions:
              """

    question_generator= genai.GenerativeModel('gemini-1.5-flash-latest')
    documents_prompt = "User's message: {paragraph}"
    questions = []

    for page in pages:
        prompt = documents_prompt.format(paragraph=page)
        response = question_generator.generate_content(f"{system_prompt}\n\n{prompt}")
        ans = preprocess(response.text)
        questions.append(ans)

    return {"question_set": questions,"pages": state["pages"]}



def generate_answer(state:graph_satate):

  system_prompt = """
      You are a helpful assistant.
      Your task is to answer questions strictly based on the given document.

      Guidelines:
      - Use only the information provided in the document to generate answers.
      - If the answer cannot be found in the document, clearly state: "The document does not contain this information."
      - Keep answers concise, clear, and accurate.
      - Do not add extra knowledge or assumptions beyond what is in the document.
      """

  answers = {}
  answer_generator = genai.GenerativeModel('gemini-1.5-flash-latest')
  

  for question in state["question_set"]:
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    context = get_summary(context)
    prompt = f"{system_prompt}\n\nDocument:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    response = answer_generator.generate_content(prompt)
    answers[question] = response.text

  return {"answers": answers, "question_set": state["question_set"], "pages": state["pages"]}

flow = StateGraph(graph_satate)

flow.add_node("quest", generate_question)
flow.add_node("ans", generate_answer)

flow.add_edge(START, "quest")
flow.add_edge("quest", "ans")
flow.add_edge("ans", END)

flow.compile()

answer = flow.compile().invoke({"pages": pages})

result=""

for question, ans in answer["answers"].items():
    print(f"Question: {question}")
    print(f"Answer: {ans}\n")
    result+=f"Question: {question}\nAnswer: {ans}\n\n"

with open ("result.txt","w") as f:
    f.write(result)


reference=""

with open("Process_QA.txt","rb") as f:
    reference = f.read()

from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

print(scorer.score(reference,result))



