import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
from typing_extensions import Concatenate

# Load API credentials
load_dotenv()
openai_organization = os.getenv("OPENAIORG")
openai_api_key = os.getenv("OPENAIAPIKEY")
serp_api_key = os.getenv("SERPAPIKEY")

# Load PDF
pdfreader = PdfReader("./Documents/Christies1_report.pdf")
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# Chunking text
text_splitter = CharacterTextSplitter(separator="\n",
                                     chunk_size=800,
                                     chunk_overlap=200,
                                     length_function=len,)
texts = text_splitter.split_text(raw_text)
print(len(texts))

# Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
document_search = FAISS.from_texts(texts, embeddings)

# Question answering
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
model_name = "gpt-3.5-turbo"
llm = OpenAI(model_name=model_name, temperature=0, openai_api_key=openai_api_key)
chain = load_qa_chain(llm, chain_type="stuff")

query = 'Can you provide details on the Birkhead formation'
docs = document_search.similarity_search(query=query)
chain.run(input_documents=docs, question=query)

