#INGESTING PDF

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

local_path = "data/books/certified.rf.sample_resumes.pdf"
if local_path:
    loader = UnstructuredPDFLoader(file_path = local_path)
    data = loader.load()
else:
    print("Upload a PDF file please.")

print(data[0].page_content)


#VECTOR EMBEDDINGS 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 

#keeps it coherent taking the context of everythign that comes before and after it 
#can always adjust the numbers to your liking / can fine-tune your model here 
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 7500, chunk_overlap = 100)
chunks = text_splitter.split_documents(data)

#takes a chunk converts them to vectors and then adds them to vector database 
vector_db = Chroma.from_documents(documents = chunks, embedding = OllamaEmbeddings(model = "nomic-embed-text", show_progress = True), collection_name = "local-rag")

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

local_model = "llama3"
llm = ChatOllama(model=local_model)
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to answer my question in one line with everyone's names.
    Original question: {question}""",
)
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

output = chain.invoke("Give me a skill")
print(output)











