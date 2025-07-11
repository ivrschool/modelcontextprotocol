from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# List of URLs to load documents from
urls = [
    "https://medium.com/fundamentals-of-artificial-intellegence/all-langchain-features-at-one-place-agent-mcp-evals-ci-cd-e26bf06667a2",
    "https://medium.com/i-am-writing/my-f1-visa-application-was-rejected-at-the-delhi-us-embassy-9e954639a7b3",
    "https://medium.com/i-am-writing/my-f1-visa-got-approved-on-2nd-attempt-d1c817c1818b",
    "https://medium.com/i-am-writing/you-are-not-loyal-to-one-field-d76150b9c242"
]

# Load documents from the URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

# Split the documents into chunks
doc_splits = text_splitter.split_documents(docs_list)

# Add the document chunks to the "vector store" using OpenAIEmbeddings
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
)


# With langchain we can easily turn any vector store into a retrieval component:
retriever = vectorstore.as_retriever(k=6)