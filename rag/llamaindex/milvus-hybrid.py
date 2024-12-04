#https://docs.llamaindex.ai/en/stable/examples/vector_stores/MilvusHybridIndexDemo/
import logging
import os
import sys
import textwrap

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from IPython.display import Markdown, display

load_dotenv()

import openai
openai.base_url = os.getenv('OPENAI_BASE_URL')
openai.api_key = os.getenv('OPENAI_API_KEY')

documents = SimpleDirectoryReader("./rag/llamaindex/data/paul_graham/").load_data()

print("Document ID:", documents[0].doc_id)

milvus_host = os.getenv('MILVUS_URI')
vector_store = MilvusVectorStore(
    uri=f"https://{milvus_host}:19530",
    dim=1536,
    overwrite=True,
    enable_sparse=True,
    hybrid_ranker="RRFRanker",
    hybrid_ranker_params={"k": 60},
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
