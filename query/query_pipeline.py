import os
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from milvus_haystack import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever

load_dotenv(override=True)

milvus_uri = os.getenv('MILVUS_URI')
embedding_model=os.getenv('EMBEDDING_MODEL')

prompt_template = """Answer the following query based on the provided context. If the context does
                     not include an answer, reply with {{documents}}.\n
                     Query: {{query}}
                     Documents:
                     {% for doc in documents %}
                        {{ doc.content }}
                     {% endfor %}
                     Answer: 
                  """

document_store = MilvusDocumentStore(
    connection_args={"uri": milvus_uri},
)

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", OpenAITextEmbedder(model=embedding_model))
rag_pipeline.add_component("retriever", MilvusEmbeddingRetriever(document_store=document_store, top_k=50))

rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

query = "What were my Venmo expenses in October 2024?"

results = rag_pipeline.run(
    {
        "text_embedder": {"text": query},
    }
)

print('Query:', query)
for result in results['retriever']['documents']:
    print(result.content)
