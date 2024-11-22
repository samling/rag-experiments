import os
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.embedders import OpenAITextEmbedder
from milvus_haystack import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever

load_dotenv(override=True)

prompt_template = """Answer the following query based on the provided context. If the context does
                     not include an answer, reply with {{documents}}.\n
                     Query: {{query}}
                     Documents:
                     {% for doc in documents %}
                        {{ doc.content }}
                     {% endfor %}
                     Answer: 
                  """

milvus_uri = os.getenv('MILVUS_URI')
base_model = os.getenv('BASE_MODEL')
embedding_model = os.getenv('EMBEDDING_MODEL')

document_store = MilvusDocumentStore(
    connection_args={"uri": milvus_uri},
)

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", OpenAITextEmbedder(model=embedding_model))
rag_pipeline.add_component("retriever", MilvusEmbeddingRetriever(document_store=document_store, top_k=50))
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
rag_pipeline.add_component("generator", OpenAIGenerator(model=base_model, generation_kwargs={"temperature": 0}))

rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "generator")

query = "What were my Venmo expenses in October 2024?"

results = rag_pipeline.run(
    {
        "text_embedder": {"text": query},
        "prompt_builder": {"query": query},
    }
)

print('Query:', query)
print('RAG answer:', results["generator"]["replies"][0])
print(results)
