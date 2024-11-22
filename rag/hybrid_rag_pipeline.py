import os
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.utils.auth import Secret
from haystack_integrations.components.rankers.cohere import CohereRanker
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from milvus_haystack import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusHybridRetriever, MilvusEmbeddingRetriever
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder

load_dotenv(override=True)

prompt_template = """Answer the following query based on the provided context.\n

                     Query: {{query}}

                     Documents:
                     {% for doc in documents %}
                        {{ doc.content }}
                     {% endfor %}

                     Answer: 
                  """

milvus_uri = os.getenv('MILVUS_URI')
embedding_model = os.getenv('EMBEDDING_MODEL')

document_store = MilvusDocumentStore(
    connection_args={"uri": milvus_uri},
    # sparse_vector_field="sparse_vector",
)



### Pipeline

rag_pipeline = Pipeline()

# Embedders
# sparse_text_embedder = FastembedSparseTextEmbedder()
# dense_text_embedder = OpenAITextEmbedder(model="text-embedding-3-large")
text_embedder = OpenAITextEmbedder(model=embedding_model)
# rag_pipeline.add_component("sparse_text_embedder", sparse_text_embedder)
# rag_pipeline.add_component("dense_text_embedder", dense_text_embedder)
rag_pipeline.add_component("text_embedder", text_embedder)

# Retrievers
embedding_retriever = MilvusEmbeddingRetriever(document_store=document_store, top_k=100)
rag_pipeline.add_component("retriever", embedding_retriever)

# Joiners
# document_joiner = DocumentJoiner()
# rag_pipeline.add_component("document_joiner", document_joiner)

# Rankers
ranker = CohereRanker(
    api_base_url=os.getenv('OPENAI_BASE_URL'),
    api_key=Secret.from_env_var('OPENAI_API_KEY'),
    model=os.getenv('RERANK_MODEL'),
)
rag_pipeline.add_component("ranker", ranker)

# Propmt builders
prompt_builder = PromptBuilder(template=prompt_template)
rag_pipeline.add_component("prompt_builder", prompt_builder)

# Generators
generator_model = os.getenv('BASE_MODEL')
generator = OpenAIGenerator(generation_kwargs={"temperature": 0}, model=generator_model)
rag_pipeline.add_component("generator", generator)

# Answer builders
answer_builder = AnswerBuilder()
rag_pipeline.add_component("answer_builder", answer_builder)



### Connections

## embedder => retriever
# rag_pipeline.connect("sparse_text_embedder.sparse_embedding", "retriever.query_sparse_embedding")
# rag_pipeline.connect("dense_text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

## retrievers => ranker
rag_pipeline.connect("retriever", "ranker")

# ## joiner => ranker
# rag_pipeline.connect("document_joiner", "ranker")

# ## ranker => prompt_builder
rag_pipeline.connect("ranker", "prompt_builder.documents")

# ## prompt_Builder => generator, answer_builder
rag_pipeline.connect("prompt_builder", "generator")
rag_pipeline.connect("generator.replies", "answer_builder.replies")
rag_pipeline.connect("generator.meta", "answer_builder.meta")
rag_pipeline.connect("retriever", "answer_builder.documents")

query = "What were my Venmo transactions in October 2024?"

results = rag_pipeline.run(
    {
        "text_embedder": {"text": query},
        # "dense_text_embedder": {"text": query},
        # "sparse_text_embedder": {"text": query},
        "ranker": {"query": query},
        "prompt_builder": {"query": query},
        "answer_builder": {"query": query}
    }
)

print('Query:', query)
print('RAG answer:', results["answer_builder"]["answers"][0].data)
# print(results)
