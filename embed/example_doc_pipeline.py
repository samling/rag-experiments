from dotenv import load_dotenv
from haystack import Document
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

load_dotenv(override=True)

document_store = PgvectorDocumentStore(
    embedding_dimension=1536,
    vector_function="cosine_similarity",
    recreate_table=True,
    search_strategy="hnsw",
)

document_store.write_documents([
    Document(content="This is first", embedding=[0.1]*1536),
    Document(content="This is second", embedding=[0.3]*1536)
    ])
print(document_store.count_documents())
