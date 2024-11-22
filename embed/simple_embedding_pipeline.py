import os
from dotenv import load_dotenv
from os import listdir
from os.path import isfile, join
from haystack import Pipeline
from haystack.components.converters.csv import CSVToDocument
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from milvus_haystack import MilvusDocumentStore

load_dotenv(override=True)

milvus_uri = os.getenv('MILVUS_URI')
embedding_model = os.getenv('EMBEDDING_MODEL')

document_store = MilvusDocumentStore(
    connection_args={"uri": milvus_uri},
    drop_old=True,
)

current_file_path = os.path.dirname(os.path.realpath(__file__))
data_path=f"{current_file_path}/data/2024"
files_list = [f"{data_path}/{f}" for f in listdir(data_path) if isfile(join(data_path, f))]

print(files_list)

indexing_pipeline = Pipeline()

indexing_pipeline.add_component("converter", CSVToDocument())
indexing_pipeline.add_component("cleaner", DocumentCleaner())
indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=1))
indexing_pipeline.add_component("embedder", OpenAIDocumentEmbedder(model=embedding_model))
indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))

indexing_pipeline.connect("converter", "cleaner")
indexing_pipeline.connect("cleaner", "splitter")
indexing_pipeline.connect("splitter", "embedder")
indexing_pipeline.connect("embedder", "writer")

results =indexing_pipeline.run({"converter": {"sources": files_list}})

print("Number of documents:", document_store.count_documents())
