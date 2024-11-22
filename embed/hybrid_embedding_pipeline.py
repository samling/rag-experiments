import os
from dotenv import load_dotenv
from os import listdir
from os.path import isfile, join
from haystack import Pipeline
from haystack.components.converters.csv import CSVToDocument
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder
from milvus_haystack import MilvusDocumentStore

load_dotenv(override=True)

milvus_uri = os.getenv('MILVUS_URI')
document_store = MilvusDocumentStore(
    connection_args={"uri": milvus_uri},
    drop_old=True,
    sparse_vector_field="sparse_vector",
)

current_file_path = os.path.dirname(os.path.realpath(__file__))
data_path=f"{current_file_path}/data/2024"
files_list = [f"{data_path}/{f}" for f in listdir(data_path) if isfile(join(data_path, f))]

print(files_list)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component(
    "converter",
    CSVToDocument()
)
indexing_pipeline.add_component(
    "cleaner",
    DocumentCleaner()
)
indexing_pipeline.add_component(
    "splitter",
    DocumentSplitter(
        split_by="sentence",
        split_length=1
    )
)
indexing_pipeline.add_component(
    "sparse_doc_embedder",
    FastembedSparseDocumentEmbedder(
        model=os.getenv('SPARSE_EMBEDDING_MODEL')
    )
)
indexing_pipeline.add_component(
    "dense_doc_embedder",
    OpenAIDocumentEmbedder(
        model=os.getenv('EMBEDDING_MODEL')
    )
)
indexing_pipeline.add_component(
    "writer",
    DocumentWriter(
        document_store=document_store,
        policy=DuplicatePolicy.OVERWRITE
    )
)

indexing_pipeline.connect("converter", "cleaner")
indexing_pipeline.connect("cleaner", "splitter")
indexing_pipeline.connect("splitter", "sparse_doc_embedder")
indexing_pipeline.connect("sparse_doc_embedder", "dense_doc_embedder")
indexing_pipeline.connect("dense_doc_embedder", "writer")

results =indexing_pipeline.run({"converter": {"sources": files_list}})

print("Number of documents:", document_store.count_documents())
