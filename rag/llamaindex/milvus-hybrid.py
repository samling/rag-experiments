#https://docs.llamaindex.ai/en/stable/examples/vector_stores/MilvusHybridIndexDemo/
import os
import uuid
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.schema import NodeWithScore
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.workflow import (
    Context,
    Event,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.milvus import MilvusVectorStore
from langfuse.decorators import observe
from langfuse.llama_index import LlamaIndexInstrumentor

load_dotenv()

instrumentor = LlamaIndexInstrumentor(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    debug=False
)
session_id = str(uuid.uuid4())

class RetrieverEvent(Event):
    """Result of running retrieval"""
    nodes: list[NodeWithScore]

class RerankEvent(Event):
    """Result of running reranking on retrieved nodes"""
    nodes: list[NodeWithScore]

class RAGIngester(Workflow):
    @step
    async def ingest(
        self, ctx: Context, ev: StartEvent
    ) -> StopEvent | None:
        dirname = ev.get("dirname")
        if not dirname:
            return None

        storage_context = ev.get("storage_context")
        if not storage_context:
            return None
        
        documents = SimpleDirectoryReader(dirname).load_data()
        index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            show_progress=True
        )
        return StopEvent(result=index)

class RAGRetriever(Workflow):
    @step
    async def retrieve(
        self, ctx: Context, ev: StartEvent
    ) -> RetrieverEvent | None:
        query = ev.get("query")
        index = ev.get("index")

        if not query:
            return None

        print(f"Query the database with: {query}")

        # Store query in the global context
        await ctx.set("query", query)

        # Get index from global context
        if index is None:
            print("Index is empty, load some documents before querying")

        retriever = index.as_retriever(similarity_top_k=2)
        nodes = await retriever.aretrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:
        llm = OpenAI(model="gpt-4o-mini")
        summarizer = CompactAndRefine(llm=llm, streaming=True, verbose=True)
        query = await ctx.get("query", default=None)

        response = await summarizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)


@observe()
async def run():
    Settings.embed_model = CohereEmbedding(
        api_key=os.getenv('COHERE_API_KEY')
    )

    with instrumentor.observe(
        session_id=session_id,
        user_id="Sam",
    ) as trace:
        milvus_host = os.getenv('MILVUS_URI')
        embedder_store = MilvusVectorStore(
            uri=f"http://{milvus_host}:19530",
            dim=1024,
            overwrite=True,
            enable_sparse=False,
            hybrid_ranker="RRFRanker",
            hybrid_ranker_params={"k": 60},
        )
        retriever_store = MilvusVectorStore(
            uri=f"http://{milvus_host}:19530",
            dim=1024,
            enable_sparse=False,
            hybrid_ranker="RRFRanker",
            hybrid_ranker_params={"k": 60},
        )
        storage_context = StorageContext.from_defaults(vector_store=embedder_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store=retriever_store,
            storage_context=storage_context,
            debug=True
        )

        ingester = RAGIngester()
        await ingester.run(
            dirname="./rag/llamaindex/data/paul_graham/",
            storage_context=storage_context
        )

        retriever = RAGRetriever()
        retriever_result = await retriever.run(
            query="What did the author do growing up?",
            index=index,
        )
        async for chunk in retriever_result.async_response_gen():
            print(chunk, end="", flush=True)
        print()

    instrumentor.flush()

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        asyncio.get_event_loop().close()