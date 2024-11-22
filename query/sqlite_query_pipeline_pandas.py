import datetime
import json
import os
import numpy as np
import pandas as pd
import sqlite3
from decimal import Decimal
from dotenv import load_dotenv
from typing import List, Any
from haystack import component, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators.openai import OpenAIGenerator
from sqlalchemy import text
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.engine.base import Connection

load_dotenv(override=True)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return str(obj)
        if isinstance(obj, (Connection, Dialect)):
            return str(obj)
        if pd.isna(obj):
            return None
        return super().default(obj)

@component
class SchemaFetcher:
    def __init__(self, sql_database: str):
        self.db_path = sql_database
        self._connection = None

    @property
    def connection(self):
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._connection

    def get_schema(self, table_name: str) -> str:
        # Get column information using SQLAlchemy inspector
        cursor = self.connection.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # Format column information
        schema_info = []
        for col in columns:
            schema_info.append(f"{col[1]} ({col[2]})")
        
        return ", ".join(schema_info)

    def get_categories(self, table_name: str) -> List[str]:
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT DISTINCT category FROM {table_name} WHERE category IS NOT NULL ORDER BY category")
            categories = cursor.fetchall()
            return ", ".join([f"{category[0]} " for category in categories])
        except Exception as e:
            print(f"Error getting categories: {str(e)}")
            return ""


    def get_accounts(self, table_name: str) -> List[str]:
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT DISTINCT account FROM {table_name} WHERE account IS NOT NULL ORDER BY account")
            accounts = cursor.fetchall()
            return ", ".join([f"{account[0]} " for account in accounts])
        except Exception as e:
            print(f"Error getting accounts: {str(e)}")
            return ""

    @component.output_types(schema=str, categories=str, accounts=str)
    def run(self, table_name: str = 'transactions'):
        schema = self.get_schema(table_name)
        categories = self.get_categories(table_name)
        accounts = self.get_accounts(table_name)
        return {"schema": schema, "categories": categories, "accounts": accounts}

@component
class SQLQuery:
    def __init__(self, sql_database: str):
        self.db_path = sql_database
        self._connection = None

    @property
    def connection(self):
        if self._connection is None:
            print("Creating a new database connection")
            self._connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=10.0,
                isolation_level=None
            )
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA cache_size=2000") # 2mb cache
            self._connection.execute("PRAGMA temp_store=MEMORY")
        return self._connection

    def get_columns(self, table_name: str) -> List[str]:
        if not hasattr(self, '_columns_cache'):
            self._columns_cache = {}
        
        if table_name not in self._columns_cache:
            cursor = self.connection.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            self._columns_cache[table_name] = [col[1] for col in columns]

        return self._columns_cache[table_name]
    
    @component.output_types(results=Any, query=Any)
    def run(self, queries: List[str]):
        query = queries[0]
        print(f"Executing query: {query}")

        try:
            df = pd.read_sql_query(
                query,
                self.connection,
            )
            result = df.to_string(index=False)
            return {"results": result, "query": query}
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return {"results": f"Error: {str(e)}", "query": query}

@component
class QuestionHolder:
    @component.output_types(question=str)
    def run(self, question: str):
        return {"question": question}

database = os.getenv('SQLITE_DB_PATH')

sql_prompt = PromptBuilder(template="""
            Please generate a SQL query. The query should answer the following Question: {{question}};

            The table 'transactions' has the following schema:
            {{schema}}

            The available categories are:
            {{categories}}

            The available accounts are:
            {{accounts}}

            Follow these suggestions for building a better query:
                - Be creative with your answer. For example, a query asking for transactions by a certain merchant name may have
                  the merchant name in the merchant column, the description column, and/or the account column.
                - Use case-insensitive matching.
                - Try alternative names; for example, if you know that SCE is Southern California Edison, look for instances of "Edison".
                - Keep the query simple; avoid too many AND statements.
                - Keep the original column names in your query, only add additional columns to the result if necessary.
                - Order the results from least to most recent.
                - Avoid relying on the category column; you can use it, but often times transactions are miscategorized or missing a category altogether.

            You should only respond with the SQL query without any code formatting, including no triple backticks or ```.

            Answer:""")
analysis_prompt = PromptBuilder(template="""I asked: {{question}}

The SQL query used was:
{{query}}

The data from the database shows:
{{results}}

Follow these instructions in formulating your response:
    - Print the results in a tabular format wrapped in a codeblock above your analysis.
    - Translate any month numbers to names, and add a $ to any dollar amount.
    - Don't truncate the list of results unless it exceeds 30 rows.
    - Add a natural language analysis of this data that answers my original question. 
    - Include specific numbers and trends if relevant. 
    - Make it conversational but informative.

Response:""")

base_model = os.getenv('BASE_MODEL')
analysis_model = os.getenv('ANALYSIS_MODEL')

schema_fetcher = SchemaFetcher(sql_database=database)
question_holder = QuestionHolder()
sql_generator = OpenAIGenerator(model=base_model, timeout=30)
sql_querier = SQLQuery(sql_database=database)
analysis_generator = OpenAIGenerator(model=analysis_model, timeout=30)

sql_pipeline = Pipeline()

sql_pipeline.add_component("schema_fetcher", schema_fetcher)
sql_pipeline.add_component("question_holder", question_holder)
sql_pipeline.add_component("sql_prompt", sql_prompt)
sql_pipeline.add_component("sql_generator", sql_generator)
sql_pipeline.add_component("sql_querier", sql_querier)
sql_pipeline.add_component("analysis_prompt", analysis_prompt)
sql_pipeline.add_component("analysis_generator", analysis_generator)

sql_pipeline.connect("schema_fetcher.schema", "sql_prompt.schema")
sql_pipeline.connect("schema_fetcher.categories", "sql_prompt.categories")
sql_pipeline.connect("schema_fetcher.accounts", "sql_prompt.accounts")
sql_pipeline.connect("question_holder.question", "sql_prompt.question")
sql_pipeline.connect("sql_prompt.prompt", "sql_generator.prompt")
sql_pipeline.connect("sql_generator.replies", "sql_querier.queries")
sql_pipeline.connect("sql_querier.results", "analysis_prompt.results")
sql_pipeline.connect("sql_querier.query", "analysis_prompt.query")
sql_pipeline.connect("question_holder.question", "analysis_prompt.question")
sql_pipeline.connect("analysis_prompt.prompt", "analysis_generator.prompt")

# columns = sql_querier.get_columns('transactions')
question: str = "Show me Venmo transactions since June."
try:
    print("\nStarting pipeline execution...")
    result = sql_pipeline.run(
        {
            "question_holder": {
                "question": question
            },
            "schema_fetcher": {
                "table_name": "transactions"
            },
        }
    )

    if "analysis_generator" in result and result["analysis_generator"]["replies"]:
        print(result["analysis_generator"]["replies"][0])
    else:
        print("No analysis generated.")

except Exception as e:
    print(f"Pipeline Error: {str(e)}")