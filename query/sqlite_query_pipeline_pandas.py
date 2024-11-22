import argparse
import datetime
import os
import pandas as pd
import sqlite3
import textwrap
from dotenv import load_dotenv
from typing import List, Any
from haystack import component, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators.openai import OpenAIGenerator

load_dotenv(override=True)

database = os.getenv('SQLITE_DB_PATH')

def print_indented(text, indent=4):
    """Print a multiline string with indentation, handling long lines"""
    width = os.get_terminal_size().columns - indent
    wrapper = textwrap.TextWrapper(width=width, initial_indent=' ' * indent, subsequent_indent=' ' * indent)

    lines = text.splitlines()
    indented_lines = []

    for line in lines:
        wrapped_line = wrapper.fill(line)
        indented_lines.append(wrapped_line)
    print("\n".join(indented_lines))

@component
class QueryHelper:
    def __init__(self, sql_database: str, question: str):
        self.db_path = sql_database
        self.question = question
        self._connection = None

    @property
    def connection(self):
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._connection

    def get_current_date(self):
        now = datetime.datetime.now()
        curr_date = now.strftime("%Y-%m-%d")
        return curr_date

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
            print(f"\nError getting categories: {str(e)}")
            return ""


    def get_accounts(self, table_name: str) -> List[str]:
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT DISTINCT account FROM {table_name} WHERE account IS NOT NULL ORDER BY account")
            accounts = cursor.fetchall()
            return ", ".join([f"{account[0]} " for account in accounts])
        except Exception as e:
            print(f"\nError getting accounts: {str(e)}")
            return ""

    @component.output_types(question=str, curr_date=str, schema=str, categories=str, accounts=str)
    def run(self, table_name: str = 'transactions'):
        question = self.question
        curr_date = self.get_current_date()
        schema = self.get_schema(table_name)
        categories = self.get_categories(table_name)
        accounts = self.get_accounts(table_name)
        return {"question": question, "curr_date": curr_date, "schema": schema, "categories": categories, "accounts": accounts}

@component
class SQLQuery:
    def __init__(self, sql_database: str):
        self.db_path = sql_database
        self._connection = None

    def _get_connection(self):
        if self._connection is None:
            print("\nEstablishing database connection...")
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
            print("Connected.")
        return self._connection

    @component.output_types(results=Any, query=Any)
    def run(self, queries: List[str]):
        query = queries[0]
        print(f"\nExecuting query:")
        print_indented(f"{query}")
        connection = self._get_connection()
        try:
            df = pd.read_sql_query(
                query,
                connection,
            )
            result = df.to_string(index=False)
            return {"results": result, "query": query}
        except Exception as e:
            print(f"\nError processing query: {str(e)}")
            return {"results": f"Error: {str(e)}", "query": query}

def create_pipeline(question: str):
    sql_prompt = PromptBuilder(template="""
                Please generate a SQL query. The query should answer the following Question: {{question}};

                The current date is:
                {{curr_date}}

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
        - Start your answer with the table; don't tell the user anything like 'here is the result'.
        - Translate any month numbers to names, and add a $ to any dollar amount.
        - Don't truncate the list of results unless it exceeds 30 rows.
        - Add a natural language analysis of this data that answers my original question. 
        - Include specific numbers and trends if relevant. 
        - Make it conversational but informative.

    Response:""")

    base_model = os.getenv('BASE_MODEL')
    analysis_model = os.getenv('ANALYSIS_MODEL')

    query_helper = QueryHelper(sql_database=database, question=question)
    sql_generator = OpenAIGenerator(model=base_model, timeout=30)
    sql_querier = SQLQuery(sql_database=database)
    analysis_generator = OpenAIGenerator(model=analysis_model, timeout=30)

    sql_pipeline = Pipeline()

    # Create components
    sql_pipeline.add_component("query_helper", query_helper)
    sql_pipeline.add_component("sql_prompt", sql_prompt)
    sql_pipeline.add_component("sql_generator", sql_generator)
    sql_pipeline.add_component("sql_querier", sql_querier)
    sql_pipeline.add_component("analysis_prompt", analysis_prompt)
    sql_pipeline.add_component("analysis_generator", analysis_generator)

    # Load the initial prompt with the query and additional context
    sql_pipeline.connect("query_helper.question", "sql_prompt.question")
    sql_pipeline.connect("query_helper.curr_date", "sql_prompt.curr_date")
    sql_pipeline.connect("query_helper.schema", "sql_prompt.schema")
    sql_pipeline.connect("query_helper.categories", "sql_prompt.categories")
    sql_pipeline.connect("query_helper.accounts", "sql_prompt.accounts")

    # Send the prompt to the generator
    sql_pipeline.connect("sql_prompt.prompt", "sql_generator.prompt")
    
    # Send the reply to the SQL querier
    sql_pipeline.connect("sql_generator.replies", "sql_querier.queries")
    
    # Send the origin question, query, and results to the analyzer prompt
    sql_pipeline.connect("sql_querier.results", "analysis_prompt.results")
    sql_pipeline.connect("sql_querier.query", "analysis_prompt.query")
    sql_pipeline.connect("query_helper.question", "analysis_prompt.question")

    # Send the prompt to the analyzer
    sql_pipeline.connect("analysis_prompt.prompt", "analysis_generator.prompt")

    return sql_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a sqlite database and analyze the results with an LLM.")
    parser.add_argument('--query', type=str, help='A natural language query about your data.', required=True)
    args = parser.parse_args()
    question: str = args.query

    sql_pipeline = create_pipeline(question)

    try:
        print("\nStarting pipeline execution...")
        print(f"\nYour question:\n    {question}")
        result = sql_pipeline.run(
            {
                "query_helper": {
                    "table_name": "transactions"
                },
            }
        )

        if "analysis_generator" in result and result["analysis_generator"]["replies"]:
            print("\nAnswer:")
            print_indented(result['analysis_generator']['replies'][0])
        else:
            print("\nNo analysis generated.")

    except Exception as e:
        print(f"\nPipeline Error: {str(e)}")