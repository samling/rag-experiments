import datetime
import json
import os
import numpy as np
import pandas as pd
from configparser import ConfigParser
from decimal import Decimal
from dotenv import load_dotenv
from typing import List, Any
from haystack import component, Pipeline
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.generators.openai import OpenAIGenerator
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.engine.base import Connection

load_dotenv()

def load_config(filename='database.ini', section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)

    config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            config[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    
    return config

def create_connection_string(config):
    return f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"

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
    def __init__(self):
        self.config = load_config()
        connection_string = create_connection_string(self.config)
        self.engine = create_engine(connection_string, future=True)

    def get_schema(self, table_name: str) -> str:
        # Get column information using SQLAlchemy inspector
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name)
        
        # Format column information
        schema_info = []
        for col in columns:
            schema_info.append(f"{col['name']} ({str(col['type'])})")
        
        return ", ".join(schema_info)

    @component.output_types(schema=str)
    def run(self, table_name: str = 'transactions'):
        schema = self.get_schema(table_name)
        return {"schema": schema}

@component
class SQLQuery:
    def __init__(self):
        self.config = load_config()
        connection_string = create_connection_string(self.config)
        self.engine = create_engine(connection_string, future=True)

    def get_columns(self, table_name: str) -> List[str]:
            query = f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position;
            """
            with self.engine.connect() as connection:
                df = pd.read_sql_query(query, connection, params={"table_name": table_name})
                return df['column_name'].tolist()

    @component.output_types(results=Any, query=Any)
    def run(self, queries: List[str]):
        with self.engine.connect() as connection:
            query = queries[0]
            print(f"Executing query: {query}")

            try:
                sql = text(query)
                df = pd.read_sql_query(sql, connection)
                result = df.to_string(index=False)
                return {"results": result, "query": query}
            except Exception as e:
                print(f"Error processing query: {str(e)}")
                raise

    

@component
class QuestionHolder:
    @component.output_types(question=str)
    def run(self, question: str):
        return {"question": question}

question_holder = QuestionHolder()
sql_querier = SQLQuery()
base_model = os.getenv('BASE_MODEL')
sql_generator = OpenAIGenerator(model=base_model)
analysis_model = os.getenv('ANALYSIS_MODEL')
analysis_generator = OpenAIGenerator(model=analysis_model)
sql_prompt = PromptBuilder(template="""i
            Please generate a Postgres SQL query. The query should answer the following Question: {{question}};

            The table 'transactions' has the following schema:
            {{schema}}

            Be creative with your answer. For example, a query asking for transactions by a certain merchant name may have
            the merchant name in the merchant column or the description column.
            Try case-insensitive matching for better results.
            Try alternative names; for example, if you know that SCE is Southern California Edison, look for instances of "Edison".
            Try and avoid too many AND statements; if you must use them, make sure to make them as flexible as possible to avoid empty results.

            Keep the original column names in your query, only add additional columns to the result if necessary.

            Order the results from least to most recent.

            Avoid including the category column in your query.
            You should only respond with the SQL query without any code formatting, including no triple backticks or ```.
            Answer:""")
analysis_prompt = PromptBuilder(template="""I asked: {{question}}

The SQL query used was:
{{query}}

The data from the database shows:
{{results}}

Please print the results in a tabular format wrapped in a codeblock above your analysis. Translate any month numbers to names, and add a $ to any dollar amount.
Please provide a natural language analysis of this data that answers my question. 
Include specific numbers and trends if relevant. 
Make it conversational but informative.
Response:""")
answer_builder = AnswerBuilder()


sql_pipeline = Pipeline()
sql_pipeline.add_component("schema_fetcher", SchemaFetcher())
sql_pipeline.add_component("question_holder", question_holder)
sql_pipeline.add_component("sql_prompt", sql_prompt)
sql_pipeline.add_component("sql_generator", sql_generator)
sql_pipeline.add_component("sql_querier", sql_querier)
sql_pipeline.add_component("analysis_prompt", analysis_prompt)
sql_pipeline.add_component("analysis_generator", analysis_generator)

sql_pipeline.connect("schema_fetcher.schema", "sql_prompt.schema")
sql_pipeline.connect("question_holder.question", "sql_prompt.question")
sql_pipeline.connect("sql_prompt.prompt", "sql_generator.prompt")
sql_pipeline.connect("sql_generator.replies", "sql_querier.queries")
sql_pipeline.connect("sql_querier.results", "analysis_prompt.results")
sql_pipeline.connect("sql_querier.query", "analysis_prompt.query")
sql_pipeline.connect("question_holder.question", "analysis_prompt.question")
sql_pipeline.connect("analysis_prompt.prompt", "analysis_generator.prompt")

question = "What can you tell me about Socal Edison bills throughout the year of 2024?",
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

print(result["analysis_generator"]["replies"][0])