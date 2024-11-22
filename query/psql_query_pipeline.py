import datetime
import os
import psycopg2
from dotenv import load_dotenv
from configparser import ConfigParser
from decimal import Decimal
from typing import List

from haystack import component, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators.openai import OpenAIGenerator

load_dotenv(override=True)

prompt = PromptBuilder(template="""Please generate a Postgres SQL query. The query should answer the following Question: {{question}};
            The query is to be answered for the table is called 'transactions' with the following columns: {{columns}};
            Be creative with your answer. For example, a query asking for transactions by a certain merchant name may have
            the merchant name in the merchant column or the description column.
            You should only respond with the SQL query without any code formatting.
            Answer:""")

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

def connect(config):
    try:
        with psycopg2.connect(**config) as conn:
            return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)

def serialize_result(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, datetime.date):
        return obj.isoformat()
    return obj

@component
class SQLQuery:
    def __init__(self):
        self.config = load_config()
        self.connection = connect(self.config)

    def get_columns(self, table_name: str) -> List[str]:
        with self.connection.cursor() as cur:
            cur.execute(f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position;
            """, (table_name,))
            columns = [row[0] for row in cur.fetchall()]
            return {"columns": columns}

    @component.output_types(result=List[str], queries=List[str])
    def run(self, queries: List[str]):
        with self.connection.cursor() as cur:
            results = []
            for query in queries:
                cur.execute(query)
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                serialized_rows = [[serialize_result(cell) for cell in row] for row in rows]
                result = [dict(zip(columns, row)) for row in serialized_rows]
                results.append(result)

        return {"result": results[0], "queries": queries}

query = SQLQuery()
base_model = os.getenv('BASE_MODEL')
llm = OpenAIGenerator(model=base_model)
columns = query.get_columns('transactions')

sql_pipeline = Pipeline()
sql_pipeline.add_component("prompt", prompt)
sql_pipeline.add_component("llm", llm)
sql_pipeline.add_component("sql_querier", query)

sql_pipeline.connect("prompt", "llm")
sql_pipeline.connect("llm.replies", "sql_querier.queries")

result = sql_pipeline.run(
    {
        "prompt": {
            "question": "What were all of the venmo transactions in October 2024?",
            "columns": columns['columns']
        }
    }
)
print(result["sql_querier"]["result"])