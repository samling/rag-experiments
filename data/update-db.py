import os
import sqlite3
import csv
import argparse
import logging
import re


def normalize_column_name(column_name):
    """
    Normalize column names:
    - Convert to lowercase
    - Replace spaces with underscores
    - Remove special characters
    """
    column_name = column_name.strip()
    column_name = column_name.lower()
    column_name = re.sub(r'\s+', '_', column_name)     # Replace spaces with underscores
    column_name = re.sub(r'[^\w]', '', column_name)    # Remove non-alphanumeric characters (optional)
    return column_name


def update_database(db_path, csv_path, table_name):
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Check if database file exists
    new_db = not os.path.exists(db_path)

    # Connect to SQLite database
    conn = sqlite3.connect(db_path)

    # Log whether a new database file has been created
    if new_db:
        logging.info(f'Database file "{db_path}" did not exist. A new database has been created.')
    else:
        logging.info(f'Connected to existing database: "{db_path}".')

    # Read the CSV file
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)

        # Normalize header column names
        csvreader.fieldnames = [normalize_column_name(col) for col in csvreader.fieldnames]
        logging.info(f'Normalized CSV column names: {csvreader.fieldnames}')

        # Dynamically create the table if it doesn't exist
        columns = ', '.join([f'{col} TEXT' for col in csvreader.fieldnames])  # All columns are TEXT by default
        unique_columns = ', '.join(csvreader.fieldnames)                      # Use all columns for the UNIQUE constraint
        create_table_query = f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY,
            {columns},
            UNIQUE({unique_columns})
        );
        '''
        conn.execute(create_table_query)

    # Count rows in the table before the update
    cursor = conn.cursor()
    cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
    before_count = cursor.fetchone()[0]
    logging.info(f'Number of rows in "{table_name}" before update: {before_count}')

    # Insert or update rows from the CSV file
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        csvreader.fieldnames = [normalize_column_name(col) for col in csvreader.fieldnames]
        rows_processed = 0

        for row in csvreader:
            placeholders = ', '.join([f':{col}' for col in csvreader.fieldnames])
            columns = ', '.join(csvreader.fieldnames)
            update_set = ', '.join([f'{col} = excluded.{col}' for col in csvreader.fieldnames])
            insert_query = f'''
            INSERT INTO {table_name} ({columns})
            VALUES ({placeholders})
            ON CONFLICT({unique_columns}) DO UPDATE SET
                {update_set};
            '''
            cursor.execute(insert_query, row)
            rows_processed += 1

    # Count rows in the table after the update
    cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
    after_count = cursor.fetchone()[0]
    logging.info(f'Number of rows in "{table_name}" after update: {after_count}')

    # Log the difference and rows processed
    logging.info(f'Number of new or updated rows: {after_count - before_count}')
    logging.info(f'Total rows processed from the CSV: {rows_processed}')

    # Commit changes and close the connection
    conn.commit()
    conn.close()


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Update an SQLite database with data from a CSV file."
    )
    parser.add_argument('--db', required=True, help='Path to the SQLite database file')
    parser.add_argument('--csv', required=True, help='Path to the CSV file')
    parser.add_argument('--table', required=True, help='Name of the table to update')

    # Parse command-line arguments
    args = parser.parse_args()

    # Run the function
    update_database(args.db, args.csv, args.table)
