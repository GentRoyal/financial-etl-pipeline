import sqlite3
import pandas as pd
import os

def connect_to_db():
    """
    Establishes a connection to the SQLite database.

    Returns:
    sqlite3.Connection: SQLite connection object or None if an error occurs.
    """
    db_name = "data.sqlite"
    try:
        path = os.path.abspath(os.path.join('airflow_projects', 'stock-etl-pipeline', 'data'))
        conn = sqlite3.connect(f"{path}/{db_name}")
        return conn
        
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def write_to_table(df, table_name, if_exists = 'replace'):
    """
    Writes a DataFrame to a specified table in the database.

    Parameters:
    conn (sqlite3.Connection): The SQLite connection.
    df (pd.DataFrame): DataFrame to write.
    table_name (str): Table name.
    if_exists (str): Behavior if table exists ('replace', 'append', etc.)

    Returns:
    None
    """
    conn = connect_to_db()
    try:
        df.to_sql(name = table_name, con = conn, if_exists = if_exists, index = False)
        print(f"Data inserted into table '{table_name}' successfully.")
        
    except ValueError as ve:
        print(f"Value error inserting into table '{table_name}': {ve}")
        
    except sqlite3.Error as sql_err:
        print(f"SQLite error inserting into table '{table_name}': {sql_err}")
        
    except Exception as e:
        print(f"Unexpected error inserting into table '{table_name}': {e}")

def read_from_table(table_name):
    """
    Reads data from a specified table in the database.

    Parameters:
    conn (sqlite3.Connection): The SQLite connection.
    table_name (str): Table name.

    Returns:
    pd.DataFrame: DataFrame with the result or empty DataFrame on failure.
    """
    conn = connect_to_db()
    path = os.path.abspath(os.path.join('airflow_projects', 'stock-etl-pipeline', 'data'))
    try:
        query = f"SELECT * FROM {table_name} LIMIT 500"
        df = pd.read_sql(query, conn)
        print(f"Data retrieved from table '{table_name}' successfully.")
        df.to_csv(f'{path}/processed/{table_name}.csv', index = False)
        
        return df
        
    except pd.io.sql.DatabaseError as db_err:
        print(f"Database error reading from table '{table_name}': {db_err}")
        
    except Exception as e:
        print(f"Unexpected error reading from table '{table_name}': {e}")

    return pd.DataFrame()