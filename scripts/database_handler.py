import sqlite3
import pandas as pd
from config import settings
import os

class MyDBRepo:
    """
    A class for storing and retrieving data using SQLite.
    """

    def __init__(self):
        """
        Initializes a connection to the SQLite database.
        Sets the connection to None if an error occurs.
        """
        self.db_name = settings.db_name

        absolute_path = os.path.abspath(__file__)
        directory_name = os.path.dirname(absolute_path)
        parent_name = os.path.dirname(directory_name)
        self.path = os.path.join(parent_name, 'data')
        
        try:
            self.conn = sqlite3.connect(f"{self.path}/{self.db_name}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            self.conn = None

    def to_table(self, df, symbol, if_exists = 'replace'):
        """
        Saves a DataFrame to a table in the SQLite database.

        Parameters:
        df (pd.DataFrame): The DataFrame to be saved.
        symbol (str): The name of the table.
        if_exists (str): Behavior when the table already exists. Defaults to 'replace'.

        Returns:
        None
        """
        if self.conn is None:
            print("No database connection.")
            return

        try:
            df.to_sql(name = symbol, con = self.conn, if_exists = if_exists, index = False)
            print(f"Data inserted into table '{symbol}' successfully.")
            
        except ValueError as ve:
            print(f"Value error inserting data into table '{symbol}': {ve}")
            
        except sqlite3.Error as sql_err:
            print(f"SQLite error inserting data into table '{symbol}': {sql_err}")
            
        except Exception as e:
            print(f"Unexpected error inserting data into table '{symbol}': {e}")

    def from_table(self, symbol):
        """
        Retrieves data from a specified table in the SQLite database.

        Parameters:
        symbol (str): The name of the table to query.

        Returns:
        pd.DataFrame: The queried data as a DataFrame, or an empty DataFrame if an error occurs.
        """
        if self.conn is None:
            print("No database connection.")
            return pd.DataFrame()

        try:
            query = f"SELECT * FROM {symbol} LIMIT 500"
            df = pd.read_sql(query, self.conn)
            
            print(f"Data retrieved from table '{symbol}' successfully.")
            df.to_csv(f'{self.path}/processed/{symbol}.csv', index = False)
            
            return df
            
        except pd.io.sql.DatabaseError as db_err:
            print(f"Database error reading from table '{symbol}': {db_err}")
            
        except Exception as e:
            print(f"Unexpected error reading from table '{symbol}': {e}")

        return pd.DataFrame()
