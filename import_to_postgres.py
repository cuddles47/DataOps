import os
import glob
import logging
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import numpy as np
from datetime import datetime
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"postgres_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# PostgreSQL connection parameters from environment variables
PG_CONFIG = {
    'host': os.getenv('POSTGRES_HOST'),
    'port': int(os.getenv('POSTGRES_PORT')),  # Convert to int
    'database': os.getenv('POSTGRES_DB'),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD')
}

# Directory with CSV files
DATA_DIR = os.getenv('DATA_DIR', '.Data/')

def connect_to_postgres():
    """Establish connection to PostgreSQL database with better error handling"""
    try:
        logger.info(f"Connecting to PostgreSQL at {PG_CONFIG['host']}:{PG_CONFIG['port']}")
        logger.info(f"Database: {PG_CONFIG['database']}, User: {PG_CONFIG['user']}")
        
        # Add connection timeout to prevent hanging
        conn = psycopg2.connect(
            **PG_CONFIG,
            connect_timeout=10  # 10 second timeout
        )
        
        logger.info("Successfully connected to PostgreSQL")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Connection error: {str(e)}")
        
        # Check common issues
        if "could not connect to server" in str(e):
            logger.error("PostgreSQL server may not be running or is not accessible at the specified host/port")
            logger.error("If using Docker, verify the container is running and port mapping is correct")
            logger.error("Try: docker ps | grep postgres")
        elif "password authentication failed" in str(e):
            logger.error("Authentication failed - verify your username and password")
        elif "database" in str(e) and "does not exist" in str(e):
            logger.error(f"Database '{PG_CONFIG['database']}' does not exist")
        
        raise
    except Exception as e:
        logger.error(f"Unexpected error connecting to PostgreSQL: {str(e)}")
        raise

def get_csv_files():
    """Get list of all CSV files in the data directory, excluding cleaned_canada.csv"""
    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory {DATA_DIR} does not exist")
        raise FileNotFoundError(f"Directory {DATA_DIR} not found")
    
    csv_pattern = os.path.join(DATA_DIR, '*.csv')
    all_csv_files = glob.glob(csv_pattern)
    
    # Filter out cleaned_canada.csv files (which could include date-stamped versions)
    csv_files = [f for f in all_csv_files if 'cleaned_canada' not in os.path.basename(f).lower()]
    
    if not csv_files:
        logger.warning(f"No CSV files found in {DATA_DIR} (or all were excluded)")
        return []
    
    excluded_count = len(all_csv_files) - len(csv_files)
    logger.info(f"Found {len(csv_files)} CSV files to import (excluded {excluded_count} files)")
    
    return csv_files

def infer_column_types(df):
    """Infer PostgreSQL column types from pandas DataFrame"""
    pg_type_map = {
        'int64': 'INTEGER',
        'float64': 'NUMERIC',
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'TIMESTAMP',
        'object': 'TEXT',
    }
    
    column_types = {}
    for column in df.columns:
        dtype = str(df[column].dtype)
        
        # Check if it's a date column based on name
        if 'date' in column.lower() and dtype == 'object':
            # Try to convert to datetime
            try:
                pd.to_datetime(df[column])
                column_types[column] = 'DATE'
                continue
            except:
                pass
        
        column_types[column] = pg_type_map.get(dtype, 'TEXT')
    
    return column_types

def create_table_if_not_exists(conn, table_name, df):
    """Create a table based on DataFrame structure if it doesn't exist"""
    column_types = infer_column_types(df)
    
    # Create columns definition
    columns_def = []
    for column in df.columns:
        col_type = column_types.get(column, 'TEXT')
        columns_def.append(f'"{column}" {col_type}')
    
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        {', '.join(columns_def)}
    );
    """
    
    try:
        logger.info(f"Creating table {table_name} if not exists")
        with conn.cursor() as cursor:
            cursor.execute(create_table_query)
        conn.commit()
        logger.info(f"Table {table_name} created or already exists")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating table {table_name}: {str(e)}")
        raise

def insert_data(conn, table_name, df):
    """Insert data from DataFrame into PostgreSQL table"""
    # Replace NaN with None for proper SQL NULL values
    df = df.replace({np.nan: None})
    
    # Prepare column list and data rows
    columns = list(df.columns)
    values = [tuple(row) for row in df.values]
    
    # Prepare insert query
    insert_query = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
        sql.Identifier(table_name),
        sql.SQL(', ').join(map(sql.Identifier, columns))
    )
    
    try:
        logger.info(f"Inserting {len(df)} rows into {table_name}")
        with conn.cursor() as cursor:
            execute_values(cursor, insert_query, values)
        conn.commit()
        logger.info(f"Successfully inserted {len(df)} rows into {table_name}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error inserting data into {table_name}: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def handle_duplicate_columns(df):
    """Handle duplicate column names by appending a number"""
    columns = df.columns.tolist()
    counts = {}
    new_columns = []
    
    for i, col in enumerate(columns):
        if col in counts:
            counts[col] += 1
            new_columns.append(f"{col}_{counts[col]}")
        else:
            counts[col] = 0
            new_columns.append(col)
    
    # Rename columns to the new unique names
    df.columns = new_columns
    
    return df

def import_csv_to_postgres(csv_file, conn):
    """Import a single CSV file to PostgreSQL"""
    try:
        # Get table name from file name (without extension and path)
        table_name = os.path.splitext(os.path.basename(csv_file))[0].lower()
        
        logger.info(f"Processing file: {csv_file} -> table: {table_name}")
        
        # Read CSV file
        df = pd.read_csv(csv_file, low_memory=False)
        logger.info(f"Read {len(df)} rows from {csv_file}")
        
        # Check for and rename duplicate columns
        orig_cols = len(df.columns)
        df = handle_duplicate_columns(df)
        if len(df.columns) == orig_cols:
            logger.info(f"No duplicate column names found in {table_name}")
        else:
            logger.info(f"Renamed duplicate columns in {table_name}")
        
        # Create table if it doesn't exist
        create_table_if_not_exists(conn, table_name, df)
        
        # Insert data into table
        insert_data(conn, table_name, df)
        
        return True
    except Exception as e:
        logger.error(f"Error importing {csv_file}: {str(e)}")
        return False

def main():
    """Main function to import all CSV files to PostgreSQL"""
    try:
        # Get list of CSV files
        csv_files = get_csv_files()
        if not csv_files:
            return
        
        # Connect to PostgreSQL
        conn = connect_to_postgres()
        
        # Process each CSV file
        successful_imports = 0
        for csv_file in csv_files:
            if import_csv_to_postgres(csv_file, conn):
                successful_imports += 1
        
        # Close connection
        conn.close()
        
        logger.info(f"Import complete. Successfully imported {successful_imports} out of {len(csv_files)} files.")
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
