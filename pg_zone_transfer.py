import os
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
        logging.FileHandler(f"pg_zone_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Base PostgreSQL connection parameters from environment variables
BASE_PG_CONFIG = {
    'host': os.getenv('POSTGRES_HOST'),
    'port': int(os.getenv('POSTGRES_PORT')),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD')
}

# Source and target configurations reusing the base config
SOURCE_PG_CONFIG = {**BASE_PG_CONFIG, 'database': 'landing-zone'}
TARGET_PG_CONFIG = {**BASE_PG_CONFIG, 'database': 'formatted-zone'}

def connect_to_postgres(config, purpose):
    """Establish connection to PostgreSQL database with error handling"""
    try:
        logger.info(f"Connecting to {purpose} PostgreSQL at {config['host']}:{config['port']}")
        logger.info(f"Database: {config['database']}, User: {config['user']}")
        
        # Add connection timeout to prevent hanging
        conn = psycopg2.connect(
            **config,
            connect_timeout=10  # 10 second timeout
        )
        
        logger.info(f"Successfully connected to {purpose} PostgreSQL database")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to {purpose} PostgreSQL: {str(e)}")
        raise

def get_tables_from_source():
    """Get list of all tables in the source database"""
    try:
        conn = connect_to_postgres(SOURCE_PG_CONFIG, "source")
        cursor = conn.cursor()
        
        # Query to get all tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        
        tables = [table[0] for table in cursor.fetchall()]
        
        if not tables:
            logger.warning("No tables found in source database")
            return []
        
        logger.info(f"Found {len(tables)} tables in source database")
        cursor.close()
        conn.close()
        
        return tables
    except Exception as e:
        logger.error(f"Error getting tables from source database: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def clean_column_name(column_name):
    """Clean column names to be compatible with PostgreSQL"""
    # Replace spaces and special characters with underscores
    cleaned = column_name.lower().replace(' ', '_')
    cleaned = ''.join(c if c.isalnum() or c == '_' else '_' for c in cleaned)
    
    # Ensure name doesn't start with a number
    if cleaned[0].isdigit():
        cleaned = 'col_' + cleaned
        
    return cleaned

def handle_duplicate_columns(df):
    """Rename duplicate columns to make them unique for PostgreSQL"""
    counts = {}
    new_columns = []
    
    for col in df.columns:
        clean_col = clean_column_name(col)
        if clean_col in counts:
            counts[clean_col] += 1
            new_columns.append(f"{clean_col}_{counts[clean_col]}")
        else:
            counts[clean_col] = 0
            new_columns.append(clean_col)
    
    # Rename columns to the new unique names
    df.columns = new_columns
    
    return df

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
    
    # Clean column names
    clean_columns = {col: clean_column_name(col) for col in df.columns}
    df = df.rename(columns=clean_columns)
    
    # Create columns definition
    columns_def = []
    for column in df.columns:
        col_type = column_types.get(column, 'TEXT')
        columns_def.append(f"{column} {col_type}")
    
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
    # Clean column names
    clean_columns = {col: clean_column_name(col) for col in df.columns}
    df = df.rename(columns=clean_columns)
    
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

def read_table_from_source(table_name):
    """Read a table from the source database into a pandas DataFrame"""
    try:
        # Connect to source database
        conn = connect_to_postgres(SOURCE_PG_CONFIG, "source")
        
        # Read data into DataFrame
        query = f"SELECT * FROM {table_name}"
        logger.info(f"Reading data from table {table_name}")
        df = pd.read_sql(query, conn)
        
        logger.info(f"Read {len(df)} rows from {table_name}")
        conn.close()
        
        return df
    except Exception as e:
        logger.error(f"Error reading table {table_name} from source: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def transfer_table(table_name, target_conn):
    """Transfer a table from source database to target database"""
    try:
        # Read data from source
        df = read_table_from_source(table_name)
        
        if len(df) == 0:
            logger.warning(f"Table {table_name} is empty, skipping")
            return False
        
        # Handle potential duplicate column issues
        df = handle_duplicate_columns(df)
        
        # Create table in target database
        create_table_if_not_exists(target_conn, table_name, df)
        
        # Insert data into target table
        insert_data(target_conn, table_name, df)
        
        return True
    except Exception as e:
        logger.error(f"Error transferring table {table_name}: {str(e)}")
        return False

def main():
    """Main function to transfer tables from landing-zone to formatted-zone"""
    try:
        # Get list of tables from source database
        tables = get_tables_from_source()
        if not tables:
            return
        
        # Connect to target database
        target_conn = connect_to_postgres(TARGET_PG_CONFIG, "target")
        
        # Process each table
        successful_transfers = 0
        for table_name in tables:
            if transfer_table(table_name, target_conn):
                successful_transfers += 1
        
        # Close connection
        target_conn.close()
        
        logger.info(f"Transfer complete. Successfully transferred {successful_transfers} out of {len(tables)} tables.")
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
