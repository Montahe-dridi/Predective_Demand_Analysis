from sqlalchemy import create_engine
import urllib.parse

# Replace these with your actual credentials
DB_USER = 'sa1'
DB_PASSWORD = urllib.parse.quote_plus('monta@123@123')
DB_HOST = 'localhost'
DB_PORT = '1433'
DB_NAME_SOURCE = 'TRALIS_DATA'
DB_NAME_TARGET = 'TRALIS_DWH'


def get_source_engine():
    conn_str = f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME_SOURCE}?driver=ODBC+Driver+17+for+SQL+Server"
    return create_engine(conn_str)

def get_target_engine():
    conn_str = f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME_TARGET}?driver=ODBC+Driver+17+for+SQL+Server"
    return create_engine(conn_str)
