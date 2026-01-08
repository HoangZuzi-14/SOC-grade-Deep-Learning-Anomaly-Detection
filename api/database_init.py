"""
Database initialization script
Run this to create/initialize the database
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.database import init_db, engine, Base
from sqlalchemy import inspect

def main():
    """Initialize database"""
    print("[*] Initializing database...")
    init_db()
    
    # Check if tables were created
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    print(f"[+] Database initialized successfully!")
    print(f"[+] Created tables: {', '.join(tables)}")
    print("\n[*] You can now start the API server:")
    print("    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")

if __name__ == "__main__":
    main()
