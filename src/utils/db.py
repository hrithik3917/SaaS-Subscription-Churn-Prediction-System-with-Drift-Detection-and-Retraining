"""
db.py - Database Connection Manager
====================================
WHY THIS FILE EXISTS:
    Every script that talks to PostgreSQL needs a "connection".
    Instead of writing connection code in every file, we write it
    ONCE here, and every other script just imports `engine`.
    
    Think of `engine` like a phone line to your database - 
    once it's set up, anyone can use it to make calls (queries).
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# Load environment variables from .env file
load_dotenv()

# Build the database connection URL safely
DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT")),
    database=os.getenv("DB_NAME"),
)

# Create the engine (lazy - only connects when first query runs)
engine = create_engine(DATABASE_URL)


def test_connection():
    """Test if we can actually reach the database."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("✅ Database connection successful!")
            print(f"   Connected to: {os.getenv('DB_NAME')} @ {os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}")
            return True
    except Exception as e:
        print("❌ Database connection FAILED!")
        print(f"   Error: {e}")
        print("\n   Troubleshooting:")
        print("   1. Is PostgreSQL running?")
        print("   2. Does the 'saas_churn' database exist in pgAdmin?")
        print("   3. Are your .env credentials correct?")
        return False


if __name__ == "__main__":
    test_connection()