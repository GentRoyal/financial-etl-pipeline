from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'v.env'))

# Load the .env file manually from /app/v.env
load_dotenv("/app/v.env")

class Settings(BaseSettings):
    api_key: str
    db_name: str
    model_dir: str

settings = Settings()


