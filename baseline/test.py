from langchain_community.graphs import Neo4jGraph

from dotenv import load_dotenv
import os

load_dotenv()

NEO4J_URI=os.getenv('NEO4J_URI')
NEO4J_USER=os.getenv('NEO4J_USERNAME')
NEO4J_PASS=os.getenv('NEO4J_PASSWORD')
# NEO4J_DATABASE=os.getenv('NEO4J_DATABSE')


graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASS,
    refresh_schema=False
)

