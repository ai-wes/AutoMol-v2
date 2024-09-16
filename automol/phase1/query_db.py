
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import os
import logging
from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseQuerier:
    def __init__(self, dataset_path):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.dataset_path = dataset_path
        self.db = self.load_db(self.dataset_path)

    def load_db(self, dataset_path):
        try:
            return DeepLake(dataset_path=dataset_path, embedding=self.embeddings, read_only=True)
        except Exception as e:
            logger.error(f"Error loading database from {dataset_path}: {str(e)}")
            return None

    def retrieve_from_db(self, query):
        if self.db is None:
            logger.error("Database is not loaded.")
            return []

        retriever = self.db.as_retriever()
        retriever.search_kwargs["distance_metric"] = "cos"
        retriever.search_kwargs["fetch_k"] = 5
        retriever.search_kwargs["k"] = 5
        
        try:
            db_context = retriever.invoke(query)
            logger.info(f"Retrieved {len(db_context)} results from the database")
            return db_context
        except Exception as e:
            logger.error(f"Error retrieving from database: {str(e)}")
            return []

    def query_database(self, query):
        return self.retrieve_from_db(query)

if __name__ == "__main__":
    dataset_path = r"C:\Users\wes\AutoProtGenerationSystem\Phase_1\technical_description_molecular_database"

    querier = DatabaseQuerier(dataset_path)
    
    query = "telomerase"
    results = querier.query_database(query)

    print("\nDatabase Results:")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Content: {result.page_content[:5000]}...")  # Print first 200 characters
        print(f"Metadata: {result.metadata}")