
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from bs4 import BeautifulSoup
import os
import sys
import time
import json
import logging
import requests
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer
import string
from typing import Dict, Any, List


from openai import OpenAI
from colorama import Fore, Style, init

import os
import logging
from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize colorama
init(autoreset=True)




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SSL Context for NLTK download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    
    
client = OpenAI(base_url="http://localhost:11434/v1", api_key="lm-studio")
llm_model = "minicpm-v:8b-2.6-q6_K"
# SSL Context for NLTK download

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Set up stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Constants for rate limiting
RATE_LIMIT = 2  # requests per second
RATE_LIMIT_PERIOD = 1  # second



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

class Phase1:
    def __init__(self):
        logger.info("Initializing ComprehensiveHypothesisSystem...")
        self.documents = []
        self.vectorizer = TfidfVectorizer()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.stop_words = set(stopwords.words('english'))
        self.dataset_path = r"C:\Users\wes\AutoProtGenerationSystem\Phase_1\technical_description_molecular_database"
        self.querier = DatabaseQuerier(self.dataset_path)
        self.db = self.querier.db
        self.rate_limit = RATE_LIMIT
        self.rate_limit_period = RATE_LIMIT_PERIOD
        self.client = client
        self.llm_model = llm_model
        
        logger.info("System initialized.")

    def generate_initial_hypothesis(self, user_input: str) -> str:
        """Generate an initial hypothesis based on the user-defined topic."""
        logger.info(f"Generating initial hypothesis for input: {user_input}")
        # Placeholder for hypothesis generation logic
        # Replace with actual implementation if necessary
        initial_hypothesis = f"Initial hypothesis based on '{user_input}'."
        logger.info(f"Initial hypothesis generated: {initial_hypothesis}")
        return initial_hypothesis

    def extract_keywords(self, text: str, num_keywords: int = 1) -> List[str]:
        """Extract keywords from text using TF-IDF and lemmatization."""
        logger.info("Extracting keywords from initial hypothesis.")
        # Tokenize and lowercase
        tokens = word_tokenize(text.lower())
        # Remove punctuation and stopwords
        tokens = [token for token in tokens if token not in string.punctuation and token not in self.stop_words]
        # Lemmatize tokens
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # Join tokens back into a string
        processed_text = ' '.join(lemmatized_tokens)
        # Use TF-IDF to identify important words
        vectorizer = TfidfVectorizer(max_features=num_keywords)
        tfidf_matrix = vectorizer.fit_transform([processed_text])
        # Get feature names and their TF-IDF scores
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        # Sort words by TF-IDF score
        word_scores = list(zip(feature_names, tfidf_scores))
        word_scores.sort(key=lambda x: x[1], reverse=True)
        # Extract keywords
        keywords = [word for word, score in word_scores[:num_keywords]]
        logger.info(f"Extracted keywords: {keywords}")
        return keywords

    def search_academic_sources(self, keywords: List[str], max_results: int = 1) -> List[Dict[str, Any]]:
        """Search multiple academic sources based on keywords."""
        logger.info(f"Searching academic sources for keywords: {keywords}")
        pubmed_articles = self.search_pubmed(keywords, max_results)
        time.sleep(self.rate_limit_period)
        arxiv_articles = self.search_arxiv(keywords, max_results)
        time.sleep(self.rate_limit_period)
        # Combine and return all articles
        all_articles = pubmed_articles + arxiv_articles
        logger.info(f"Total articles found: {len(all_articles)}")
        return all_articles

    def search_pubmed(self, keywords: List[str], max_results: int = 1) -> List[Dict[str, Any]]:
        """Search PubMed for articles based on keywords."""
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        all_articles = []
        for keyword in keywords:
            params = {
                "db": "pubmed",
                "term": keyword,
                "retmax": max_results,
                "retmode": "json"
            }
            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                article_ids = data.get('esearchresult', {}).get('idlist', [])
                if not article_ids:
                    logger.warning(f"No PubMed articles found for the keyword: {keyword}")
                    continue
                fetch_params = {
                    "db": "pubmed",
                    "id": ",".join(article_ids),
                    "retmode": "xml",
                    "rettype": "abstract"
                }
                fetch_response = requests.get(fetch_url, params=fetch_params)
                fetch_response.raise_for_status()
                soup = BeautifulSoup(fetch_response.content, "xml")  # Use built-in xml parser
                for article in soup.find_all("PubmedArticle"):
                    metadata = {
                        "title": article.find("ArticleTitle").text if article.find("ArticleTitle") else "N/A",
                        "abstract": " ".join([abstract.text for abstract in article.find_all("AbstractText")]),
                        "authors": [author.find("LastName").text for author in article.find_all("Author") if author.find("LastName")],
                        "journal": article.find("Title").text if article.find("Title") else "N/A",
                        "doi": article.find("ELocationID", {"EIdType": "doi"}).text if article.find("ELocationID", {"EIdType": "doi"}) else "N/A",
                        "source": "PubMed",
                        "keyword": keyword
                    }
                    all_articles.append(metadata)
                time.sleep(1 / self.rate_limit)
            except requests.RequestException as e:
                logger.error(f"Error fetching PubMed articles for keyword '{keyword}': {e}")
        logger.info(f"Total PubMed articles found: {len(all_articles)}")
        return all_articles

    def search_arxiv(self, keywords: List[str], max_results: int = 1) -> List[Dict[str, Any]]:
        """Search arXiv for articles based on keywords."""
        base_url = "http://export.arxiv.org/api/query"
        query = "+AND+".join(keywords)
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "lxml-xml")
            articles = []
            for entry in soup.find_all("entry"):
                metadata = {
                    "title": entry.find("title").text if entry.find("title") else "N/A",
                    "abstract": entry.find("summary").text if entry.find("summary") else "N/A",
                    "authors": [author.find("name").text for author in entry.find_all("author")],
                    "journal": "arXiv",
                    "doi": entry.find("id").text if entry.find("id") else "N/A",
                    "source": "arXiv"
                }
                articles.append(metadata)
            time.sleep(1 / self.rate_limit)
            logger.info(f"Total arXiv articles found: {len(articles)}")
            return articles
        except requests.RequestException as e:
            logger.error(f"Error fetching arXiv articles: {e}")
            return []

    def extract_key_information(self, text: str) -> Dict[str, List[str]]:
        """Extract key information such as methods and results from text."""
        # Placeholder for key information extraction
        # You can replace this with actual NLP if necessary
        methods = []
        results = []
        # Simple keyword-based extraction
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            if "method" in sentence.lower():
                methods.append(sentence)
            if "result" in sentence.lower():
                results.append(sentence)
        return {
            "METHODS": methods,
            "RESULTS": results
        }

    def summarize_text(self, text: str) -> str:
        """Summarize the given text using OpenAI API."""
        logger.info("Summarizing text.")
        prompt = f"Summarize the following text in a concise manner:\n\n{text}"
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a text summarizer."},
                {"role": "user", "content": prompt}
            ],
            model=self.llm_model,
            stream=False,
        )
        summary = response.choices[0].message.content
        logger.info("Text summarized.")
        return summary

    def generate_description(self, methods: List[str], results: List[str]) -> str:
        """Generate a description based on methods and results."""
        logger.info("Generating description based on methods and results.")
        methods_text = ' '.join(methods)
        results_text = ' '.join(results)
        prompt = (
            f"Based on the following methods and results, generate a concise and clear description:\n\n"
            f"Methods:\n{methods_text}\n\nResults:\n{results_text}\n\n"
            "Provide a comprehensive summary that explains the significance and implications of these findings."
        )
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a scientific writer."},
                {"role": "user", "content": prompt}
            ],
            model=self.llm_model,
            stream=False,
        )
        description = response.choices[0].message.content
        logger.info("Description generated.")
        return description

    def process_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and summarize articles."""
        logger.info("Processing articles.")
        summarized_articles = []
        os.makedirs("automol/phase1/processed_articles", exist_ok=True)  # Ensure the directory exists
        for article in articles:
            abstract = article.get('abstract', '')
            if not abstract:
                continue
            key_info = self.extract_key_information(abstract)
            summary = self.summarize_text(abstract)
            generated_desc = self.generate_description(key_info['METHODS'], key_info['RESULTS'])
            summarized_article = {
                'title': article.get('title', 'N/A'),
                'abstract': abstract,
                'summary': summary,
                'extracted_info': key_info,
                'generated_description': generated_desc,
                'citation_info': {
                    'authors': article.get('authors', []),
                    'year': article.get('year', ''),
                    'title': article.get('title', 'N/A'),
                    'journal': article.get('journal', 'N/A'),
                    'doi': article.get('doi', 'N/A'),
                    'publication_date': article.get('publication_date', 'N/A')
                }
            }
            summarized_articles.append(summarized_article)
            # Save the summarized article to a file
            article_id = article.get('id', f"article_{len(summarized_articles)}")
            with open(f"processed_articles/{article_id}.json", 'w') as f:
                json.dump(summarized_article, f, indent=2)
        logger.info(f"Processed {len(summarized_articles)} articles.")
        return summarized_articles

    def extract_json_between_markers(self, content: str) -> Dict[str, Any] | None:
        protein_start_marker = "<technical_description_protein>"
        protein_end_marker = "</technical_description_protein>"
        ligand_start_marker = "<technical_description_ligand>"
        ligand_end_marker = "</technical_description_ligand>"

        try:
            # Extract protein description
            protein_start = content.index(protein_start_marker) + len(protein_start_marker)
            protein_end = content.index(protein_end_marker, protein_start)
            protein_description = content[protein_start:protein_end].strip()
            print(f"DEBUG: Extracted protein description: {protein_description[:100]}...")  # Print first 100 chars

            # Extract ligand description
            ligand_start = content.index(ligand_start_marker) + len(ligand_start_marker)
            ligand_end = content.index(ligand_end_marker, ligand_start)
            ligand_description = content[ligand_start:ligand_end].strip()
            print(f"DEBUG: Extracted ligand description: {ligand_description[:100]}...")  # Print first 100 chars

            result = {
                "technical_description_protein": protein_description,
                "technical_description_ligand": ligand_description
            }
            print(f"DEBUG: Extracted JSON: {result}")
            return result
        except ValueError as ve:
            logger.error(f"DEBUG: ValueError in extract_json_between_markers: {str(ve)}")
            print(f"DEBUG: Content causing error: {content}")
            return None
        except Exception as e:
            logger.error(f"DEBUG: Error extracting descriptions: {str(e)}")
            print(f"DEBUG: Content causing error: {content}")
            return None
        
        
    def generate_technical_descriptions(self, research_descriptions):
        model = "deepseek-coder-v2:16b-lite-instruct-q6_K"
        description_system_prompt = f"""You are an expert in molecular biology and chemistry, tasked with generating technical descriptions for molecule generation.Take the excerpt of text  and from it create a very technical, concise technical description that will be used to generate the molecule you describe. Generate a protein and ligand pair that are designed to interact with each other based on the user prompt.
        Please format y our response exactly as follows:

        <technical_description_protein>
        [Your detailed technical instruction for protein generation goes here]
        </technical_description_protein>

        <technical_description_ligand>
        [Your matching description for ligand generation designed to interact with the generated protein goes here]
        </technical_description_ligand>

        Make sure to replace the text in square brackets with your actual descriptions, and do not include the square brackets in your response.

        ONLY GENERATE ONE AT A TIME OTHERWISE SMALL BABY TINY INNOCENT KITTENS WILL DIE DO NOT RESPOND WITH ANY OTHER TEXT
        RESEARCH DATA:"""
        
        messages = [
            {"role": "system", "content": description_system_prompt},
            {"role": "user", "content": research_descriptions}
        ]
        
        print(f"DEBUG: Sending request to LLM with messages: {messages}")
        response = self.client.chat.completions.create(
            messages=messages,
            model=model,
            stream=False,
        )
        response_content = response.choices[0].message.content
        print(f"DEBUG: Raw LLM Response: {response_content}")
        
        json_output = self.extract_json_between_markers(response_content)
        print(f"DEBUG: Extracted JSON Output: {json_output}")
        assert json_output is not None, "Failed to extract JSON from LLM output"
        print(f"DEBUG: Final JSON output: {json_output}")
        
        return json_output
    
    def prepare_technical_description(self, refined_hypothesis: str, research_data: List[Dict[str, Any]]) -> str:
        """Prepare the final technical description."""
        logger.info("Preparing technical description.")
        research_descriptions = "\n".join([article['generated_description'] for article in research_data])
        print(f"DEBUG: Combined research descriptions: {research_descriptions[:200]}...")  # Print first 200 chars
        generated_technical_description = self.generate_technical_descriptions(research_descriptions)
        print(f"DEBUG: Generated technical description: {generated_technical_description}")
        technical_description = (
            f"<technical_description_protein>\n{refined_hypothesis}\n</technical_description_protein>\n\n"
            f"<technical_description_ligand>\n{research_descriptions}\n</technical_description_ligand>"
        )
        print(f"DEBUG: Final technical description: {technical_description[:200]}...")  # Print first 200 chars
        logger.info("Technical description prepared.")
        return technical_description

import time
import random

def run_pipeline(self, user_input: str, max_attempts=5):
    """Run the entire pipeline with a user input."""
    logger.info("Starting the pipeline.")
    
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"Attempt {attempt} of {max_attempts}")
            
            # Generate Initial Hypothesis
            initial_hypothesis = self.generate_initial_hypothesis(user_input)
            print(f"DEBUG: Initial Hypothesis: {initial_hypothesis}")
            
            # Extract Keywords
            keywords = self.extract_keywords(initial_hypothesis)
            print(f"DEBUG: Extracted Keywords: {keywords}")
            
            # Research Articles
            articles = self.search_academic_sources(keywords, max_results=1)
            if not articles:
                logger.warning("No relevant articles found to refine hypotheses.")
                continue  # Try again
            print(f"DEBUG: Found Articles: {len(articles)}")
            
            # Process Articles
            summarized_articles = self.process_articles(articles)
            print(f"DEBUG: Processed Articles: {len(summarized_articles)}")
            
            # Query Database
            db_results = self.querier.query_database(user_input)
            print(f"DEBUG: Database Query Results: {len(db_results)}")
            
            # Prepare Technical Description
            research_descriptions = self.prepare_technical_description(initial_hypothesis, research_data=summarized_articles)
            print(f"DEBUG: Prepared Research Descriptions: {research_descriptions[:200]}...")  # Print first 200 chars
            
            # Output Technical Description
            technical_description = self.generate_technical_descriptions(research_descriptions)
            print(Fore.CYAN + "\nDEBUG: Final Technical Description for Sequence Generation:")
            print(Fore.GREEN + json.dumps(technical_description, indent=2))  # Convert dict to JSON string for printing
            
            # Create the directory if it doesn't exist
            directory = "technical_descriptions"
            os.makedirs(directory, exist_ok=True)

            # Save the final technical description to a json file
            file_path = os.path.join(directory, "technical_description.json")
            with open(file_path, "w") as f:
                json.dump(technical_description, f, indent=4)
            print(f"DEBUG: Saved technical description to {file_path}")

            # Verify if the file was created
            if os.path.exists(file_path):
                print(f"DEBUG: File {file_path} was successfully created.")
                
                # Print Database Results
                print(Fore.CYAN + "\nDEBUG: Database Results:")
                for i, result in enumerate(db_results, 1):
                    print(Fore.YELLOW + f"\nDEBUG: Result {i}:")
                    print(Fore.WHITE + f"Content: {result.page_content[:200]}...")  # Print first 200 characters
                    print(Fore.MAGENTA + f"Metadata: {result.metadata}")
                
                return technical_description  # Successfully completed, exit the function
            else:
                print(f"ERROR: Failed to create file {file_path}.")
                raise Exception("File creation failed")

        except Exception as e:
            print(f"ERROR: An error occurred during attempt {attempt}: {str(e)}")
            if attempt < max_attempts:
                wait_time = random.uniform(1, 5)  # Random wait between 1 and 5 seconds
                print(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                print("Max attempts reached. Pipeline failed.")
                return None

    print("Pipeline completed all attempts without success.")
    return None