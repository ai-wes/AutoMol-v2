from typing import Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any
from tqdm import tqdm
from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.cluster import KMeans
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        """
        Initializes the Graph Neural Network with improved architecture.
        
        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden layers.
            output_dim (int): Dimension of output embeddings.
            dropout (float): Dropout rate.
        """
        super(GraphNeuralNetwork, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn4 = BatchNorm(hidden_dim)
        
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of the GNN.
        
        Args:
            data (Data): A batch of graph data.
        
        Returns:
            torch.Tensor: Output embeddings.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x

class OmicsRAGPipeline:
    def __init__(self, vector_store_path: str):
        """
        Initializes the Omics RAG Pipeline.
        
        Args:
            vector_store_path (str): Path to the vector store.
        """
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store_path = self._sanitize_path(vector_store_path)
        self.db = None
        self.qa_chain = None
        self.gnn_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("OmicsRAGPipeline initialized.")
        logger.info(f"Vector store path: {self.vector_store_path}")

    def _sanitize_path(self, path: str) -> str:
        """
        Sanitizes the provided file path.
        
        Args:
            path (str): Original file path.
        
        Returns:
            str: Sanitized file path.
        """
        path = path.replace("\\", "/")
        path = path.split(":")[-1]
        path = path.lstrip("/")
        sanitized_path = os.path.abspath(path)
        logger.debug(f"Sanitized path: {sanitized_path}")
        return sanitized_path

    def load_vector_store(self):
        """
        Loads the vector store using DeepLake.
        """
        logger.info("Loading vector store...")
        logger.info(f"Vector store path: {self.vector_store_path}")
        self.db = DeepLake(dataset_path=self.vector_store_path, embedding=self.embeddings, read_only=True)
        logger.info("Vector store loaded successfully.")

    def create_rag_pipeline(self):
        """
        Creates the Retrieval-Augmented Generation (RAG) pipeline.
        """
        if self.db is None:
            raise ValueError("Vector store is not loaded. Call load_vector_store() first.")

        retriever = self.db.as_retriever(search_kwargs={"k": 5})

        custom_prompt_template = """
        You are an AI assistant specialized in omics data analysis. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. {context}
        Question: {question}

        Answer:"""

        PROMPT = PromptTemplate(
            template=custom_prompt_template, input_variables=["question", "context"]
        )

        llm = ChatOllama(
            model="mathstral:7b-v0.1-q6_K",
            temperature=0.2,
            max_tokens=512,
            top_p=0.5,
        )
        logger.info("Creating RAG pipeline...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        logger.info("RAG pipeline created successfully.")

    def initialize_gnn(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        """
        Initializes the GNN model.
        
        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden layers.
            output_dim (int): Dimension of output embeddings.
            dropout (float): Dropout rate.
        """
        self.gnn_model = GraphNeuralNetwork(input_dim, hidden_dim, output_dim, dropout).to(self.device)
        logger.info("GNN model initialized and moved to device.")

    def train_gnn(self, train_data: List[Data], val_data: List[Data], epochs: int = 50, patience: int = 10):
        """
        Trains the GNN model with early stopping and learning rate scheduling.
        
        Args:
            train_data (List[Data]): Training dataset.
            val_data (List[Data]): Validation dataset.
            epochs (int): Maximum number of training epochs.
            patience (int): Patience for early stopping.
        """
        if self.gnn_model is None:
            raise ValueError("GNN model is not initialized. Call initialize_gnn() first.")

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

        optimizer = Adam(self.gnn_model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        criterion = torch.nn.MSELoss()  # Adjust based on your task

        best_val_loss = float('inf')
        counter = 0

        for epoch in range(1, epochs + 1):
            self.gnn_model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}/{epochs}"):
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = self.gnn_model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gnn_model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch}: Average Training Loss: {avg_train_loss:.4f}")

            # Validation
            self.gnn_model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = self.gnn_model(batch)
                    val_loss += criterion(out, batch.y).item()

            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Epoch {epoch}: Average Validation Loss: {avg_val_loss:.4f}")

            # Scheduler step
            scheduler.step(avg_val_loss)

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                torch.save(self.gnn_model.state_dict(), 'best_gnn_model.pth')
                logger.info(f"Epoch {epoch}: Validation loss improved. Model saved.")
            else:
                counter += 1
                logger.info(f"Epoch {epoch}: No improvement in validation loss.")
                if counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break

    def analyze_protein_sequence(self, protein_sequence: str) -> np.ndarray:
        """
        Analyzes a protein sequence using the trained GNN model.
        
        Args:
            protein_sequence (str): Protein sequence to analyze.
        
        Returns:
            np.ndarray: Embedding of the protein sequence.
        """
        if self.gnn_model is None:
            raise ValueError("GNN model is not trained. Call train_gnn() first.")

        graph_data = self.protein_to_graph(protein_sequence)

        self.gnn_model.eval()
        graph_data = graph_data.to(self.device)
        with torch.no_grad():
            embedding = self.gnn_model(graph_data)

        return embedding.cpu().numpy()

    def protein_to_graph(self, protein_sequence: str) -> Data:
        """
        Converts a protein sequence to a graph data object.
        
        Args:
            protein_sequence (str): Protein sequence.
        
        Returns:
            Data: Graph data object.
        """
        amino_acids = list(protein_sequence)
        num_nodes = len(amino_acids)

        # Create node features
        node_features = [self.aa_to_features(aa) for aa in amino_acids]
        x = torch.tensor(node_features, dtype=torch.float)

        # Create edges (connecting each amino acid to its neighbors)
        edge_index = []
        for i in range(num_nodes):
            if i > 0:
                edge_index.append([i - 1, i])
            if i < num_nodes - 1:
                edge_index.append([i, i + 1])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Create a dummy target (if necessary for inference, this can be removed)
        y = torch.tensor([0.0], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, y=y)

    def aa_to_features(self, amino_acid: str) -> List[float]:
        """
        Converts an amino acid to its feature vector.
        
        Args:
            amino_acid (str): Single-letter amino acid code.
        
        Returns:
            List[float]: Feature vector.
        """
        aa_properties = {
            'A': [89.09, 6.00, 1.8, 0.0, 88.6, 0.0, 0.0, 1.0, 0.62],
            'R': [174.20, 10.76, -4.5, 1.0, 173.4, 1.0, 0.0, 0.0, 0.64],
            'N': [132.12, 5.41, -3.5, 0.0, 114.1, 0.5, 0.0, 0.0, 0.49],
            'D': [133.10, 2.77, -3.5, -1.0, 111.1, 0.5, 0.0, 0.0, 0.48],
            'C': [121.15, 5.07, 2.5, 0.0, 108.5, 0.0, 0.0, 0.0, 0.62],
            'E': [147.13, 3.22, -3.5, -1.0, 138.4, 0.5, 0.0, 0.0, 0.54],
            'Q': [146.15, 5.65, -3.5, 0.0, 143.8, 0.5, 0.0, 0.0, 0.56],
            'G': [75.07, 5.97, -0.4, 0.0, 60.1, 0.0, 0.0, 1.0, 0.48],
            'H': [155.16, 7.59, -3.2, 0.0, 153.2, 0.5, 0.0, 0.0, 0.61],
            'I': [131.17, 6.02, 4.5, 0.0, 166.7, 0.0, 0.0, 0.0, 0.73],
            'L': [131.17, 5.98, 3.8, 0.0, 166.7, 0.0, 0.0, 0.0, 0.69],
            'K': [146.19, 9.74, -3.9, 1.0, 168.6, 1.0, 0.0, 0.0, 0.52],
            'M': [149.21, 5.74, 1.9, 0.0, 162.9, 0.0, 0.0, 0.0, 0.70],
            'F': [165.19, 5.48, 2.8, 0.0, 189.9, 0.0, 1.0, 0.0, 0.80],
            'P': [115.13, 6.30, -1.6, 0.0, 112.7, 0.0, 0.0, 0.0, 0.36],
            'S': [105.09, 5.68, -0.8, 0.0, 89.0, 0.5, 0.0, 0.0, 0.41],
            'T': [119.12, 5.60, -0.7, 0.0, 116.1, 0.5, 0.0, 0.0, 0.48],
            'W': [204.23, 5.89, -0.9, 0.0, 227.8, 0.0, 1.0, 0.0, 0.85],
            'Y': [181.19, 5.66, -1.3, 0.0, 193.6, 0.5, 1.0, 0.0, 0.76],
            'V': [117.15, 5.96, 4.2, 0.0, 140.0, 0.0, 0.0, 0.0, 0.64],
        }
        # Features: [MW, pKa, hydrophobicity, charge, volume, polarity, aromaticity, flexibility, alpha-helix propensity]
        return aa_properties.get(amino_acid.upper(), [0.0]*9)  # Default to [0.0]*9 for unknown amino acids

    def query(self, question: str) -> Dict[str, Any]:
        """
        Queries the RAG pipeline with the given question.
        
        Args:
            question (str): The question to query.
        
        Returns:
            Dict[str, Any]: The result containing the answer and source documents.
        """
        if self.qa_chain is None:
            raise ValueError("RAG pipeline is not created. Call create_rag_pipeline() first.")
        logger.info(f"Processing question: {question}")
        result = self.qa_chain({"query": question, "context": ""})
        logger.info("RAG query completed.")
        return result

def generate_protein_sequence(prompt: str, max_length: int = 512, min_length: int = 100) -> str:
    """
    Generates a protein sequence using a pre-trained language model.
    
    Args:
        prompt (str): The prompt to generate the protein sequence.
        max_length (int): Maximum length of the generated sequence.
        min_length (int): Minimum length of the generated sequence.
    
    Returns:
        str: Generated protein sequence.
    """
    model_name = "nferruz/ProtGPT2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Protein generation model loaded and moved to device.")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        min_length=min_length,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.2
    )

    generated_sequence = tokenizer.decode(output[0], skip_special_tokens=True)
    protein_sequence = generated_sequence[len(prompt):].strip()
    logger.info("Protein sequence generated successfully.")
    return protein_sequence

def load_and_preprocess_protein_data(file_path: str) -> tuple[List[Data], List[Data]]:
    """
    Loads and preprocesses protein data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing protein data.
    
    Returns:
        Tuple[List[Data], List[Data]]: Training and validation datasets.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Protein data file not found at: {file_path}")

    df = pd.read_csv(file_path)
    logger.info(f"Loaded protein data from {file_path} with {len(df)} records.")

    graph_data = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Preprocessing protein data"):
        sequence = row['sequence']
        function = row['function']  # Using function as a label

        x, edge_index = protein_to_graph_features(sequence)
        label = function if isinstance(function, (int, float)) else len(str(function))
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float))
        graph_data.append(data)

    train_data, val_data = train_test_split(graph_data, test_size=0.2, random_state=42)
    logger.info(f"Data split into {len(train_data)} training and {len(val_data)} validation samples.")
    return train_data, val_data

def protein_to_graph_features(sequence: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a protein sequence to graph features.
    
    Args:
        sequence (str): Protein sequence.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Node features and edge indices.
    """
    amino_acids = list(sequence)
    num_nodes = len(amino_acids)

    node_features = [aa_to_features(aa) for aa in amino_acids]
    x = torch.tensor(node_features, dtype=torch.float)

    edge_index = []
    for i in range(num_nodes):
        if i > 0:
            edge_index.append([i - 1, i])
        if i < num_nodes - 1:
            edge_index.append([i, i + 1])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return x, edge_index

def aa_to_features(amino_acid: str) -> List[float]:
    """
    Converts an amino acid to its feature vector.
    
    Args:
        amino_acid (str): Single-letter amino acid code.
    
    Returns:
        List[float]: Feature vector.
    """
    aa_properties = {
        'A': [89.09, 6.00, 1.8, 0.0, 88.6, 0.0, 0.0, 1.0, 0.62],
        'R': [174.20, 10.76, -4.5, 1.0, 173.4, 1.0, 0.0, 0.0, 0.64],
        'N': [132.12, 5.41, -3.5, 0.0, 114.1, 0.5, 0.0, 0.0, 0.49],
        'D': [133.10, 2.77, -3.5, -1.0, 111.1, 0.5, 0.0, 0.0, 0.48],
        'C': [121.15, 5.07, 2.5, 0.0, 108.5, 0.0, 0.0, 0.0, 0.62],
        'E': [147.13, 3.22, -3.5, -1.0, 138.4, 0.5, 0.0, 0.0, 0.54],
        'Q': [146.15, 5.65, -3.5, 0.0, 143.8, 0.5, 0.0, 0.0, 0.56],
        'G': [75.07, 5.97, -0.4, 0.0, 60.1, 0.0, 0.0, 1.0, 0.48],
        'H': [155.16, 7.59, -3.2, 0.0, 153.2, 0.5, 0.0, 0.0, 0.61],
        'I': [131.17, 6.02, 4.5, 0.0, 166.7, 0.0, 0.0, 0.0, 0.73],
        'L': [131.17, 5.98, 3.8, 0.0, 166.7, 0.0, 0.0, 0.0, 0.69],
        'K': [146.19, 9.74, -3.9, 1.0, 168.6, 1.0, 0.0, 0.0, 0.52],
        'M': [149.21, 5.74, 1.9, 0.0, 162.9, 0.0, 0.0, 0.0, 0.70],
        'F': [165.19, 5.48, 2.8, 0.0, 189.9, 0.0, 1.0, 0.0, 0.80],
        'P': [115.13, 6.30, -1.6, 0.0, 112.7, 0.0, 0.0, 0.0, 0.36],
        'S': [105.09, 5.68, -0.8, 0.0, 89.0, 0.5, 0.0, 0.0, 0.41],
        'T': [119.12, 5.60, -0.7, 0.0, 116.1, 0.5, 0.0, 0.0, 0.48],
        'W': [204.23, 5.89, -0.9, 0.0, 227.8, 0.0, 1.0, 0.0, 0.85],
        'Y': [181.19, 5.66, -1.3, 0.0, 193.6, 0.5, 1.0, 0.0, 0.76],
        'V': [117.15, 5.96, 4.2, 0.0, 140.0, 0.0, 0.0, 0.0, 0.64],
    }
    # Features: [MW, pKa, hydrophobicity, charge, volume, polarity, aromaticity, flexibility, alpha-helix propensity]
    return aa_properties.get(amino_acid.upper(), [0.0]*9)  # Default to [0.0]*9 for unknown amino acids

def main():
    """
    Main function to run the Omics RAG Pipeline.
    """
    # Initialization
    vector_store_path = "/mnt/c/users/wes/vector_data_good/omics_vector_store"
    rag_pipeline = OmicsRAGPipeline(vector_store_path)
    rag_pipeline.load_vector_store()
    rag_pipeline.create_rag_pipeline()

    # Initialize GNN
    input_dim = 9  # Based on the number of features per amino acid
    hidden_dim = 128  # Increased hidden dimension for better capacity
    output_dim = 64  # Embedding dimension
    rag_pipeline.initialize_gnn(input_dim, hidden_dim, output_dim, dropout=0.3)  # Increased dropout for regularization

    # Load and preprocess protein data for GNN training
    protein_data_file = 'protein_data_combined.csv'
    train_data, val_data = load_and_preprocess_protein_data(protein_data_file)

    # Train GNN
    rag_pipeline.train_gnn(train_data, val_data, epochs=100, patience=15)

    # Load the best GNN model
    rag_pipeline.gnn_model.load_state_dict(torch.load('best_gnn_model.pth'))
    logger.info("Best GNN model loaded for inference.")

    # Generate protein sequence
    prompt = "Design a protein sequence that enhances mitochondrial function and increases lifespan. The protein sequence is: "
    protein_sequence = generate_protein_sequence(prompt)
    logger.info(f"Generated protein sequence: {protein_sequence}")

    # Analyze the generated protein sequence using GNN
    protein_embedding = rag_pipeline.analyze_protein_sequence(protein_sequence)
    logger.info(f"Protein embedding: {protein_embedding}")

    # Use the embedding in your RAG pipeline query
    context = f"Protein embedding: {protein_embedding.tolist()}"
    question = (
        f"Your objective is to take the user's initial query, take the context and data you are given, "
        f"as well as the search results supplied to you, and offer any specific details about the viability, "
        f"implementation, data findings, or potential novel benefits/attributes that the predicted structure might have "
        f"according to your dataset findings. User Input: {prompt} Sequence: {protein_sequence} Context: {context}?"
    )

    result = rag_pipeline.query(question)
    logger.info(f"Answer: {result['result']}")
    logger.info("Source documents:")
    for doc in result['source_documents']:
        logger.info(f"- {doc.metadata.get('id', 'N/A')}: {doc.page_content[:100]}...")

if __name__ == "__main__":
    main()
