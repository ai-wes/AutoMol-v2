import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class Visualization:
    def __init__(self):
        logging.info("Initialized Visualization.")

    def visualize_results(self, results: dict):
        try:
            logging.info("Visualizing results...")
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))

            # Telomere length distribution
            if 'telomere_length' in results:
                sns.histplot(results['telomere_length'], ax=axes[0, 0], kde=True)
                axes[0, 0].set_title('Telomere Length Distribution')
                axes[0, 0].set_xlabel('Telomere Length')
                axes[0, 0].set_ylabel('Frequency')

            # Gene expression heatmap
            if 'gene_expression' in results:
                sns.heatmap(results['gene_expression'], ax=axes[0, 1], cmap='viridis')
            axes[0, 1].set_title('Telomere-related Gene Expression')

            # PCA plot
            if 'pca_result' in results:
                pca_df = pd.DataFrame(results['pca_result'], columns=['PC1', 'PC2'])
                sns.scatterplot(data=pca_df, x='PC1', y='PC2', ax=axes[1, 0])
                axes[1, 0].set_title('PCA of Sequencing Data')

            # Screening results
            if 'screening_hits' in results:
                screening_hits = results['screening_hits']
                sns.scatterplot(x=range(len(screening_hits)), y=screening_hits, ax=axes[1, 1])
                axes[1, 1].set_title('High-throughput Screening Hits')
                axes[1, 1].set_xlabel('Sample Index')
                axes[1, 1].set_ylabel('Activity')

            plt.tight_layout()
            plt.savefig('visualization_results.png')
            plt.show()
            logging.info("Visualization completed and saved as 'visualization_results.png'.")
        except Exception as e:
            logging.error(f"Error in visualize_results: {e}")
            raise