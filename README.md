# ECE 219 Project 2: Clustering & Unsupervised Learning and Introduction to Multi-modal Models

This project explores unsupervised learning techniques applied to Steam game reviews, focusing on clustering and text analysis. The project is divided into three main tasks involving review length discovery, game genre clustering, and theme discovery using LLMs.

## Project Overview

### Dataset
- **Main dataset**: Reviews from the top 200 games ranked by total reviews on Steam
- For each game: 100 English Recommended reviews + 100 English Not Recommended reviews
- Selected by highest helpfulness (upvotes) to ensure informative content
- **Held-out dataset**: Secret game's reviews (Black Myth: Wukong) for Task 3

### Review Features
Each review includes:
- `user`: User name
- `playtime`: Hours played
- `post_date`: Review posting date
- `helpfulness`: Number of upvotes
- `review_text`: The review content
- `recommend`: True/False recommendation
- `early_access_review`: Whether posted during early access
- `appid`: Unique game identifier
- `game_name`: Game title
- `release_date`: Game release date
- `genres`: Game genres (multi-label, comma-separated)

## Task 1: Unsupervised Review Length Discovery

### Objective
Discover review length structure from textual representations using clustering techniques.

### Methodology

#### 1.1 Defining Pseudo-labels (for evaluation only)
- Reviews in **top 25% (≥ q75)** by length labeled as **Long**
- Reviews in **bottom 25% (≤ q25)** by length labeled as **Short**
- Middle 50% discarded for clarity
- **Results**: 
  - q25: 11.0 tokens, q75: 179.0 tokens
  - 20,497 reviews retained
  - Average Short review length: 6.4 tokens
  - Average Long review length: 493.0 tokens

#### 1.2 Representations
1. **TF-IDF**: 40,000 × 27,736 matrix (min_df=3, English stopwords, unigrams only)
2. **MiniLM Embeddings**: 40,000 × 384 dense vectors using `sentence-transformers/all-MiniLM-L6-v2`

#### 1.3 Clustering Pipelines
- **Dimensionality Reduction**: None, SVD(50), UMAP(50)
- **Clustering Methods**: K-Means (k=2), Agglomerative (n_clusters=2)
- **Evaluation Metrics**: Homogeneity, Completeness, V-measure, ARI, AMI

#### Key Findings
- **Best Pipeline**: MiniLM + SVD(50) + K-Means
  - Homogeneity: 0.274
  - Completeness: 0.243
  - V-measure: 0.258
  - ARI: 0.369
  - AMI: 0.258
- **TF-IDF vs MiniLM**: MiniLM separates Short vs Long reviews more cleanly due to its dense 384-dimensional space optimized for sentence-level similarity, compared to TF-IDF's sparse, noisy high-dimensional space

## Task 2: Unsupervised Game Genre Discovery

### Objective
Discover game genre structure by clustering games based on aggregated review text.

### Methodology

#### 2.1 Aggregating Reviews
- Concatenate all review texts for each game
- 200 games total from the dataset

#### 2.2 Representations
1. **TF-IDF**: Game-level aggregated text (min_df=3, English stopwords)
2. **MiniLM Embeddings**: Sentence embeddings of aggregated text

#### 2.3 Clustering Pipelines
- **Dimensionality Reduction**:
  - For TF-IDF: SVD(50), Autoencoder
  - For MiniLM: None, SVD(50), UMAP(50)
- **Clustering Methods**: K-Means (k=5), Agglomerative (k=5), HDBSCAN

#### 2.4 Multi-genre Interpretation
- Genres treated as multi-label metadata
- Cluster genre purity: Fraction of games containing cluster's most common genre
- Analysis of top 3 genres per cluster with percentages

#### Key Findings
Various clustering configurations explored:
- **TF-IDF + SVD(50) + K-Means**: 5 clusters with Action-Adventure-Indie dominance
- **MiniLM + SVD + HDBSCAN**: 3 main clusters (146, 42, 12 games) + noise cluster
- **MiniLM + UMAP + Agglomerative**: 5 clusters with Action, Indie, Simulation, Adventure, RPG mix

## Task 3: Theme Clustering Discovery with LLMs and Multi-modal Learning

### Objective
Analyze reviews from a held-out game (Black Myth: Wukong) to discover complaint and praise themes, plus explore multi-modal classification.

### 3.1 Review Theme Discovery

#### Methodology
- Separate positive and negative reviews for Black Myth: Wukong
- Cluster reviews using TF-IDF (unigrams + bigrams, min_df=2)
- Use Agglomerative clustering (k=5)
- Identify top TF-IDF terms and exemplar reviews per cluster

#### Negative Review Clusters (Complaints)
1. **"Boring design & not worth it"**: Visual quality vs actual gameplay disappointment
2. **"Technical issues, hardware & politics"**: Crashes, performance problems, developer controversies
3. **"Story and bosses don't land"**: Narrative and boss fight dissatisfaction

#### Positive Review Clusters (Praise)
1. **"Overall GOTY / general praise"**: High-level appreciation, 10/10 ratings
2. **"Return-to-monke meme hype"**: Meme-based appreciation ("REJECT MODERNITY, RETURN TO MONKE")

### 3.2 LLM Labeling
- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **Approach**: Feed LLM with top terms + exemplar reviews to generate interpretable cluster labels
- Prompt template includes cluster evidence (terms and examples)

### 3.3 Multi-modal Classification (Flower Classification)

#### Dataset
- 5 flower classes: Daisy, Dandelion, Rose, Sunflower, Tulip
- VGG features extracted from images

#### Methodology
- **MLP Classifier**: 3-layer neural network
  - Input-> 1280 -> 640 -> 5 classes
- **Baseline**: Original VGG features
- **Dimensionality Reduction**: Various methods tested (PCA, UMAP, etc.)
- **Training**: 100 epochs, batch size 128, Adam optimizer

#### Key Findings
- Test accuracy on original VGG features reported
- Comparison with reduced-dimension features
- Analysis of performance trade-offs with dimensionality reduction
- Connection to clustering results from Question 18

## Technical Stack

### Libraries Used
- **Data Processing**: pandas, numpy
- **Text Analysis**: scikit-learn (TfidfVectorizer, clustering)
- **Embeddings**: sentence-transformers (MiniLM)
- **Dimensionality Reduction**: scikit-learn (PCA, TruncatedSVD), umap-learn
- **Clustering**: scikit-learn (KMeans, AgglomerativeClustering), hdbscan
- **Deep Learning**: torch, torchvision (for MLP and image features)
- **LLM**: Transformers (Qwen model)
- **Visualization**: matplotlib

### Evaluation Metrics
- **Homogeneity**: Clusters contain only members of a single class
- **Completeness**: Members of a given class are in the same cluster
- **V-measure**: Harmonic mean of homogeneity and completeness
- **ARI (Adjusted Rand Index)**: Similarity measure accounting for chance
- **AMI (Adjusted Mutual Information)**: Information-theoretic similarity measure

## Running the Notebook

The notebook (`305965764_ClydeVillacrusis_Project2.ipynb`) contains:
- 158 cells total
- Code cells with Python implementations
- Markdown cells with analysis and interpretations
- Visualizations using matplotlib

### Requirements
- Python 3.x
- CUDA-compatible GPU (for deep learning tasks)
- Google Colab environment (original execution environment)

### Key Sections
1. Data loading and preprocessing
2. Task 1: Review length clustering experiments
3. Task 2: Game genre clustering experiments
4. Task 3: Theme discovery and multi-modal classification

## Results Summary

### Task 1 Best Performance
- **Pipeline**: MiniLM + SVD(50) + K-Means
- **ARI**: 0.369 (best separation of Short vs Long reviews)

### Task 2 Insights
- Multiple clustering configurations explored
- Genre distributions analyzed per cluster
- High-purity clusters identified for specific genre combinations

### Task 3 Discoveries
- Interpretable complaint/praise themes identified
- LLM-generated labels provide semantic understanding
- Multi-modal classification demonstrates VGG features effectiveness

## Authors
Clyde Villacrusis (ID: 305965764)

## License
See [LICENSE](LICENSE) file for details.

## Acknowledgments
- ECE 219: Large-Scale Data Mining
- Steam review dataset
- Sentence-transformers library
- Qwen LLM