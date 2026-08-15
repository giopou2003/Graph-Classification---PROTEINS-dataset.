# Graph Classification - PROTEINS Dataset (Milestone 1)

This repository covers **Milestone 1** for graph classification on the **PROTEINS** benchmark dataset using three distinct methodological approaches.

---

## 📌 Methods Overview

### 🔹 Method A: Feature Engineering + MLP

Extracts hand-crafted topological and node-level features from graphs before feeding them into a Multi-Layer Perceptron (MLP) classifier.

1. **Feature Engineering**:
   - **Graph-level Features**: Global structural statistics (node count $|V|$, edge count $|E|$, average node degree, graph density, Laplacian spectrum, clustering coefficient).
   - **Node-level Aggregations**: Summary statistics (Sum, Mean, Max, Std) aggregated across node attributes/labels per graph.
2. **Feature Selection**:
   - Dimensionality reduction and noise removal techniques (e.g., Variance Thresholding, Correlation Analysis, SelectKBest, Mutual Information).
3. **MLP Training**:
   - Training a Feedforward Neural Network (MLP) on the selected feature vectors.
   - Regularization via Dropout, activation functions (ReLU/GELU), and Early Stopping.

---

### 🔹 Method B: Shallow Embeddings + MLP

Generates low-dimensional node representations using random walk-based algorithms and pools them into graph-level embeddings.

#### 1. Standard Node2Vec (Without Contrastive Loss)
- **Node2Vec Embeddings**: Generates node embeddings followed by Graph-level Pooling (Mean, Max, Sum) to create single graph representations.
- **Node2Vec Hyperparameter Tuning**:
  - Parameters $p$ (return) & $q$ (in-out) to balance BFS vs. DFS exploration.
  - Embedding dimension $d$, Walk length, Number of walks, and Window size.
- **MLP Training**: Training an MLP classifier on the pooled graph embeddings.

#### 2. Node2Vec + Contrastive Loss
- **Node2Vec With Contrastive Loss**:
  - Employs Contrastive Loss (e.g., InfoNCE / SimCLR-style loss) with graph augmentations (node dropping, edge perturbation).
  - Pulls graphs with similar structures/classes closer in the embedding space.
- **Hyperparameter Tuning**: Tuning temperature parameter ($\tau$), augmentation rates, and $p, q$ values.
- **MLP Training With Contrastive Loss**: Training the MLP classifier on the contrastively optimized embeddings.

---

### 🔹 Method C: GNN Graph Classification

End-to-end training using Graph Neural Networks (GCN, GIN, GraphSAGE) to jointly learn node representations and perform graph classification.

1. **GNN Architecture**:
   - **Message Passing Layers**: Neighborhood aggregation layers (e.g., `GCNConv`, `GINConv`).
   - **Global Readout Layer**: Collapses node embeddings into a graph-level vector (e.g., `global_mean_pool`, `global_max_pool`).
   - **Classification Head**: Fully Connected layers producing classification logits.
2. **Grid Search & Hyperparameter Tuning**:
   - **Network Setup**: Number of layers, Hidden Dimensions (32, 64, 128).
   - **Optimization**: Learning rate, Weight Decay, Dropout rate.
   - **Pooling Selection**: Mean vs. Sum vs. Attention Pooling.
3. **GNN Training & Evaluation**:
   - End-to-end training using Cross-Entropy Loss.
   - Evaluation via Stratified $K$-Fold Cross-Validation.
   - Target metrics: **Accuracy**, **ROC-AUC**, and **F1-Score**.
