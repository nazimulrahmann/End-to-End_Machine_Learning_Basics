import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import ndcg_score
from tqdm import tqdm
from collections import defaultdict

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================
# Data Preparation
# ==============================================

class RecommendationDataset(Dataset):
    def __init__(self, ratings_df, users_df, items_df):
        self.ratings = ratings_df
        self.users = users_df
        self.items = items_df

        # Create mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(users_df['user_id'].unique())}
        self.item_to_idx = {item: idx for idx, item in enumerate(items_df['item_id'].unique())}

        # Prepare user and item features
        self.user_features = self._prepare_user_features()
        self.item_features = self._prepare_item_features()

    def _prepare_user_features(self):
        """Prepare user features including metadata"""
        # One-hot encode categorical features
        users_processed = pd.get_dummies(self.users, columns=['gender'])

        # Normalize numerical features
        users_processed['age'] = (users_processed['age'] - users_processed['age'].mean()) / users_processed['age'].std()

        return torch.FloatTensor(users_processed.drop('user_id', axis=1).values)

    def _prepare_item_features(self):
        """Prepare item features including metadata"""
        # One-hot encode categorical features
        items_processed = pd.get_dummies(self.items, columns=['category'])

        # Normalize numerical features
        items_processed['price'] = (items_processed['price'] - items_processed['price'].mean()) / items_processed[
            'price'].std()

        return torch.FloatTensor(items_processed.drop('item_id', axis=1).values)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        row = self.ratings.iloc[idx]
        user_idx = self.user_to_idx[row['user_id']]
        item_idx = self.item_to_idx[row['item_id']]
        rating = row['rating']

        return {
            'user_idx': torch.LongTensor([user_idx]),
            'item_idx': torch.LongTensor([item_idx]),
            'user_features': self.user_features[user_idx],
            'item_features': self.item_features[item_idx],
            'rating': torch.FloatTensor([rating])
        }


# Sample data (replace with your actual data)
ratings_data = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
    'item_id': [101, 102, 103, 101, 102, 104, 101, 103, 105, 102, 104, 105, 103, 104, 105],
    'rating': [5, 3, 4, 4, 5, 2, 3, 5, 4, 2, 5, 3, 4, 3, 5]
}

items_data = {
    'item_id': [101, 102, 103, 104, 105],
    'category': ['A', 'B', 'A', 'C', 'B'],
    'price': [10, 20, 15, 25, 30]
}

users_data = {
    'user_id': [1, 2, 3, 4, 5],
    'age': [25, 30, 35, 40, 45],
    'gender': ['M', 'F', 'M', 'F', 'M']
}

ratings_df = pd.DataFrame(ratings_data)
items_df = pd.DataFrame(items_data)
users_df = pd.DataFrame(users_data)

# Create dataset and dataloader
dataset = RecommendationDataset(ratings_df, users_df, items_df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# ==============================================
# Neural Network Models
# ==============================================

class NeuralCollaborativeFiltering(nn.Module):
    """Neural Collaborative Filtering with GMF and MLP"""

    def __init__(self, num_users, num_items, user_features_dim, item_features_dim, embedding_dim=64):
        super().__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Feature layers
        self.user_feature_layer = nn.Linear(user_features_dim, embedding_dim)
        self.item_feature_layer = nn.Linear(item_features_dim, embedding_dim)

        # GMF (Generalized Matrix Factorization) path
        self.gmf_layer = nn.Linear(embedding_dim, embedding_dim)

        # MLP (Multi-Layer Perceptron) path
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU()
        )

        # Final prediction layer
        self.output_layer = nn.Linear(embedding_dim + embedding_dim // 2, 1)

    def forward(self, user_idx, item_idx, user_features, item_features):
        # Embeddings
        user_emb = self.user_embedding(user_idx).squeeze(1)
        item_emb = self.item_embedding(item_idx).squeeze(1)

        # Feature processing
        user_feat = self.user_feature_layer(user_features)
        item_feat = self.item_feature_layer(item_features)

        # Combined embeddings
        user_combined = user_emb + user_feat
        item_combined = item_emb + item_feat

        # GMF path
        gmf = user_combined * item_combined
        gmf = self.gmf_layer(gmf)

        # MLP path
        mlp_input = torch.cat([user_combined, item_combined], dim=1)
        mlp_output = self.mlp(mlp_input)

        # Concatenate and predict
        concat = torch.cat([gmf, mlp_output], dim=1)
        prediction = self.output_layer(concat)

        return prediction.squeeze()


class TransformerRecSys(nn.Module):
    """Transformer-based Recommendation System"""

    def __init__(self, num_users, num_items, user_features_dim, item_features_dim, embedding_dim=64, num_heads=4):
        super().__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Feature projection layers
        self.user_feature_proj = nn.Linear(user_features_dim, embedding_dim)
        self.item_feature_proj = nn.Linear(item_features_dim, embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim)

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, user_idx, item_idx, user_features, item_features):
        # Embeddings
        user_emb = self.user_embedding(user_idx).squeeze(1)
        item_emb = self.item_embedding(item_idx).squeeze(1)

        # Feature projections
        user_feat = self.user_feature_proj(user_features)
        item_feat = self.item_feature_proj(item_features)

        # Combined representations
        user_combined = user_emb + user_feat
        item_combined = item_emb + item_feat

        # Prepare sequence for transformer (user and item as sequence)
        sequence = torch.stack([user_combined, item_combined], dim=1)  # [batch, 2, dim]

        # Add positional encoding
        sequence = self.positional_encoding(sequence)

        # Transformer processing
        transformer_out = self.transformer_encoder(sequence)

        # Pooling
        user_trans = transformer_out[:, 0, :]
        item_trans = transformer_out[:, 1, :]

        # Final prediction
        concat = torch.cat([user_trans, item_trans], dim=1)
        prediction = self.prediction_head(concat)

        return prediction.squeeze()


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model, max_len=2):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


class GraphRecSys(nn.Module):
    """Graph Neural Network for Recommendations"""

    def __init__(self, num_users, num_items, user_features_dim, item_features_dim, embedding_dim=64):
        super().__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Feature projection layers
        self.user_feature_proj = nn.Linear(user_features_dim, embedding_dim)
        self.item_feature_proj = nn.Linear(item_features_dim, embedding_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(embedding_dim, embedding_dim),
            GraphConvLayer(embedding_dim, embedding_dim)
        ])

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, user_idx, item_idx, user_features, item_features):
        # Embeddings
        user_emb = self.user_embedding(user_idx).squeeze(1)
        item_emb = self.item_embedding(item_idx).squeeze(1)

        # Feature projections
        user_feat = self.user_feature_proj(user_features)
        item_feat = self.item_feature_proj(item_features)

        # Combined representations
        user_combined = user_emb + user_feat
        item_combined = item_emb + item_feat

        # Graph processing (simplified)
        for layer in self.gnn_layers:
            user_combined, item_combined = layer(user_combined, item_combined)

        # Final prediction
        concat = torch.cat([user_combined, item_combined], dim=1)
        prediction = self.prediction_head(concat)

        return prediction.squeeze()


class GraphConvLayer(nn.Module):
    """Simplified Graph Convolution Layer"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.user_to_item = nn.Linear(in_dim, out_dim)
        self.item_to_user = nn.Linear(in_dim, out_dim)

    def forward(self, user_repr, item_repr):
        # User to item message passing
        item_update = self.user_to_item(user_repr)

        # Item to user message passing
        user_update = self.item_to_user(item_repr)

        return user_update, item_update


# ==============================================
# Training and Evaluation
# ==============================================

def train_model(model, dataloader, epochs=10, lr=0.001):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            # Move data to device
            user_idx = batch['user_idx'].to(device)
            item_idx = batch['item_idx'].to(device)
            user_features = batch['user_features'].to(device)
            item_features = batch['item_features'].to(device)
            rating = batch['rating'].to(device)

            # Forward pass
            preds = model(user_idx, item_idx, user_features, item_features)
            loss = criterion(preds, rating)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")


def evaluate_model(model, dataset):
    model.eval()
    all_preds = []
    all_ratings = []

    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=32):
            user_idx = batch['user_idx'].to(device)
            item_idx = batch['item_idx'].to(device)
            user_features = batch['user_features'].to(device)
            item_features = batch['item_features'].to(device)
            rating = batch['rating'].to(device)

            preds = model(user_idx, item_idx, user_features, item_features)

            all_preds.extend(preds.cpu().numpy())
            all_ratings.extend(rating.cpu().numpy())

    # Calculate metrics
    mse = np.mean((np.array(all_preds) - np.array(all_ratings)) ** 2)
    rmse = np.sqrt(mse)
    ndcg = ndcg_score([np.array(all_ratings)], [np.array(all_preds)])

    print(f"\nEvaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"NDCG: {ndcg:.4f}")


def generate_recommendations(model, dataset, user_id, n_recommendations=5):
    model.eval()

    # Get user index
    user_idx = torch.LongTensor([dataset.user_to_idx[user_id]]).to(device)
    user_features = dataset.user_features[dataset.user_to_idx[user_id]].unsqueeze(0).to(device)

    # Get all items
    item_indices = torch.LongTensor(list(range(len(dataset.item_to_idx)))).to(device)
    item_features = dataset.item_features.to(device)

    # Get user's rated items
    rated_items = ratings_df[ratings_df['user_id'] == user_id]['item_id'].tolist()
    rated_item_indices = [dataset.item_to_idx[item] for item in rated_items]

    # Predict for all items
    with torch.no_grad():
        # Repeat user features for all items
        user_indices = user_idx.repeat(len(item_indices))
        user_features_repeated = user_features.repeat(len(item_indices), 1)

        predictions = model(user_indices, item_indices, user_features_repeated, item_features)

    # Create recommendations dataframe
    recommendations = pd.DataFrame({
        'item_id': [item_id for item_id, idx in dataset.item_to_idx.items()],
        'predicted_rating': predictions.cpu().numpy()
    })

    # Filter out already rated items
    recommendations = recommendations[~recommendations['item_id'].isin(rated_items)]

    # Sort and get top recommendations
    top_recommendations = recommendations.sort_values('predicted_rating', ascending=False).head(n_recommendations)

    return top_recommendations.merge(items_df, on='item_id')


# ==============================================
# Example Usage
# ==============================================

if __name__ == "__main__":
    # Initialize models
    num_users = len(dataset.user_to_idx)
    num_items = len(dataset.item_to_idx)
    user_features_dim = dataset.user_features.shape[1]
    item_features_dim = dataset.item_features.shape[1]

    ncf_model = NeuralCollaborativeFiltering(num_users, num_items, user_features_dim, item_features_dim)
    transformer_model = TransformerRecSys(num_users, num_items, user_features_dim, item_features_dim)
    graph_model = GraphRecSys(num_users, num_items, user_features_dim, item_features_dim)

    # Train models
    print("Training Neural Collaborative Filtering Model...")
    train_model(ncf_model, dataloader)

    print("\nTraining Transformer Model...")
    train_model(transformer_model, dataloader)

    print("\nTraining Graph Model...")
    train_model(graph_model, dataloader)

    # Evaluate models
    print("\nEvaluating Neural Collaborative Filtering Model:")
    evaluate_model(ncf_model, dataset)

    print("\nEvaluating Transformer Model:")
    evaluate_model(transformer_model, dataset)

    print("\nEvaluating Graph Model:")
    evaluate_model(graph_model, dataset)

    # Generate recommendations
    user_id = 1
    print(f"\nRecommendations for user {user_id} from NCF Model:")
    print(generate_recommendations(ncf_model, dataset, user_id))

    print(f"\nRecommendations for user {user_id} from Transformer Model:")
    print(generate_recommendations(transformer_model, dataset, user_id))

    print(f"\nRecommendations for user {user_id} from Graph Model:")
    print(generate_recommendations(graph_model, dataset, user_id))