import torch
import torch.nn as nn


class RecommendationSys(nn.Module):
    # vocab: to encode and decode input features
    def __init__(self, vocab, input_dim, hidden_dim, output_dim, size_user, size_movie, embed_size):
        super(RecommendationSys, self).__init__()
        
        self.vocab = vocab
        
        self.encode_user = nn.Embedding(size_user, embed_size)
        self.encode_movie = nn.Embedding(size_movie, embed_size)
        
        self.model = nn.Sequential(
            nn.Linear(input_dim + 2 * embed_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        
    def forward(self, x):
        user_embed = self.encode_user(x[:, 0].long())
        movie_embed = self.encode_movie(x[:, 2].long())
        x = torch.cat((x, user_embed, movie_embed), 1)
        return self.model(x)