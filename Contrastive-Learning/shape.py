import numpy as np
emb = np.load("data/embeddings/val_semantic_embeddings.npy")
print(emb.shape)  # will print either (N, 1024) or (N, 128)