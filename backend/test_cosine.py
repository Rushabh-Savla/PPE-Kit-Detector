import os
import sys
import numpy as np
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    pkl_path = "../known_faces/embeddings.pkl"
    if not os.path.exists(pkl_path):
        print("No embeddings found.")
        return
        
    with open(pkl_path, "rb") as f:
        known_faces = pickle.load(f)
        
    print(f"Loaded {len(known_faces)} faces: {list(known_faces.keys())}")
    
    if "RS001" in known_faces and "PM001" in known_faces:
        rs_emb = known_faces["RS001"]["embedding"]
        pm_emb = known_faces["PM001"]["embedding"]
        
        # normalize
        rs_norm = rs_emb / (np.linalg.norm(rs_emb) + 1e-10)
        pm_norm = pm_emb / (np.linalg.norm(pm_emb) + 1e-10)
        
        cosine_sim = np.dot(rs_norm, pm_norm)
        l2_dist = np.linalg.norm(rs_emb - pm_emb)
        norm_l2_dist = np.linalg.norm(rs_norm - pm_norm)
        
        print(f"Cosine Similarity (RS vs PM): {cosine_sim:.4f}")
        print(f"L2 Distance (raw): {l2_dist:.4f}")
        print(f"L2 Distance (normalized): {norm_l2_dist:.4f}")

if __name__ == "__main__":
    main()
