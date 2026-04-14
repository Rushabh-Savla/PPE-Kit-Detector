import os
import sys
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from facenet_pytorch import MTCNN, InceptionResnetV1

def get_emb(mtcnn, facenet, path, out_path):
    img = Image.open(path).convert('RGB')
    face = mtcnn(img)
    if face is None: return None
    if len(face.shape) == 4:
        face = face[0]
        
    # save cropped face
    import torchvision.transforms as T
    to_pil = T.ToPILImage()
    # mtcnn outputs [-1, 1], so we denormalize to [0, 1]
    denorm = (face + 1) / 2
    to_pil(denorm).save(out_path)
    
    face = face.unsqueeze(0)
    with torch.no_grad():
        emb = facenet(face)
    emb = emb.numpy()[0]
    return emb / np.linalg.norm(emb)

def main():
    mtcnn = MTCNN(keep_all=True)
    facenet = InceptionResnetV1(pretrained='vggface2').eval()

    path1 = r"C:\Users\rvsav\Downloads\Photo.jpg"
    path2 = r"C:\Users\rvsav\Downloads\Image.jpeg"

    emb1 = get_emb(mtcnn, facenet, path1, "test_face1.png")
    emb2 = get_emb(mtcnn, facenet, path2, "test_face2.png")

    if emb1 is None or emb2 is None:
        print("Failed to get one of the embeddings")
        return

    sim = np.dot(emb1, emb2)
    print(f"Direct Cosine Sim (Photo.jpg vs Image.jpeg): {sim:.4f}")

if __name__ == "__main__":
    main()
