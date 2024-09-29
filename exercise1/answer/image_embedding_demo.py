import math
from sentence_transformers import SentenceTransformer, util
import os.path
import pickle
from scipy import spatial
from PIL import Image
import faiss
import numpy as np
from pathlib import Path
import base64
import openai


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

image_pkl_path = 'clip_embedding.pkl'

# Add description, and tutorial type of comments
class image_embedding_store:
    def __init__(self, dataset_dir) -> None:
        self.dataset_dir = dataset_dir
        self.model = SentenceTransformer('clip-ViT-B-32')
        self.embedding_dict = self.update_imgage_embedding()

        self.id_to_name = {}
        self.embedding_list = []
        next_id = 0
        for file_name, embedding in self.embedding_dict.items():
            self.id_to_name[next_id] = file_name
            self.embedding_list.append(embedding)
            next_id = next_id + 1

        self.animal_labels = ["cat", "dog", "bird", "fish", "horse"]
        self.animal_label_embedding = self.model.encode(self.animal_labels)

    def update_imgage_embedding(self):
        embedding_dict = {}

        if os.path.isfile(f"{self.dataset_dir}/{image_pkl_path}"):
            with open(f"{self.dataset_dir}/{image_pkl_path}", 'rb') as file:
                embedding_dict = pickle.load(file)

        embedding_dict_updated = False
        file_in_dataset = []
        for path in Path(self.dataset_dir).rglob('*.[jp][pn]g'):
            filename = os.fsdecode(path)
            file_in_dataset.append(filename)
            if filename in embedding_dict:
                continue
            embedding_dict_updated = True
            print(f"Generating embedding for {filename}")

            embedding_dict[filename] = self.model.encode(Image.open(f"{filename}"))
        all_files = list(embedding_dict.keys())
        files_not_in_dataset = [x for x in all_files if x not in file_in_dataset]
        for file in files_not_in_dataset:
            del embedding_dict[file]
            embedding_dict_updated = True

        print(f"Number of images in dataset: {len(embedding_dict)}")

        if embedding_dict_updated:
            with open(f"{self.dataset_dir}/{image_pkl_path}", "wb") as file:
                pickle.dump(embedding_dict, file)

        return embedding_dict

    def get_all_files(self):
        return list(self.embedding_dict.keys())

    def find_top_k_similar_images_by_text(self, description, k=3):
        text_embedding = self.model.encode(description)
        distances = []
        for name, vec in self.embedding_dict.items():
            similarity = util.cos_sim(text_embedding, vec)
            distances.append((name, similarity))

        # Sort by similarity in descending order
        distances.sort(key=lambda x: x[1], reverse=True)

        # Get the top k similar images
        top_k_images = [name for name, _ in distances[:k]]
        return top_k_images
    
    def find_top_k_similar_images_by_image(self, image_path, k=3):
        image_embedding = self.model.encode(Image.open(f"{image_path}"))
        distances = []
        for name, vec in self.embedding_dict.items():
            similarity = util.cos_sim(image_embedding, vec)
            distances.append((name, similarity))

        # Sort by similarity in descending order
        distances.sort(key=lambda x: x[1], reverse=True)

        # Get the top k similar images
        top_k_images = [name for name, _ in distances[:k]]
        return top_k_images
    
    def find_top_k_by_faiss(self, description, k=3):
        text_embedding = np.array([self.model.encode(description)]).astype('float32')
        embedding_array = np.array(self.embedding_list).astype('float32')

        # EXPERIMENT: Try a different type of index. What's its pros and cons.
        index = faiss.IndexFlatL2(embedding_array.shape[1]) 
        index.add(embedding_array)

        distances, indices = index.search(text_embedding, k)

        top_k_images = [self.id_to_name[i] for i in indices[0]]
        return top_k_images
    
    def categorize_animal_image(self, image_path):
        image_embedding = self.model.encode(Image.open(f"{image_path}"))
        distances = [[i, spatial.distance.cosine(image_embedding, vec)] for i, vec in enumerate(self.animal_label_embedding)]
        sorted_distances = sorted(distances, key=lambda x: x[1])
        return self.animal_labels[sorted_distances[0][0]]

    def plot_tsne(self):
        to_label_ids = []
        label_ids = {}
        for id, name in self.id_to_name.items():
            label = name.split("\\")[1]
            if label not in label_ids:
                label_ids[label] = len(label_ids)
            to_label_ids.append(label_ids[label])
            
        embedding_array = np.array(self.embedding_list).astype('float32')

        print(f"{label_ids}")

        # Create a t-SNE model and transform the data
        tsne = TSNE(n_components=2, perplexity=8, random_state=42, init='random')
        vis_dims = tsne.fit_transform(embedding_array)

        x = [x for x,y in vis_dims]
        y = [y for x,y in vis_dims]

        # colormap = matplotlib.colors.ListedColormap(colors)
        colormap = matplotlib.colors.ListedColormap(np.random.rand(len(label_ids.keys()), 3))
        plt.scatter(x, y, c=to_label_ids, cmap=colormap, alpha=0.3)
        # plt.scatter(x, y, alpha=0.3)
        plt.title("Images visualized using t-SNE")
        # Add legend based on keys of label_ids
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colormap(i), markersize=10) for i in range(len(label_ids))]
        plt.legend(handles, label_ids.keys(), title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

# Test stub for the different search algorithms
if __name__ == "__main__":
    dataset_dir = 'animals'
    # Example usage of the image embedding functionalities
    img_store = image_embedding_store(dataset_dir)  # Assuming ImageEmbedding is the class name

    query = "a cat reading a book"

    img_store.plot_tsne()

    # closest_image = img_store.find_top_k_similar_images_by_text(query, k=1)
    # print(f"Closest image: {closest_image}")

    # closest_image = img_store.find_top_k_similar_images_by_image("cat_studying_b.png", k=1)
    # print(f"Closest image: {closest_image}")

    # animal = img_store.categorize_animal_image("dog_a.png")
    # print(f"Animal: {animal}")
