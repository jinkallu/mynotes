import numpy as np
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import uuid
import os

class VectorDatabase:
    def __init__(self):
        # 1. Prepare HuggingFace embedder for text embeddings
        self.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def test(self):
        print(self.embedder.embed_query("mother"))

    def create(self):
        

        # 2. Generate synthetic sensory vectors
        dim = 384  # Must match embedding dim of the text model
        np.random.seed(42)

        modalities = ["vision", "audio", "touch", "smell"]
        vectors = [np.random.rand(dim).astype('float32') for _ in modalities]
        weights = [0.4, 0.3, 0.2, 0.1]

        # 3. Create child documents with metadata
        child_docs = []
        child_ids = []

        for i, (v, modality) in enumerate(zip(vectors, modalities)):
            doc_id = str(uuid.uuid4())
            child_ids.append(doc_id)
            child_docs.append(
                Document(
                    page_content="",
                    metadata={
                        "id": doc_id,
                        "modality": modality,
                        "timestamp": "t1",
                        "parent_id": "mother",
                        "weight_to_parent": weights[i]
                    }
                )
            )

        # 4. Create identity vector for "mother" using real text embedding
        v_mother = np.array(self.embedder.embed_query("mother")).astype("float32")

        # Create parent document
        parent_doc = Document(
            page_content="mother",  # Text for retrievability
            metadata={
                "id": "mother",
                "modality": "identity",
                "timestamp": "t1",
                "children_ids": child_ids,
                "children_weights": weights
            }
        )

        # 5. Combine documents and vectors
        all_docs = child_docs + [parent_doc]
        all_vectors = vectors + [v_mother]

        # 6. Create FAISS index
        # Format: list of (text, vector) pairs
        text_embeddings = [(doc.page_content, vec) for doc, vec in zip(all_docs, all_vectors)]

        vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=self.embedder
        )


        # 7. Save to disk (optional)
        vectorstore.save_local("faiss_mother_with_text")

        # 8. Search using the word "mother"
        results = vectorstore.similarity_search_with_score("mother", k=5)

        # 9. Print results
        print("\n🔍 Search results for 'mother':\n")
        for doc, score in results:
            print(f"- doc {doc}, Score: {score:.4f}")


if __name__ == '__main__':
    vectorDatabase = VectorDatabase()
    vectorDatabase.create()
