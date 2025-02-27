import os

class OpenSearchClient:
    def __init__(self, global_index_file="global_index_name.txt"):
        self.index_prefix = "open_webui"
        self.global_index_file = global_index_file  # Path to the file for persisting the global index name
        
        # Check if the global index name file exists and read from it, if so
        self.global_index_name = self._load_global_index_name()

        # Initialize OpenSearch client
        self.client = OpenSearch(
            hosts=[OPENSEARCH_URI],
            use_ssl=False,
            verify_certs=False,
            http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        )

    def _load_global_index_name(self) -> str:
        """
        Load the global index name from a persistent file (if it exists).
        """
        if os.path.exists(self.global_index_file):
            with open(self.global_index_file, "r") as file:
                return file.read().strip()  # Read the stored global index name
        return None  # No global index name exists yet

    def _save_global_index_name(self, index_name: str):
        """
        Save the global index name to the persistent file.
        """
        with open(self.global_index_file, "w") as file:
            file.write(index_name)  # Store the global index name in the file

    def _get_or_create_index(self, dimension: int):
        """
        Ensure that only one index is created. 
        If it doesn't exist, create it. If it exists, skip creation.
        """
        if self.global_index_name is None:
            # If no global index name exists, create the first one
            self.global_index_name = f"{self.index_prefix}_global_index"
            print(f"Creating index '{self.global_index_name}' as it's the first upload.")
            self._create_index(self.global_index_name, dimension)
            self._save_global_index_name(self.global_index_name)  # Save the index name to the file
        else:
            print(f"Index '{self.global_index_name}' already exists. Skipping creation.")

    def _create_index(self, collection_name: str, dimension: int):
        """
        Create a new index if it doesn't already exist.
        """
        if self.has_collection(collection_name):  # Don't create if the index already exists
            print(f"Index for '{collection_name}' already exists. Skipping creation.")
            return

        body = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": dimension,  # Adjust based on your vector dimensions
                        "similarity": "faiss",
                        "method": {
                            "name": "hnsw",
                            "space_type": "ip",  # Use inner product to approximate cosine similarity
                            "engine": "faiss",
                            "ef_construction": 128,
                            "m": 16,
                        },
                    },
                    "text": {"type": "text"},
                    "metadata": {"type": "object"},
                }
            }
        }
        # Create the index
        self.client.indices.create(index=f"{self.index_prefix}_{collection_name}", body=body)

    def has_collection(self, collection_name: str) -> bool:
        """
        Check if an index (collection) exists in OpenSearch.
        """
        return self.client.indices.exists(index=f"{self.index_prefix}_{collection_name}")

    def insert(self, items: list[VectorItem]):
        """
        Insert items into the index (using the first index created).
        """
        # Ensure the global index is created (only once)
        self._get_or_create_index(dimension=len(items[0]["vector"]))

        actions = [
            {
                "index": {
                    "_id": item["id"],
                    "_source": {
                        "vector": item["vector"],
                        "text": item["text"],
                        "metadata": item["metadata"],
                    },
                }
            }
            for item in items
        ]
        # Insert data into the global index
        self.client.bulk(actions, index=self.global_index_name)

    def search(self, vectors: list[list[float]], limit: int) -> Optional[SearchResult]:
        """
        Perform a search on the global index.
        """
        if self.global_index_name is None:
            raise ValueError("Global index has not been created. Please upload a file first.")

        query = {
            "size": limit,
            "_source": ["text", "metadata"],
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.vector, 'vector') + 1.0",
                        "params": {
                            "vector": vectors[0]
                        },
                    },
                }
            },
        }

        result = self.client.search(
            index=self.global_index_name, body=query
        )

        return self._result_to_search_result(result)

    def get(self) -> Optional[GetResult]:
        """
        Retrieve all items from the global index.
        """
        if self.global_index_name is None:
            raise ValueError("Global index has not been created. Please upload a file first.")

        query = {"query": {"match_all": {}}, "_source": ["text", "metadata"]}

        result = self.client.search(
            index=self.global_index_name, body=query
        )
        return self._result_to_get_result(result)
