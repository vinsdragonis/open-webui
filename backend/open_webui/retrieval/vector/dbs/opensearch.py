from opensearchpy import OpenSearch
from typing import Optional

from open_webui.retrieval.vector.main import VectorItem, SearchResult, GetResult
from open_webui.config import (
    OPENSEARCH_URI,
    OPENSEARCH_SSL,
    OPENSEARCH_CERT_VERIFY,
    OPENSEARCH_USERNAME,
    OPENSEARCH_PASSWORD,
)


class OpenSearchClient:
    def __init__(self):
        self.index_prefix = "open_webui"
        self.hash_to_index_map = {}
        self.client = OpenSearch(
            hosts=[OPENSEARCH_URI],
            use_ssl=False,
            verify_certs=False,
            http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        )

    def _result_to_get_result(self, result) -> GetResult:
        ids = []
        documents = []
        metadatas = []

        for hit in result["hits"]["hits"]:
            ids.append(hit["_id"])
            documents.append(hit["_source"].get("text"))
            metadatas.append(hit["_source"].get("metadata"))

        return GetResult(ids=ids, documents=documents, metadatas=metadatas)

    def _result_to_search_result(self, result) -> SearchResult:
        ids = []
        distances = []
        documents = []
        metadatas = []

        for hit in result["hits"]["hits"]:
            ids.append(hit["_id"])
            distances.append(hit["_score"])
            documents.append(hit["_source"].get("text"))
            metadatas.append(hit["_source"].get("metadata"))

        return SearchResult(
            ids=ids, distances=distances, documents=documents, metadatas=metadatas
        )

    def _generate_index_name(self, collection_name: str):
        """Generate or get the index name for a given collection (hash)"""
        # For this, we will assume collection_name is already a SHA256 hash
        if collection_name not in self.hash_to_index_map:
            self.hash_to_index_map[collection_name] = f"{self.index_prefix}_{collection_name}"
        return self.hash_to_index_map[collection_name]

    def _create_index(self, collection_name: str, dimension: int):
        """Ensure the index is created only once for a given collection_name (unique hash)."""
        # Generate the index name based on the collection name (hash)
        index_name = self._generate_index_name(collection_name)
        
        # Check if the index already exists, if so, skip creation
        if self.client.indices.exists(index=index_name):
            print(f"Index {index_name} already exists, skipping creation.")
            return  # Skip index creation if it already exists

        # If the index doesn't exist, create it
        body = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": dimension,  # Adjust based on your vector dimensions
                        "index": f"{self.index_prefix}_{collection_name}",
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
        self.client.indices.create(index=index_name, body=body)
        print(f"Index {index_name} created.")

    def _create_batches(self, items: list[VectorItem], batch_size=100):
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def has_collection(self, collection_name: str) -> bool:
        # has_collection here means has index.
        # We are simply adapting to the norms of the other DBs.
        return self.client.indices.exists(index=f"{self.index_prefix}_{collection_name}")

    def delete_colleciton(self, collection_name: str):
        # delete_collection here means delete index.
        # We are simply adapting to the norms of the other DBs.
        self.client.indices.delete(index=f"{self.index_prefix}_{collection_name}")

    def search(self, collection_name: str, vectors: list[list[float]], limit: int):
        """Perform a search query on the collection's index."""
        index_name = self._generate_index_name(collection_name)
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
                        },  # Assuming single query vector
                    },
                }
            },
        }
        result = self.client.search(
            index=index_name, body=query
        )
        return self._result_to_search_result(result)

    def query(self, collection_name: str, filter: dict, limit: Optional[int] = None) -> Optional[GetResult]:
        """Query in the collection by using the unique index name derived from collection_name."""
        if not self.has_collection(collection_name):
            return None
        index_name = self._generate_index_name(collection_name)
        query_body = {
            "query": {
                "bool": {
                    "filter": []
                }
            },
            "_source": ["text", "metadata"],
        }
        for field, value in filter.items():
            query_body["query"]["bool"]["filter"].append({
                "term": {field: value}
            })
        size = limit if limit else 10
        result = self.client.search(
            index=index_name,
            body=query_body,
            size=size
        )
        return self._result_to_get_result(result)


    def get_or_create_index(self, collection_name: str, dimension: int):
        if not self.has_collection(collection_name):
            self._create_index(collection_name, dimension)

    def get(self, collection_name: str) -> Optional[GetResult]:
        query = {"query": {"match_all": {}}, "_source": ["text", "metadata"]}

        result = self.client.search(
            index=f"{self.index_prefix}_{collection_name}", body=query
        )
        return self._result_to_get_result(result)

    def insert(self, collection_name: str, items: list[VectorItem]):
        """Insert documents into the existing or newly created index."""
        index_name = self._generate_index_name(collection_name)
        # Ensure the index exists before inserting documents
        if not self.client.indices.exists(index=index_name):
            self._create_index(collection_name, dimension=len(items[0]["vector"]))

        for batch in self._create_batches(items):
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
                for item in batch
            ]
            self.client.bulk(actions)

    def upsert(self, collection_name: str, items: list[VectorItem]):
        """Upsert documents into the existing or newly created index."""
        index_name = self._generate_index_name(collection_name)
        
        # Ensure the index exists before performing the upsert
        if not self.client.indices.exists(index=index_name):
            self._create_index(collection_name, dimension=len(items[0]["vector"]))
    
        for batch in self._create_batches(items):
            actions = [
                {
                    "update": {
                        "_id": item["id"],
                        "_index": index_name,
                        "doc": {
                            "vector": item["vector"],
                            "text": item["text"],
                            "metadata": item["metadata"],
                        },
                        "doc_as_upsert": True  # This ensures it performs an insert if not found
                    }
                }
                for item in batch
            ]
            self.client.bulk(body=actions)

    def delete(self, collection_name: str, ids: list[str]):
        actions = [
            {"delete": {"_index": f"{self.index_prefix}_{collection_name}", "_id": id}}
            for id in ids
        ]
        self.client.bulk(body=actions)

    def reset(self):
        indices = self.client.indices.get(index=f"{self.index_prefix}_*")
        for index in indices:
            self.client.indices.delete(index=index)
