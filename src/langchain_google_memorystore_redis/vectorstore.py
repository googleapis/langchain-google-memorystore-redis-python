# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import operator
import re
import uuid
from abc import ABC
from itertools import zip_longest
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import redis
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# Setting up a basic logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

DEFAULT_CONTENT_FIELD = "page_content"
DEFAULT_VECTOR_FIELD = "vector"
DEFAULT_DATA_TYPE = "FLOAT32"
DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE


class IndexConfig(ABC):
    """
    Base configuration class for all types of indexes.
    """

    def __init__(
        self,
        name: str,
        field_name: str,
        type: str,
    ):
        """
        Initializes the IndexConfig object.

        Args:
            name (str): A unique identifier for the index. This name is used to
                distinguish the index within the storage system, enabling targeted
                operations such as updates, queries, and deletions.
            field_name (str): Specifies the name of the field within the data that
                will be indexed. This field name directs the indexing process to
                the relevant part of the data structure.
            type (str): Indicates the type of index to be created, which determines
                the underlying algorithm or structure used for indexing and search
                operations. Examples include "HNSW" for hierarchical navigable small
                world indexes and "FLAT" for brute-force search indexes.
            data_type (str, optional): Defines the data type of the elements within
                the vector being indexed, such as "FLOAT32" for 32-bit floating-point
                numbers. This parameter is crucial for ensuring that the index
                accommodates the vector data appropriately.

        """
        self.name = name
        self.field_name = field_name
        self.type = type


class VectorIndexConfig(IndexConfig):
    SUPPORTED_DISTANCE_STRATEGIES = {
        DistanceStrategy.COSINE,
        DistanceStrategy.EUCLIDEAN_DISTANCE,
        DistanceStrategy.MAX_INNER_PRODUCT,
    }

    def __init__(
        self,
        name: str,
        field_name: str,
        type: str,
        distance_strategy: DistanceStrategy,
        vector_size: int,
        data_type: str = DEFAULT_DATA_TYPE,
    ):
        """
        Initializes the VectorIndexConfig object.

        Args:
            name (str): The unique name for the vector index. This name is used to
                identify and reference the index within the vector storage system.
            field_name (str): The name of the field in the data structure that contains
                the vector data to be indexed. This specifies the target data for indexing.
            type (str): The type of vector index. This parameter determines the indexing
                algorithm or structure to be used (e.g., "FLAT", "HNSW").
            distance_strategy (DistanceStrategy): Enum specifying the metric used to
                calculate the distance or similarity between vectors. Supported strategies
                include COSINE, EUCLIDEAN_DISTANCE (L2), and MAX_INNER_PRODUCT (IP),
                influencing how search results are ranked and returned.
            vector_size (int): The dimensionality of the vectors that will be stored
                and indexed. All vectors must conform to this specified size.
            data_type (str, optional): The data type of the vector elements (e.g., "FLOAT32").
                This specifies the precision and format of the vector data, affecting storage
                requirements and possibly search performance.
        """
        if distance_strategy not in self.SUPPORTED_DISTANCE_STRATEGIES:
            supported_strategies = ", ".join(
                [ds.value for ds in self.SUPPORTED_DISTANCE_STRATEGIES]
            )
            raise ValueError(
                f"Unsupported distance strategy: {distance_strategy}. "
                f"Supported strategies are: {supported_strategies}."
            )

        if data_type.upper() != DEFAULT_DATA_TYPE:
            raise ValueError(f"Unsupported data type: {data_type}")

        if vector_size < 0:
            raise ValueError(f"Unsupported vector size: {vector_size}")

        super().__init__(name, field_name, type)
        self.distance_strategy = distance_strategy
        self.vector_size = vector_size
        self.data_type = data_type

    @property
    def distance_metric(self):
        mapping = {
            DistanceStrategy.EUCLIDEAN_DISTANCE: "L2",
            DistanceStrategy.MAX_INNER_PRODUCT: "IP",
            DistanceStrategy.DOT_PRODUCT: "IP",
            DistanceStrategy.COSINE: "COSINE",
        }
        return mapping[self.distance_strategy]


class HNSWConfig(VectorIndexConfig):
    DEFAULT_VECTOR_SIZE = 128
    DEFAULT_INITIAL_CAP = 10000
    DEFAULT_M = 16
    DEFAULT_EF_CONSTRUCTION = 200
    DEFAULT_EF_RUNTIME = 10

    """
    Configuration class for HNSW (Hierarchical Navigable Small World) vector indexes.
    """

    def __init__(
        self,
        name: str,
        field_name: Optional[str] = None,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        initial_cap: int = DEFAULT_INITIAL_CAP,
        m: int = DEFAULT_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_runtime: int = DEFAULT_EF_RUNTIME,
    ):
        """
        Initializes the HNSWConfig object.

        Args:
            name (str): The unique name for the vector index, serving as an identifier
                within the vector store or database system.
            field_name (str): The name of the field in the dataset that holds the vector
                data to be indexed. This specifies which part of the data structure is
                used for indexing and searching.
            vector_size (int): The dimensionality of the vectors that the index will
                accommodate. All vectors must match this specified size.
            distance_strategy (DistanceStrategy): The metric used for calculating
                distances or similarities between vectors, influencing how search results
                are ranked.
            initial_cap (int): Specifies the initial capacity of the index in terms of
                the number of vectors it can hold, impacting the initial memory allocation.
            m (int): Determines the maximum number of outgoing edges each node in the
                index graph can have, directly affecting the graph's connectivity and
                search performance.
            ef_construction (int): Controls the size of the dynamic candidate list during
                the construction of the index, influencing the index build time and quality.
            ef_runtime (int): Sets the size of the dynamic candidate list during search
                queries, balancing between search speed and accuracy.

        """
        if field_name is None:
            field_name = DEFAULT_VECTOR_FIELD
        super().__init__(
            name, field_name, "HNSW", distance_strategy, vector_size, DEFAULT_DATA_TYPE
        )
        self.initial_cap = initial_cap
        self.m = m
        self.ef_construction = ef_construction
        self.ef_runtime = ef_runtime


class FLATConfig(VectorIndexConfig):
    """
    Configuration class for FLAT vector indexes, utilizing brute-force search.
    """

    DEFAULT_VECTOR_SIZE = 128

    def __init__(
        self,
        name: str,
        field_name: Optional[str] = None,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
    ):
        """
        Initializes the FLATConfig object.

        Args:
            name (str): The unique name for the vector index. This name is used
                to identify the index within the vector store or database.
            field_name (str): The name of the field that contains the vector data
                to be indexed. This field should exist within the data structure
                that stores the documents or vectors.
            vector_size (int): Specifies the dimensionality of the vectors to be
                indexed. All vectors added to this index must conform to this size.
            distance_strategy (DistanceStrategy, optional): Determines the metric
                used to calculate the distance or similarity between vectors during
                search operations.
        """
        if field_name is None:
            field_name = DEFAULT_VECTOR_FIELD
        super().__init__(
            name, field_name, "FLAT", distance_strategy, vector_size, DEFAULT_DATA_TYPE
        )


class RedisVectorStore(VectorStore):
    def __init__(
        self,
        client: redis.Redis,
        index_name: str,
        embeddings: Embeddings,
        content_field: str = DEFAULT_CONTENT_FIELD,
        vector_field: str = DEFAULT_VECTOR_FIELD,
    ):
        """
        Initialize a RedisVectorStore instance.

        Args:
            client (redis.Redis): The Redis client instance to be used for database
                operations, providing connectivity and command execution against the
                Redis instance.
            index_name (str): The name assigned to the vector index within Redis. This
                name is used to identify the index for operations such as searching and
                indexing.
            embeddings (Embeddings): An instance of an embedding service or model
                capable of generating vector embeddings from document content. This
                service is utilized to convert text documents into vector representations
                for storage and search.
            content_field (str, optional): The field within the Redis HASH where document
                content is stored. This field is read to obtain document text for
                embedding during indexing operations. Defaults to 'page_content', which
                can be overridden if document content is stored under a different field.
            vector_field (str, optional): The field within the Redis HASH designated for
                storing the vector embedding of the document. This field is used both
                when adding new documents to the store and when retrieving or searching
                documents based on their vector embeddings. Defaults to 'vector'.
        """
        if not isinstance(client, redis.Redis):
            raise ValueError(
                "A Redis 'client' must be provided to initialize RedisVectorStore"
            )

        if not isinstance(index_name, str):
            raise ValueError(
                "A 'index_name' must be provided to initialize RedisVectorStore"
            )

        if not isinstance(embeddings, Embeddings):
            raise ValueError(
                "An 'embeddings' must be provided to initialize RedisVectorStore"
            )

        self._client = client
        self.index_name = index_name
        self.embeddings_service = embeddings
        self.key_prefix = self.get_key_prefix(index_name)
        self.content_field = content_field
        self.vector_field = vector_field
        self.encoding = client.get_encoder().encoding

    @staticmethod
    def get_key_prefix(index_name: str, key_prefix: Optional[str] = None):
        return key_prefix if key_prefix is not None else index_name

    @staticmethod
    def _is_json_parsable(s: str) -> bool:
        try:
            json.loads(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def init_index(client: redis.Redis, index_config: IndexConfig):
        """
        Initializes a named VectorStore index in Redis with specified configurations.
        """
        if not isinstance(index_config, VectorIndexConfig):
            raise ValueError("index_config must be an instance of VectorIndexConfig")

        # Preparing the command string to avoid long lines
        if isinstance(index_config, HNSWConfig):
            command = (
                f"FT.CREATE {index_config.name} ON HASH PREFIX 1 {RedisVectorStore.get_key_prefix(index_config.name)} "
                f"SCHEMA {index_config.field_name} VECTOR {index_config.type} "
                f"14 TYPE {index_config.data_type} DIM {index_config.vector_size} "
                f"DISTANCE_METRIC {index_config.distance_metric} "
                f"TYPE {index_config.data_type} "
                f"M {index_config.m} "
                f"EF_CONSTRUCTION {index_config.ef_construction} "
                f"EF_RUNTIME {index_config.ef_runtime}"
            )
        else:
            command = (
                f"FT.CREATE {index_config.name} ON HASH PREFIX 1 {RedisVectorStore.get_key_prefix(index_config.name)} "
                f"SCHEMA {index_config.field_name} VECTOR {index_config.type} "
                f"8 TYPE {index_config.data_type} DIM {index_config.vector_size} "
                f"DISTANCE_METRIC {index_config.distance_metric} "
                f"TYPE {index_config.data_type} "
            )

        try:
            client.execute_command(command)
        except redis.exceptions.ResponseError as e:
            if re.match(r"Redis module key \w+ already exists", str(e)):
                logger.info("Index already exists, skipping creation.")
            else:
                raise

        # TODO: When "Redis module key \w+ already exists" is caught, we should
        # call FT.INFO to validate if the index properties match the arguments.
        # If not, an exception should be thrown. This check is pending support
        # for FT.INFO in the client library.

    @staticmethod
    def drop_index(client: redis.Redis, index_name: str, index_only: bool = True):
        """
        Drops an index from the Redis database. Optionally, it can also delete
        the documents associated with the index.

        Args:
            client (Redis): The Redis client instance used to connect to the database.
                This client provides the necessary commands to interact with the database.
            index_name (str): The name of the index to be dropped. This name must exactly
                match the name of the existing index in the Redis database.
            index_only (bool, optional): A flag indicating whether to drop only the index
                structure (True) or to also delete the documents associated with the index (False).
                Defaults to True, implying that only the index will be deleted.

        Raises:
            redis.RedisError: If any Redis-specific error occurs during the operation. This
                includes connection issues, authentication failures, or errors from executing
                the command to drop the index. Callers should handle these exceptions to
                manage error scenarios gracefully.
        """
        if index_only == False:
            raise ValueError("Not supported")

        command = f"FT.DROPINDEX {index_name} {'DD' if not index_only else ''}".strip()
        client.execute_command(command)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: Optional[int] = 1000,
        **kwargs: Any,
    ) -> List[str]:
        """
        Adds a collection of texts along with their metadata to a vector store,
        generating unique keys for each entry if not provided.

        Args:
            texts (Iterable[str]): An iterable collection of text documents to be added to the vector store.
            metadatas (Optional[List[dict]], optional): An optional list of metadata dictionaries,
                where each dictionary corresponds to a text document in the same order as the `texts` iterable.
                Each metadata dictionary should contain key-value pairs representing the metadata attributes
                for the associated text document.
            ids (Optional[List[str]], optional): An optional list of unique identifiers for each text document.
                If not provided, the system will generate unique identifiers for each document. If provided,
                the length of this list should match the length of `texts`.
            batch_size (int, optional): The number of documents to process in a single batch operation.
                This parameter helps manage memory and performance when adding a large number of documents.
                Defaults to 1000.

        Returns:
            List[str]: A list containing the unique keys or identifiers for each added document. These keys
                can be used to retrieve or reference the documents within the vector store.

        Note:
            If both 'ids' and 'metadatas' are provided, they must be of the same length as the `texts`
            iterable to ensure each document is correctly associated with its metadata and identifier.
        """

        # Generate ids if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Check if both ids and metadatas are provided and have the same length
        if ids is not None:
            if len(ids) != len(list(texts)):
                raise ValueError("The length of 'ids' and 'texts' must be the same.")

        if not metadatas:
            metadatas = [{} for _ in texts]

        if metadatas is not None:
            if len(metadatas) != len(list(texts)):
                raise ValueError(
                    "The length of 'metadatas' and 'texts' must be the same."
                )

        if not batch_size or batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        # Generate embeddings for all documents
        embeddings = self.embeddings_service.embed_documents(list(texts))

        new_ids = []
        pipeline = self._client.pipeline(transaction=False)
        for i, bundle in enumerate(zip(ids, texts, embeddings, metadatas), start=1):
            id, text, embedding, metadata = bundle
            new_id = self.key_prefix + id

            # Initialize the mapping with content and vector fields
            mapping = {
                self.content_field: text,
                self.vector_field: np.array(embedding)
                .astype(DEFAULT_DATA_TYPE.lower())
                .tobytes(),
            }

            # Process metadata: directly add non-dict items, JSON-serialize dict items
            for meta_key, meta_value in metadata.items():
                if isinstance(meta_value, dict):
                    # If the value is a dict, JSON-serialize it
                    mapping[meta_key] = json.dumps(meta_value)
                else:
                    # Directly add non-dict items
                    mapping[meta_key] = str(meta_value)

            # Add the document to the Redis hash
            pipeline.hset(new_id, mapping=mapping)
            new_ids.append(new_id)

            # Ensure to execute any remaining commands in the pipeline after the loop
            if i % batch_size == 0:
                pipeline.execute()

        # Final execution to catch any remaining items in the pipeline
        pipeline.execute()

        logger.info(f"{len(ids)} documents ingested into Redis.")

        return new_ids

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        client: Optional[redis.Redis] = None,
        index_name: Optional[str] = None,
        **kwargs: Any,
    ) -> "RedisVectorStore":
        """
        Creates an instance of RedisVectorStore from provided texts.

        Args:
            texts (List[str]): A list of text documents to be embedded and indexed.
            embedding (Embeddings): An instance capable of generating embeddings for the
                provided text documents.
            metadatas (Optional[List[dict]]): A list of dictionaries where each dictionary
                contains metadata corresponding to each text document in `texts`. If provided,
                the length of `metadatas` must match the length of `texts`.
            ids (Optional[List[str]], optional): An optional list of unique identifiers for
                each text document.  If not provided, the system will generate unique identifiers
                for each document. If provided, the length of this list should match the length
                of `texts`.
            client (redis.Redis): The Redis client instance to be used for database
                operations, providing connectivity and command execution against the
                Redis instance.
            index_name (str): The name assigned to the vector index within Redis. This
                name is used to identify the index for operations such as searching and
                indexing.
            **kwargs (Any): Additional keyword arguments

        Returns:
            RedisVectorStore: An instance of RedisVectorStore that has been populated with
                the embeddings of the provided texts, along with their associated metadata.
        """

        if not isinstance(client, redis.Redis):
            raise ValueError(
                "A 'client' must be provided to initialize RedisVectorStore"
            )

        if not isinstance(index_name, str):
            raise ValueError(
                "A 'index_name' must be provided to initialize RedisVectorStore"
            )

        # Initialize RedisVectorStore instance
        instance = cls(
            client,
            index_name,
            embedding,
            **kwargs,
        )

        # Add texts and their corresponding metadata to the instance
        instance.add_texts(texts, metadatas, ids)

        return instance

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not ids:  # Check if ids list is empty or None
            logger.info("No IDs provided for deletion.")
            return False

        try:
            self._client.delete(*ids)
            logger.info("Entries deleted.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete entries: {e}")
            return False

    def _similarity_search_by_vector_with_score_and_embeddings(
        self, query_embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float, List[float]]]:
        """
        Performs a similarity search by a vector with score and embeddings, offering
        various customization options via keyword arguments.

        Args:
            query_embedding (List[float]): A list of floats representing the embedding
                vector of the query for similarity search.
            k (int, optional): The number of nearest neighbors to retrieve. Defaults to 4.
            **kwargs (Any): Additional keyword arguments allowing for customization of
                the search operation. Key options include:

                - 'distance_threshold' (float, optional): A threshold value for filtering
                  results based on their distance or score. If not specified directly,
                  it may use 'score_threshold' if provided.

                - 'distance_strategy' (str, optional): Strategy to apply when comparing
                  distances or scores. Uses a default strategy if not specified.

        Returns:
            List[Tuple[Document, float, List[float]]]: A list of tuples, each containing
            a Document object, its distance (score) from the query embedding, and its own
            embedding vector. The Document object includes content and metadata.

        Note:
            The function dynamically adjusts its behavior based on the presence and values
            of keyword arguments. For instance, if a 'distance_threshold' is provided,
            only results meeting this threshold are returned.
        """

        distance_threshold = kwargs.get(
            "distance_threshold", kwargs.get("score_threshold")
        )

        distance_strategy = kwargs.get("distance_strategy", DEFAULT_DISTANCE_STRATEGY)

        query_args = [
            "FT.SEARCH",
            self.index_name,
            f"*=>[KNN {k} @{self.vector_field} $query_vector AS distance]",
            "PARAMS",
            2,
            "query_vector",
            np.array([query_embedding]).astype(DEFAULT_DATA_TYPE.lower()).tobytes(),
            "DIALECT",
            2,
        ]

        initial_results = self._client.execute_command(*query_args)

        logger.info(f"{int((len(initial_results)-1)/2)} documents returned by Redis")

        # Process the results
        final_results: List[Tuple[Document, float, List[float]]] = []

        if not initial_results:
            return final_results

        for i in range(2, len(initial_results), 2):
            page_content: str = ""
            metadata = {}
            distance = 0.0
            embedding: List[float] = []
            for j in range(0, len(initial_results[i]), 2):
                key = initial_results[i][j].decode(self.encoding)
                value = initial_results[i][j + 1]
                if key == self.content_field:
                    page_content = value.decode(self.encoding)
                elif key == self.vector_field:
                    embedding = np.frombuffer(
                        value, dtype=DEFAULT_DATA_TYPE.lower()
                    ).tolist()
                elif key == "distance":
                    distance = float(value.decode(self.encoding))
                else:
                    if isinstance(value, bytes) and self._is_json_parsable(
                        value.decode(self.encoding)
                    ):
                        metadata[key] = json.loads(value.decode(self.encoding))
                    else:
                        metadata[key] = value.decode(self.encoding)

            final_results.append(
                (
                    Document(page_content=page_content, metadata=metadata),
                    distance,
                    embedding,
                )
            )

        if distance_threshold is not None:
            cmp = (
                operator.ge
                if distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT
                else operator.le
            )
            final_results = [
                (doc, distance, embedding)
                for doc, distance, embedding in final_results
                if cmp(distance, distance_threshold)
            ]

        # Directly sort final_results based on distance, applying the determined sort order
        final_results = sorted(
            final_results,
            key=lambda d: d[1],
            reverse=distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT,
        )

        return final_results

    def _similarity_search_by_vector_with_score(
        self, query_embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        initial_results = self._similarity_search_by_vector_with_score_and_embeddings(
            query_embedding, k, **kwargs
        )
        # Extract just the Document objects from the search results
        return [(doc, embedding) for doc, embedding, _ in initial_results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Performs a similarity search using the given query, returning documents and their similarity scores.

        Args:
            query (str): The query string to search for.
            k (int): The number of closest documents to return.
            **kwargs (Any): Additional keyword arguments for future use.
        Returns:
            List[Tuple[Document, float]]: A ranked list of tuples, each containing a Document object and its
                corresponding similarity score. The list includes up to 'k' entries, representing the
                documents most relevant to the query according to the similarity scores.
        """
        # Embed the query using the embedding function
        query_embedding = self.embeddings_service.embed_query(query)
        return self._similarity_search_by_vector_with_score(
            query_embedding, k, **kwargs
        )

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Performs a similarity search for the given embedding and returns the top k
        most similar Document objects, discarding their similarity scores.

        Args:
            embedding (List[float]): The query embedding for the similarity search.
            k (int): The number of top documents to return.
            **kwargs (Any): Additional keyword arguments to pass to the search.

        Returns:
            List[Document]: A list containing up to 'k' Document objects, ranked by their
                similarity to the query. These documents represent the most relevant
                results found by the search operation, subject to the additional constraints
                and configurations specified.
        """
        initial_results = self._similarity_search_by_vector_with_score(
            embedding, k, **kwargs
        )
        # Extract just the Document objects from the search results
        return [doc for doc, _ in initial_results]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Conducts a similarity search based on the specified query, returning a list
        of the top 'k' documents that are most similar to the query.

        Args:
            query (str): The text query based on which similar documents are to be retrieved.
            k (int): Specifies the number of documents to return, effectively setting a limit
                on the size of the result set. Defaults to 4.
            **kwargs (Any): A flexible argument allowing for the inclusion of additional
                search parameters or options.

        Returns:
            List[Document]: A list containing up to 'k' Document objects, ranked by their
                similarity to the query. These documents represent the most relevant results
                found by the search operation, subject to the additional constraints and
                configurations specified.

        Raises:
            ValueError: If any of the provided search parameters are invalid or if the search
                operation encounters an error due to misconfiguration or execution issues within
                the search backend.
        """
        # Embed the query using the embedding function
        query_embedding = self.embeddings_service.embed_query(query)
        return self.similarity_search_by_vector(query_embedding, k, **kwargs)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Performs a search to find documents that are both relevant to the query and diverse
        among each other based on Maximal Marginal Relevance (MMR).

        The MMR algorithm optimizes a combination of relevance to the query and diversity
        among the results, controlled by the lambda_mult parameter.

        Args:
            query (str): The query string used to find similar documents.
            k (int): The number of documents to return.
            fetch_k (int): The number of documents to fetch for consideration. This should
                be larger than k to allow for diversity calculation.
            lambda_mult (float): Controls the trade-off between relevance and diversity.
                Ranges from 0 (max diversity) to 1 (max relevance).
            **kwargs: Additional keyword arguments

        Returns:
            List[Document]: A list of document objects selected based on maximal marginal relevance.

        Raises:
            ValueError: If lambda_mult is not in the range [0, 1].
        """
        # Validate the lambda_mult parameter to ensure it's within the valid range.
        if not 0 <= lambda_mult <= 1:
            raise ValueError("lambda_mult must be between 0 and 1.")

        # Embed the query using a hypothetical method to convert text to vector.
        query_embedding = self.embeddings_service.embed_query(query)

        # Fetch initial documents based on query embedding.
        initial_results = self._similarity_search_by_vector_with_score_and_embeddings(
            query_embedding,
            k=fetch_k,
            **kwargs,
        )

        inital_embeddings = [embedding for _, _, embedding in initial_results]

        # Calculate MMR to select diverse and relevant documents.
        selected_indices = maximal_marginal_relevance(
            np.array(query_embedding), inital_embeddings, lambda_mult=lambda_mult, k=k
        )

        return [initial_results[i][0] for i in selected_indices]
