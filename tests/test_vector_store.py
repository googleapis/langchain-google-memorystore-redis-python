# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
import uuid

import numpy
import pytest
import redis
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain_core.documents.base import Document

from langchain_google_memorystore_redis import (
    DistanceStrategy,
    HNSWConfig,
    RedisVectorStore,
    VectorIndexConfig,
)


def test_vector_store_init_index():
    client = redis.from_url(get_env_var("REDIS_URL", "URL of the Redis instance"))
    index_name = str(uuid.uuid4())

    index_config = HNSWConfig(
        name=index_name, distance_strategy=DistanceStrategy.COSINE, vector_size=128
    )

    assert not check_index_exists(client, index_name, index_config)
    RedisVectorStore.init_index(client=client, index_config=index_config)
    assert check_index_exists(client, index_name, index_config)
    RedisVectorStore.drop_index(client=client, index_name=index_name)
    assert not check_index_exists(client, index_name, index_config)
    client.flushall()


@pytest.mark.parametrize(
    "texts,metadatas,ids",
    [
        # Test case 1: Basic scenario with texts only
        (["text1", "text2"], None, None),
        # Test case 2: Texts with metadatas
        (["text1", "text2"], [{"meta1": "data1"}, {"meta2": "data2"}], None),
        # Test case 3: Texts with metadatas and ids
        (["text1", "text2"], [{"meta1": "data1"}, {"meta2": "data2"}], ["id1", "id2"]),
        # Test case 4: Texts with ids only
        (["text1", "text2"], None, ["id1", "id2"]),
        # Additional test cases can be added as needed
    ],
)
def test_vector_store_add_texts(texts, metadatas, ids):
    client = redis.from_url(get_env_var("REDIS_URL", "URL of the Redis instance"))

    # Initialize the vector index
    index_name = str(uuid.uuid4())
    index_config = HNSWConfig(
        name=index_name, distance_strategy=DistanceStrategy.COSINE, vector_size=128
    )
    RedisVectorStore.init_index(client=client, index_config=index_config)

    # Insert the documents
    rvs = RedisVectorStore(
        client=client, index_name=index_name, embeddings=FakeEmbeddings(size=128)
    )
    returned_ids = rvs.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    original_metadatas = metadatas if metadatas is not None else [None] * len(texts)
    original_ids = ids if ids is not None else [""] * len(texts)

    # Validate the results
    for original_id, text, original_metadata, returned_id in zip(
        original_ids, texts, original_metadatas, returned_ids
    ):
        expected_id = f"{index_name}{original_id}"
        # Check if original_id is empty and adjust assertion accordingly
        if original_id == "":
            assert returned_id.startswith(
                expected_id
            ), f"Returned ID {returned_id} does not start with expected prefix {expected_id}"
        else:
            assert (
                returned_id == expected_id
            ), f"Returned ID {returned_id} does not match expected {expected_id}"

        # Fetch the record from Redis
        hash_record = client.hgetall(returned_id)

        # Validate page_content
        fetched_page_content = hash_record[b"page_content"].decode("utf-8")
        assert fetched_page_content == text, "Page content does not match"

        # Validate vector embedding
        vector = numpy.frombuffer(hash_record[b"vector"], dtype=numpy.float32)
        assert (
            len(vector) == 128
        ), f"Decoded 'vector' length is {len(vector)}, expected 128"

        # Iterate over each key-value pair in the hash_record
        fetched_metadata = {}
        for key, value in hash_record.items():
            # Decode the key from bytes to string
            key_decoded = key.decode("utf-8")

            # Skip 'page_content' and 'vector' keys, include all others in fetched_metadata
            if key_decoded not in ["page_content", "vector"]:
                # Decode the value from bytes to string or JSON as needed
                try:
                    # Attempt to load JSON content if applicable
                    value_decoded = json.loads(value.decode("utf-8"))
                except json.JSONDecodeError:
                    # Fallback to simple string decoding if it's not JSON
                    value_decoded = value.decode("utf-8")

                # Add the decoded key-value pair to fetched_metadata
                fetched_metadata[key_decoded] = value_decoded

        if original_metadata is None:
            original_metadata = {}

        assert fetched_metadata == original_metadata, "Metadata does not match"

    # Verify no extra keys are present
    all_keys = [key.decode("utf-8") for key in client.keys(f"{index_name}*")]
    # Currently RedisQuery stores the index schema as a key using the index_name
    assert len(all_keys) == len(returned_ids) + 1, "Found unexpected keys in Redis"

    # Clena up
    RedisVectorStore.drop_index(client=client, index_name=index_name)
    client.flushall()


def test_vector_store_knn_query():

    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A clever fox outwitted the guard dog to sneak into the farmyard at night",
        "Exploring the mysteries of deep space and black holes",
        "Delicious recipes for homemade pasta and pizza",
        "Advanced techniques in machine learning and artificial intelligence",
        "Sustainable living: Tips for reducing your carbon footprint",
    ]

    client = redis.from_url(get_env_var("REDIS_URL", "URL of the Redis instance"))

    # Initialize the vector index
    index_name = str(uuid.uuid4())
    index_config = HNSWConfig(
        name=index_name, distance_strategy=DistanceStrategy.COSINE, vector_size=128
    )
    RedisVectorStore.init_index(client=client, index_config=index_config)

    # Insert the documents
    rvs = RedisVectorStore(
        client=client, index_name=index_name, embeddings=FakeEmbeddings(size=128)
    )
    rvs.add_texts(texts=texts)

    # Validate knn query
    query_result = rvs.similarity_search(query="fox dog", k=2)
    assert len(query_result) == 2, "Expected 2 documents to be returned"

    # Clean up
    RedisVectorStore.drop_index(client=client, index_name=index_name)
    client.flushall()


@pytest.mark.parametrize(
    "distance_strategy, distance_threshold",
    [
        (DistanceStrategy.COSINE, 0.8),
        (DistanceStrategy.MAX_INNER_PRODUCT, 1.0),
        (DistanceStrategy.EUCLIDEAN_DISTANCE, 2.0),
    ],
)
def test_vector_store_range_query(distance_strategy, distance_threshold):

    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A clever fox outwitted the guard dog to sneak into the farmyard at night",
        "Exploring the mysteries of deep space and black holes",
        "Delicious recipes for homemade pasta and pizza",
        "Advanced techniques in machine learning and artificial intelligence",
        "Sustainable living: Tips for reducing your carbon footprint",
    ]

    client = redis.from_url(get_env_var("REDIS_URL", "URL of the Redis instance"))

    # Initialize the vector index
    index_name = str(uuid.uuid4())
    index_config = HNSWConfig(
        name=index_name, distance_strategy=distance_strategy, vector_size=128
    )
    RedisVectorStore.init_index(client=client, index_config=index_config)

    # Insert the documents
    rvs = RedisVectorStore(
        client=client, index_name=index_name, embeddings=FakeEmbeddings(size=128)
    )
    rvs.add_texts(texts=texts)

    # Validate range query
    query_result = rvs.similarity_search_with_score(
        query="dog",
        k=3,
        distance_strategy=distance_strategy,
        distance_threshold=distance_threshold,
    )
    assert len(query_result) <= 3, "Expected less than 3 documents to be returned"
    for _, score in query_result:
        if distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            assert (
                score > distance_threshold
            ), f"Score {score} is not greater than {distance_threshold} for {distance_strategy}"
        else:
            assert (
                score < distance_threshold
            ), f"Score {score} is not less than {distance_threshold} for {distance_strategy}"

    # Clean up
    RedisVectorStore.drop_index(client=client, index_name=index_name)
    client.flushall()


def check_index_exists(
    client: redis.Redis, index_name: str, index_config: VectorIndexConfig
) -> bool:

    try:
        index_info = client.ft(index_name).info()
    except:
        return False

    return (
        index_info["index_name"] == index_name
        and index_info["index_definition"][1] == b"HASH"
    )


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v
