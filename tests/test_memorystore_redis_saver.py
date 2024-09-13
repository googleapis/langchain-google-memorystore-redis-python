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
from typing import Union

import pytest
import redis
import redis.cluster
from langchain_core.documents.base import Document

from langchain_google_memorystore_redis.saver import MemorystoreDocumentSaver


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


def get_all_keys(prefix: str, client: Union[redis.Redis, redis.cluster.RedisCluster]):
    if isinstance(client, redis.Redis):
        return client.keys(f"{prefix}*")
    else:
        return client.keys(
            f"{prefix}*", target_nodes=redis.cluster.RedisCluster.ALL_NODES
        )


@pytest.mark.parametrize(
    "client",
    [
        redis.from_url(get_env_var("REDIS_URL", "URL of the Redis instance")),
        redis.cluster.RedisCluster.from_url(
            get_env_var("REDIS_CLUSTER_URL", "URL of the Redis cluster")
        ),
    ],
    ids=["redis_standalone", "redis_cluster"],
)
@pytest.mark.parametrize(
    "page_content,metadata,content_field,metadata_fields",
    [
        (
            '"content1"',
            {"key1": "doc1_value1", "key2": "doc1_value2"},
            "page_content",
            None,
        ),
        (
            '"content2"',
            {"key1": {'"nested_key"': {'"double_nested"': '"doc2_value1"'}}},
            "special_page_content",
            None,
        ),
        (
            '"content3"',
            {"key1": {"k": "not_in_filter"}, "key2": {'"key"': "in_filter"}},
            "page_content",
            set(["key2"]),
        ),
    ],
)
def test_document_saver_add_documents_one_doc(
    client, page_content, metadata, content_field, metadata_fields
):
    prefix = "prefix:"

    saver = MemorystoreDocumentSaver(
        client=client,
        key_prefix=prefix,
        content_field=content_field,
        metadata_fields=metadata_fields,
    )

    doc = Document.construct(page_content=page_content, metadata=metadata)
    doc_id = "doc"
    saver.add_documents([doc], [doc_id])

    # Only verify the metadata keys given in the metadata_fields
    metadata_to_verify = {}
    for k, v in metadata.items():
        if not metadata_fields or k in metadata_fields:
            metadata_to_verify[k] = v

    verify_stored_values(
        client,
        prefix + doc_id,
        page_content,
        content_field,
        metadata_to_verify,
    )

    assert get_all_keys(prefix, client) != []
    saver.delete()
    assert get_all_keys(prefix, client) == []


@pytest.mark.parametrize(
    "client",
    [
        redis.from_url(get_env_var("REDIS_URL", "URL of the Redis instance")),
        redis.cluster.RedisCluster.from_url(
            get_env_var("REDIS_CLUSTER_URL", "URL of the Redis cluster")
        ),
    ],
    ids=["redis_standalone", "redis_cluster"],
)
def test_document_saver_add_documents_multiple_docs(client):
    prefix = "multidocs:"
    content_field = "page_content"
    saver = MemorystoreDocumentSaver(
        client=client,
        key_prefix=prefix,
        content_field=content_field,
    )
    docs = []
    number_of_docs = 10
    doc_ids = [f"{i}" for i in range(number_of_docs)]
    for content in range(number_of_docs):
        docs.append(
            Document.construct(
                page_content=f"{content}",
                metadata={"metadata": f"meta: {content}"},
            )
        )
    saver = MemorystoreDocumentSaver(
        client=client,
        key_prefix=prefix,
        content_field=content_field,
    )
    saver.add_documents(docs, ids=doc_ids)

    for i, doc_id in enumerate(doc_ids):
        verify_stored_values(
            client,
            prefix + doc_id,
            f"{i}",
            content_field,
            {"metadata": f"meta: {i}"},
        )

    assert len(get_all_keys(prefix, client)) == number_of_docs
    saver.delete(ids=doc_ids, batch_size=3)
    assert get_all_keys(prefix, client) == []


def verify_stored_values(
    client: Union[redis.Redis, redis.cluster.RedisCluster],
    key: str,
    page_content: str,
    content_field: str,
    metadata_to_verify: dict,
):
    stored_value = client.hgetall(key)
    assert isinstance(stored_value, dict)
    assert len(stored_value) == 1 + len(metadata_to_verify)

    for k, v in stored_value.items():
        decoded_value = v.decode()
        if k == content_field.encode():
            assert page_content == decoded_value
        else:
            assert (
                metadata_to_verify[k.decode()] == json.loads(decoded_value)
                if is_json_parsable(decoded_value)
                else decoded_value
            )


def is_json_parsable(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v
