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

import pytest
import redis
from langchain_core.documents.base import Document

from langchain_google_memorystore_redis.doc_saver import MemorystoreDocumentSaver


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
def test_doc_saver_add_documents_one_doc(
    page_content, metadata, content_field, metadata_fields
):
    client = redis.from_url(get_env_var("REDIS_URL", "URL of the Redis instance"))
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

    client.delete(prefix + doc_id)


def verify_stored_values(
    client: redis.Redis,
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
