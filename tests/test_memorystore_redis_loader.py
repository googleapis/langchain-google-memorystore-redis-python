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

from langchain_google_memorystore_redis.document_loader import MemorystoreDocumentLoader
from langchain_google_memorystore_redis.document_saver import MemorystoreDocumentSaver


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
def test_document_loader_one_doc(
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
    doc_id = "saved_doc"
    saver.add_documents([doc], [doc_id])

    loader = MemorystoreDocumentLoader(
        client=client,
        key_prefix=prefix,
        content_fields=set([content_field]),
        metadata_fields=metadata_fields,
    )
    loaded_docs = loader.load()
    expected_doc = (
        doc
        if not metadata_fields
        else Document.construct(
            page_content=page_content,
            metadata={k: metadata[k] for k in metadata_fields},
        )
    )
    assert loaded_docs == [expected_doc]
    client.delete(prefix + doc_id)


def test_document_loader_multiple_docs():
    client = redis.from_url(get_env_var("REDIS_URL", "URL of the Redis instance"))

    prefix = "multidocs:"
    content_field = "page_content"
    saver = MemorystoreDocumentSaver(
        client=client,
        key_prefix=prefix,
        content_field=content_field,
    )
    docs = []
    for content in range(10):
        docs.append(
            Document.construct(
                page_content=f"{content}",
                metadata={"metadata": f"meta: {content}"},
            )
        )
    saver.add_documents(docs)

    loader = MemorystoreDocumentLoader(
        client=client,
        key_prefix=prefix,
        content_fields=set([content_field]),
    )
    loaded_docs = []
    for doc in loader.lazy_load():
        loaded_docs.append(doc)
    assert sorted(loaded_docs, key=lambda d: d.page_content) == docs
    saver.delete()


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v
