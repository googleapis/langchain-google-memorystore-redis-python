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
import uuid
from typing import Optional, Sequence, Set, Union

import redis
from langchain_core.documents.base import Document


class MemorystoreDocumentSaver:
    """Document Saver for Cloud Memorystore for Redis database."""

    def __init__(
        self,
        client: redis.Redis,
        key_prefix: str,
        content_field: str,
        metadata_fields: Optional[Set[str]] = None,
    ):
        """Initializes the Document Saver for Memorystore for Redis.

        Args:
            client: A redis.Redis client object.
            key_prefix: A prefix for the keys to store Documents in Redis.
            content_field: The field of the hash that Redis uses to store the
                page_content of the Document.
            metadata_fields: The metadata fields of the Document that will be
                stored in the Redis. If None, Redis stores all metadata fields.
        """

        self._redis = client
        if not key_prefix:
            raise ValueError("key_prefix must not be empty")
        self._key_prefix = key_prefix
        self._content_field = content_field
        self._metadata_fields = metadata_fields

    def add_documents(
        self,
        documents: Sequence[Document],
        ids: Optional[Sequence[str]] = None,
        batch_size: int = 1000,
    ) -> None:
        """Save a list of Documents to Redis.

        Args:
            documents: A List of Documents.
            ids: The list of suffixes for keys that Redis uses to store the
                Documents. If specified, the length of the IDs must be the same
                as Documents. If not specified, random UUIDs appended after
                prefix are used to store each Document.
            batch_size: The number of documents to process in a single batch
                operation. This parameter helps manage memory and performance
                when adding a large number of documents. Defaults to 1000.
        """
        if ids and len(documents) != len(ids):
            raise ValueError("The length of documents must match the length of the IDs")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        doc_ids = ids if ids else [str(uuid.uuid4()) for _ in documents]
        doc_ids = [self._key_prefix + doc_id for doc_id in doc_ids]
        if self._redis.exists(*doc_ids):
            raise RuntimeError(
                "At least one of the keys exist(s) in Redis. Please delete the "
                "existing key(s) or select another prefix to save documents."
            )

        pipeline = self._redis.pipeline(transaction=False)
        for i, doc in enumerate(documents):
            mapping = self._filter_metadata_by_fields(doc.metadata)
            mapping.update({self._content_field: doc.page_content})

            pipeline.hset(doc_ids[i], mapping=mapping)
            if (i + 1) % batch_size == 0 or i == len(documents) - 1:
                pipeline.execute()

    def delete(
        self,
        ids: Optional[Sequence[str]] = None,
        batch_size: int = 1000,
    ) -> None:
        """Delete a list of Documents from Redis.

        Args:
            ids: The list of suffixes for keys that Redis uses to store the
                Documents. If not specified, all Documents with the initialized
                prefix are deleted.
            batch_size: The number of delete commands to process in a single
                batch operation. This parameter helps manage memory and
                performance when deleting a large number of documents. Defaults
                to 1000.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        if ids:
            doc_ids = [self._key_prefix + doc_id for doc_id in ids]
            for i in range(0, len(doc_ids), batch_size):
                self._redis.delete(*doc_ids[i : i + batch_size])
            return

        pipeline = self._redis.pipeline(transaction=False)
        for i, key in enumerate(
            self._redis.scan_iter(match=f"{self._key_prefix}*", _type="HASH")
        ):
            pipeline.delete(key)
            if i % batch_size == 0:
                pipeline.execute()
        pipeline.execute()

    def _filter_metadata_by_fields(self, metadata: Optional[dict] = None) -> dict:
        """Filter metadata fields to be stored in Redis.

        Args:
            metadata: The metadata field of a Document object.

        Returns:
            dict: A subset dict of the metadata that only contains the fields
                specified in the initialization of the saver. The value of each
                metadata key is serialized by JSON if it is a dict.
        """
        if not metadata:
            return {}
        filtered_fields = (
            self._metadata_fields & metadata.keys()
            if self._metadata_fields
            else metadata.keys()
        )
        filtered_metadata = {
            k: self._jsonify_if_dict(metadata[k]) for k in filtered_fields
        }
        return filtered_metadata

    @staticmethod
    def _jsonify_if_dict(s: Union[str, dict]) -> str:
        return s if isinstance(s, str) else json.dumps(s)
