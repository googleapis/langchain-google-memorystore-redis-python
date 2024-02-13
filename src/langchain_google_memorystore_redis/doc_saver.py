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
        content_field: str,
        metadata_fields: Optional[Set[str]] = None,
        key_prefix: Optional[str] = None,
    ):
        """Initializes the Document Saver for Memorystore for Redis.

        Args:
            client: A redis.Redis client object.
            content_field: The field of the hash that Redis uses to store the
              page_content of the Document.
            metadata_fields: The metadata fields of the Document that will be
              stored in the Redis. If None, Redis stores all metadata fields.
            key_prefix: A prefix for the key used to store history for this
              session.
        """

        self._redis = client
        self._content_field = content_field
        self._metadata_fields = metadata_fields
        self._key_prefix = key_prefix if key_prefix else ""

    def add_documents(
        self,
        documents: Sequence[Document],
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        """Save a list of Documents to Redis.

        Args:
            documents: A List of Documents.
            ids: The keys in Redis used to store the Documents. If specified,
              the length of the IDs must be the same as Documents. If not
              specified, random UUIDs with prefix are generated to store each
              Document.
        """
        doc_ids = ids
        if not doc_ids:
            doc_ids = [self._key_prefix + str(uuid.uuid4()) for _ in documents]
        if len(documents) != len(doc_ids):
            raise ValueError("The length of documents must match the length of the IDs")

        for i, doc in enumerate(documents):
            mapping = self._filter_metadata_by_fields(doc.metadata)
            mapping.update({self._content_field: doc.page_content})

            # Remove existing key in Redis to avoid reusing the doc ID.
            self._redis.delete(doc_ids[i])
            self._redis.hset(doc_ids[i], mapping=mapping)

    def _filter_metadata_by_fields(self, metadata: Optional[dict]) -> dict:
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
