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
from typing import Iterator, List, Optional, Sequence, Set, Union

import redis
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents.base import Document


class MemorystoreDocumentLoader(BaseLoader):
    """Document Loader for Cloud Memorystore for Redis database."""

    def __init__(
        self,
        client: redis.Redis,
        key_prefix: str,
        content_fields: Set[str],
        metadata_fields: Optional[Set[str]] = None,
    ):
        """Initializes the Document Loader for Memorystore for Redis.

        Args:
            client: A redis.Redis client object.
            key_prefix: A prefix for the keys to store Documents in Redis.
            content_fields: The set of fields of the hash that Redis uses to
               store the page_content of the Document. If more than one field
               are specified, a JSON encoded dict containing the fields as top
               level keys will be filled in the page_content of the Documents.
            metadata_fields: The metadata fields of the Document that will be
               stored in the Redis. If None, Redis stores all metadata fields.
        """

        self._redis = client
        self._content_fields = content_fields
        self._metadata_fields = metadata_fields
        if metadata_fields and len(content_fields & metadata_fields):
            raise ValueError(
                "Fields {} are specified in both content_fields and"
                " metadata_fields.".format(content_fields & metadata_fields)
            )
        self._key_prefix = key_prefix if key_prefix else ""
        self._encoding = client.get_encoder().encoding

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load the Documents and yield them one by one."""
        for key in self._redis.scan_iter(match=f"{self._key_prefix}*", _type="HASH"):
            doc = {}
            stored_value = self._redis.hgetall(key)
            if not isinstance(stored_value, dict):
                raise RuntimeError(f"{key} returns unexpected {stored_value}")
            decoded_value = {
                k.decode(self._encoding): v.decode(self._encoding)
                for k, v in stored_value.items()
            }

            if len(self._content_fields) == 1:
                doc["page_content"] = decoded_value[next(iter(self._content_fields))]
            else:
                doc["page_content"] = json.dumps(
                    {k: decoded_value[k] for k in self._content_fields}
                )

            filtered_fields = (
                self._metadata_fields if self._metadata_fields else decoded_value.keys()
            )
            filtered_fields = filtered_fields - self._content_fields
            doc["metadata"] = {
                k: self._decode_if_json_parsable(decoded_value[k])
                for k in filtered_fields
            }

            yield Document.construct(**doc)

    def load(self) -> List[Document]:
        """Load all Documents at once."""
        return list(self.lazy_load())

    @staticmethod
    def _decode_if_json_parsable(s: str) -> Union[str, dict]:
        """Decode a JSON string to a dict if it is JSON."""
        try:
            decoded = json.loads(s)
            return decoded
        except ValueError:
            pass
        return s
