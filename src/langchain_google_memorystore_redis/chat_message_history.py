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
from typing import List, Optional

import redis
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


class MemorystoreChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Cloud Memorystore for Redis database."""

    def __init__(
        self,
        client: redis.Redis,
        session_id: str,
        ttl: Optional[int] = None,
    ):
        """Initializes the chat message history for Memorystore for Redis.

        Args:
            client: A redis.Redis client object.
            session_id: A string that uniquely identifies the chat history.
            ttl: Specifies the time in seconds after which the session will
                expire and be eliminated from the Redis instance since the most
                recent message is added.
        """

        self._redis = client
        self._key = session_id
        self._ttl = ttl
        self._encoding = client.get_encoder().encoding

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve all messages chronologically stored in this session."""
        all_elements = self._redis.lrange(self._key, 0, -1)

        if not isinstance(all_elements, list):
            raise TypeError("Expected a list from `lrange` but got a different type.")

        loaded_messages = messages_from_dict(
            [json.loads(e.decode(self._encoding)) for e in all_elements]
        )
        return loaded_messages

    def add_message(self, message: BaseMessage) -> None:
        """Append one message to this session."""
        self._redis.rpush(self._key, json.dumps(message_to_dict(message)))
        if self._ttl:
            self._redis.expire(self._key, self._ttl)

    def clear(self) -> None:
        """Clear all messages in this session."""
        self._redis.delete(self._key)
