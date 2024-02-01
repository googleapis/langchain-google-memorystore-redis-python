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
import redis
from typing import List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)


class MemorystoreChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Cloud Memorystore for Redis database."""

    def __init__(
        self,
        session_id: str,
        url: str,
        key_prefix: Optional[str] = "langchain:",
        ttl: Optional[int] = None,
    ):
        """Initializes the chat message history for Memorystore for Redis.

        Args:
            session_id: The session ID for this chat message history.
            url: The Redis address to connect to. For example:
                redis://localhost:6379. All supported schemes are in
                https://redis.readthedocs.io/en/latest/connections.html#redis.Redis.from_url.
            key_prefix: A prefix for the key used to store history for this
                session.
            ttl: The expiration time of the whole chat history after the most
                recent add_message was called.
        """

        self.redis = redis.from_url(url)
        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl
        self.key = self.key_prefix + self.session_id

    def __del__(self):
        self.redis.close()

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve all messages chronologically stored in this session."""
        all_elements = self.redis.lrange(self.key, 0, -1)
        messages = messages_from_dict(
            [json.loads(e.decode("utf-8")) for e in all_elements]
        )
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append one message to this session."""
        self.redis.rpush(self.key, json.dumps(message_to_dict(message)))
        if self.ttl:
            self.redis.expire(self.key, self.ttl)

    def clear(self) -> None:
        """Clear all messages in this session."""
        self.redis.delete(self.key)
