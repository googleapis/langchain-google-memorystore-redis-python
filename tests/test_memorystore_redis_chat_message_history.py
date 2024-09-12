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


import os
import uuid

import pytest
import redis
from langchain_core.messages import AIMessage, HumanMessage

from langchain_google_memorystore_redis import MemorystoreChatMessageHistory


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


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
def test_redis_multiple_sessions(client) -> None:
    session_id1 = uuid.uuid4().hex
    history1 = MemorystoreChatMessageHistory(
        client=client,
        session_id=session_id1,
    )
    session_id2 = uuid.uuid4().hex
    history2 = MemorystoreChatMessageHistory(
        client=client,
        session_id=session_id2,
    )

    history1.add_ai_message("Hey! I am AI!")
    history1.add_user_message("Hey! I am human!")
    history2.add_user_message("Hey! I am human in another session!")
    messages1 = history1.messages
    messages2 = history2.messages

    assert len(messages1) == 2
    assert len(messages2) == 1
    assert isinstance(messages1[0], AIMessage)
    assert messages1[0].content == "Hey! I am AI!"
    assert isinstance(messages1[1], HumanMessage)
    assert messages1[1].content == "Hey! I am human!"
    assert isinstance(messages2[0], HumanMessage)
    assert messages2[0].content == "Hey! I am human in another session!"

    history1.clear()
    assert len(history1.messages) == 0
    assert len(history2.messages) == 1

    history2.clear()
    assert len(history1.messages) == 0
    assert len(history2.messages) == 0
