from memobase import MemoBaseClient, ChatBlob

PROJECT_URL = "http://localhost:8019"
PROJECT_TOKEN = "secret"

client = MemoBaseClient(
    project_url=PROJECT_URL,
    api_key=PROJECT_TOKEN,
)

assert client.ping(), "Your Memobase server is not running"

client.update_config(
    """
overwrite_user_profiles:
  - topic: "psychological"
    sub_topics:
      - name: "mood"
  - topic: "interest"
    sub_topics:
      - name: "travel"
"""
)
uid = client.add_user()
u = client.get_user(uid)


def pack_blob(message):
    print("User said:", message)
    print("-----------------------")
    return ChatBlob(messages=[{"role": "user", "content": message}])


u.insert(pack_blob("I love traveling to China"))

u.insert(pack_blob("I'm feeling really stressed today"))
u.flush(sync=True)
print(
    "MEMORY：\n",
    "\n".join([f"- {p.describe}" for p in u.profile()]),
    "\n-----------------------",
)

u.insert(pack_blob("I'm happy today!"))
u.flush(sync=True)
print(
    "MEMORY：\n",
    "\n".join([f"- {p.describe}" for p in u.profile()]),
    "\n-----------------------",
)
id = [p.id for p in u.profile() if p.sub_topic == "mood"][0]
u.delete_profile(id)
print("DELETE： mood")
print("-----------------------")
print(
    "MEMORY：\n",
    "\n".join([f"- {p.describe}" for p in u.profile()]),
    "\n-----------------------",
)


print(
    "Events:",
    "\n".join(
        [
            f"📅{e.created_at}\n{e.event_data.event_tip}\n{e.event_data.profile_delta}"
            for e in u.search_event("stressed", topk=1, similarity_threshold=0.2)
        ]
    ),
)

client.update_config(None)
