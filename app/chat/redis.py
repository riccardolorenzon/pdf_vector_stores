import os
import redis

client = redis.Redis.from_url(
    os.environ.get("REDIS_URI", "redis://localhost:6379"), 
    decode_responses=True)
    