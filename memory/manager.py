import redis.asyncio as redis
import json
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
from schema.models import MemoryEntry, DocumentFormat, DocumentIntent

class MemoryManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
    
    async def connect(self):
        """Initialize Redis connection"""
        self.redis_client = redis.from_url(
            self.redis_url, 
            encoding="utf-8", 
            decode_responses=True
        )
        
    async def disconnect(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def store_entry(
        self, 
        source: str,
        document_type: DocumentFormat,
        intent: DocumentIntent,
        extracted_values: Dict[str, Any],
        thread_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> str:
        """Store a memory entry and return its ID"""
        if not self.redis_client:
            await self.connect()
            
        entry_id = str(uuid.uuid4())
        memory_entry = MemoryEntry(
            id=entry_id,
            source=source,
            document_type=document_type,
            intent=intent,
            extracted_values=extracted_values,
            thread_id=thread_id,
            conversation_id=conversation_id
        )
        entry_data = memory_entry.model_dump()
        entry_data['timestamp'] = entry_data['timestamp'].isoformat()
        entry_data['extracted_values'] = json.dumps(entry_data['extracted_values'])

        for key, value in entry_data.items():
            if value is None:
                entry_data[key] = ""
        
        await self.redis_client.hset(
            f"memory:{entry_id}",
            mapping=entry_data
        )

        await self.redis_client.sadd(f"by_type:{document_type}", entry_id)
        await self.redis_client.sadd(f"by_intent:{intent}", entry_id)
        if thread_id:
            await self.redis_client.sadd(f"by_thread:{thread_id}", entry_id)
        if conversation_id:
            await self.redis_client.sadd(f"by_conversation:{conversation_id}", entry_id)

        await self.redis_client.expire(f"memory:{entry_id}", 30 * 24 * 3600)
        
        return entry_id
    
    async def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID"""
        if not self.redis_client:
            await self.connect()
            
        data = await self.redis_client.hgetall(f"memory:{entry_id}")
        if not data:
            return None

        if 'timestamp' in data:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'extracted_values' in data:
            data['extracted_values'] = json.loads(data['extracted_values'])

        for key, value in data.items():
            if value == "" and key in ['thread_id', 'conversation_id']:
                data[key] = None
        
        return MemoryEntry.model_validate(data)
    
    async def get_entries_by_thread(self, thread_id: str) -> List[MemoryEntry]:
        """Get all entries for a specific thread"""
        if not self.redis_client:
            await self.connect()
            
        entry_ids = await self.redis_client.smembers(f"by_thread:{thread_id}")
        entries = []
        
        for entry_id in entry_ids:
            entry = await self.get_entry(entry_id)
            if entry:
                entries.append(entry)
                
        return sorted(entries, key=lambda x: x.timestamp)
    
    async def get_entries_by_type(self, document_type: DocumentFormat) -> List[MemoryEntry]:
        """Get all entries of a specific document type"""
        if not self.redis_client:
            await self.connect()
            
        entry_ids = await self.redis_client.smembers(f"by_type:{document_type}")
        entries = []
        
        for entry_id in entry_ids:
            entry = await self.get_entry(entry_id)
            if entry:
                entries.append(entry)
                
        return sorted(entries, key=lambda x: x.timestamp, reverse=True)
    
    async def get_recent_context(self, limit: int = 10) -> List[MemoryEntry]:
        """Get recent entries for context"""
        if not self.redis_client:
            await self.connect()

        keys = await self.redis_client.keys("memory:*")
        entries = []
        
        for key in keys:
            entry_id = key.split(":")[1]
            entry = await self.get_entry(entry_id)
            if entry:
                entries.append(entry)

        entries.sort(key=lambda x: x.timestamp, reverse=True)
        return entries[:limit]
