"""
Streaming inference for real-time processing
"""
import asyncio
from typing import AsyncIterator, Callable, List, Dict, Any
from datetime import datetime


class StreamingInference:
    """
    Real-time inference on streaming sequences
    """
    
    def __init__(self, model, scoring_func: Callable):
        """
        Args:
            model: Trained model (can be None if scoring_func handles it)
            scoring_func: Async function to score sequence
        """
        self.model = model
        self.scoring_func = scoring_func
        self.stats = {
            "processed": 0,
            "alerts": 0,
            "errors": 0,
            "start_time": datetime.utcnow().isoformat()
        }
    
    async def process_sequences(
        self,
        sequence_stream: AsyncIterator[Dict[str, Any]]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process sequences and yield scores
        
        Args:
            sequence_stream: Async iterator of sequence dictionaries
            
        Yields:
            Score results with metadata
        """
        async for seq_data in sequence_stream:
            try:
                sequence = seq_data.get("sequence", [])
                
                if not sequence:
                    continue
                
                # Score sequence (async or sync)
                if asyncio.iscoroutinefunction(self.scoring_func):
                    result = await self.scoring_func(sequence)
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, self.scoring_func, sequence)
                
                # Add metadata
                result.update({
                    "timestamp": seq_data.get("timestamp", datetime.utcnow().isoformat()),
                    "raw_log": seq_data.get("raw_log", ""),
                    "sequence": sequence
                })
                
                self.stats["processed"] += 1
                if result.get("alert"):
                    self.stats["alerts"] += 1
                
                yield result
                
            except Exception as e:
                self.stats["errors"] += 1
                yield {
                    "error": str(e),
                    "sequence": seq_data.get("sequence", []),
                    "timestamp": datetime.utcnow().isoformat()
                }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.stats.copy()
        if stats.get("start_time"):
            start = datetime.fromisoformat(stats["start_time"])
            now = datetime.utcnow()
            stats["duration_seconds"] = (now - start).total_seconds()
            if stats["duration_seconds"] > 0:
                stats["throughput"] = stats["processed"] / stats["duration_seconds"]
        return stats
