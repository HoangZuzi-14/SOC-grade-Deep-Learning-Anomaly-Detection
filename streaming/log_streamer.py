"""
Real-time log streaming and processing
"""
import asyncio
import json
from pathlib import Path
from typing import AsyncIterator, Optional, Callable
from datetime import datetime
import aiofiles


class LogStreamer:
    """
    Stream logs from file in real-time (tail -f style)
    """
    
    def __init__(self, log_file: str, delay: float = 0.1):
        """
        Args:
            log_file: Path to log file
            delay: Delay between reads (seconds)
        """
        self.log_file = Path(log_file)
        self.delay = delay
        self.position = 0
        self.running = False
    
    async def stream(self) -> AsyncIterator[str]:
        """
        Stream log lines as they are written
        """
        self.running = True
        
        # If file doesn't exist, wait for it
        while not self.log_file.exists() and self.running:
            await asyncio.sleep(1)
        
        if not self.running:
            return
        
        # Open file and seek to end (tail behavior)
        async with aiofiles.open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
            # Seek to end for new logs only
            await f.seek(0, 2)  # Seek to end
            
            while self.running:
                line = await f.readline()
                
                if line:
                    yield line.strip()
                else:
                    # No new data, wait a bit
                    await asyncio.sleep(self.delay)
    
    def stop(self):
        """Stop streaming"""
        self.running = False


class LogProcessor:
    """
    Process log streams and extract sequences
    """
    
    def __init__(self, parser_func: Optional[Callable] = None):
        """
        Args:
            parser_func: Function to parse log line to template_id
        """
        self.parser_func = parser_func
        self.sequence_buffer = []
        self.window_size = 20
    
    def set_window_size(self, window_size: int):
        """Set sliding window size"""
        self.window_size = window_size
    
    def parse_log_line(self, line: str) -> Optional[int]:
        """
        Parse log line to template ID
        Override this or provide parser_func
        """
        if self.parser_func:
            return self.parser_func(line)
        # Default: extract first number as template_id
        try:
            parts = line.split()
            if parts:
                return int(parts[0])
        except:
            pass
        return None
    
    async def process_stream(
        self,
        log_stream: AsyncIterator[str]
    ) -> AsyncIterator[dict]:
        """
        Process log stream and yield sequences
        """
        async for line in log_stream:
            template_id = self.parse_log_line(line)
            
            if template_id is not None:
                self.sequence_buffer.append(template_id)
                
                # Keep buffer size
                if len(self.sequence_buffer) > self.window_size * 2:
                    self.sequence_buffer = self.sequence_buffer[-self.window_size:]
                
                # Yield sequence when we have enough
                if len(self.sequence_buffer) >= self.window_size:
                    sequence = self.sequence_buffer[-self.window_size:]
                    yield {
                        "sequence": sequence,
                        "timestamp": datetime.utcnow().isoformat(),
                        "raw_log": line
                    }


class StreamingInference:
    """
    Real-time inference on streaming sequences
    """
    
    def __init__(self, model, scoring_func: Callable):
        """
        Args:
            model: Trained model
            scoring_func: Function to score sequence
        """
        self.model = model
        self.scoring_func = scoring_func
        self.stats = {
            "processed": 0,
            "alerts": 0,
            "errors": 0
        }
    
    async def process_sequences(
        self,
        sequence_stream: AsyncIterator[dict]
    ) -> AsyncIterator[dict]:
        """
        Process sequences and yield scores
        """
        async for seq_data in sequence_stream:
            try:
                sequence = seq_data["sequence"]
                
                # Score sequence
                result = self.scoring_func(sequence)
                
                # Add metadata
                result.update({
                    "timestamp": seq_data.get("timestamp"),
                    "raw_log": seq_data.get("raw_log"),
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
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        return self.stats.copy()
