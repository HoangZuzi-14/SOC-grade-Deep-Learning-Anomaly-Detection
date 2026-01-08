"""
CLI tool to start streaming inference
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from streaming.log_streamer import LogStreamer, LogProcessor
from streaming.streaming_inference import StreamingInference
from api.main import score_sequence, MODEL, MODEL_CONFIG
from api.database import SessionLocal, create_alert_from_score


async def main():
    parser = argparse.ArgumentParser(description="Start real-time log streaming")
    parser.add_argument("--log_file", required=True, help="Path to log file")
    parser.add_argument("--window_size", type=int, default=5, help="Window size")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between reads (seconds)")
    parser.add_argument("--output", choices=["console", "json"], default="console", help="Output format")
    
    args = parser.parse_args()
    
    if MODEL is None:
        print("Error: Model not loaded. Please ensure model is available.")
        return
    
    # Create streamer
    streamer = LogStreamer(args.log_file, delay=args.delay)
    processor = LogProcessor()
    processor.set_window_size(args.window_size)
    
    # Create database session
    db = SessionLocal()
    
    # Create inference
    def score_func(sequence):
        return score_sequence(sequence, "lstm", db, store_alert=False)
    
    inference = StreamingInference(MODEL, score_func)
    
    print(f"Starting streaming from {args.log_file}...")
    print(f"Window size: {args.window_size}")
    print("Press Ctrl+C to stop\n")
    
    try:
        async for seq_data in processor.process_stream(streamer.stream()):
            async for result in inference.process_sequences([seq_data]):
                if "error" in result:
                    print(f"Error: {result['error']}")
                    continue
                
                if args.output == "json":
                    print(json.dumps(result))
                else:
                    score = result.get("score", 0)
                    severity = result.get("severity", "NONE")
                    alert = result.get("alert", False)
                    
                    alert_marker = "🚨" if alert else "  "
                    print(f"{alert_marker} Score: {score:.4f} | Severity: {severity} | Alert: {alert}")
                    
                    if alert:
                        # Create alert in database
                        try:
                            alert_obj = create_alert_from_score(
                                db=db,
                                sequence=result["sequence"],
                                score=result["score"],
                                severity=result["severity"]
                            )
                            print(f"   → Alert created: ID={alert_obj.id}")
                        except Exception as e:
                            print(f"   → Error creating alert: {e}")
                
                # Print stats periodically
                if inference.stats["processed"] % 100 == 0:
                    stats = inference.get_stats()
                    print(f"\n📊 Stats: Processed={stats['processed']} | "
                          f"Alerts={stats['alerts']} | "
                          f"Errors={stats['errors']}\n")
    
    except KeyboardInterrupt:
        print("\n\nStopping streaming...")
        streamer.stop()
        
        # Print final stats
        stats = inference.get_stats()
        print(f"\nFinal Stats:")
        print(f"  Processed: {stats['processed']}")
        print(f"  Alerts: {stats['alerts']}")
        print(f"  Errors: {stats['errors']}")
        if stats.get('throughput'):
            print(f"  Throughput: {stats['throughput']:.2f} seq/s")
    
    finally:
        db.close()


if __name__ == "__main__":
    import json
    asyncio.run(main())
