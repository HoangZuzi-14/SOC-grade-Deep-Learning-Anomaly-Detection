import json
import pickle
import os
import argparse

def load_data(input_file):
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} log entries.")
    return data

def build_sequences(data, window_size=20, step_size=1):
    print(f"Building sequences with window_size={window_size}, step_size={step_size}...")
    
    # Extract template_ids
    # Assumes data is a list of dicts with 'template_id'
    raw_event_ids = [entry['template_id'] for entry in data]
    
    # Create mapping just in case IDs are not 0-indexed contiguous integers
    unique_events = sorted(list(set(raw_event_ids)))
    event2idx = {event: idx for idx, event in enumerate(unique_events)}
    idx2event = {idx: event for idx, event in enumerate(unique_events)}
    
    print(f"Found {len(unique_events)} unique templates.")
    
    # Transform to indices
    indexed_events = [event2idx[eid] for eid in raw_event_ids]
    
    sequences = []
    # Sliding window
    # Sequence: [e_t, e_{t+1}, ..., e_{t+w-1}] -> label could be e_{t+w}
    # Current task output requirements: List[List[int]]
    
    for i in range(0, len(indexed_events) - window_size, step_size):
        seq = indexed_events[i : i + window_size]
        # Optionally, we might want the next event as label, 
        # but usually 'sequences' list contains the input features.
        # DeepLog/DeepLoglizer usually expects X and Y or just raw sequences.
        # Based on requirement "List[List[int]]", we will store the sequence window.
        sequences.append(seq)
        
    print(f"Generated {len(sequences)} sequences.")
    return sequences, event2idx, idx2event

def main():
    parser = argparse.ArgumentParser(description='Build sequences for Deep Learning Anomaly Detection')
    parser.add_argument('--input_file', type=str, default='../data/parsed/parsed_data.json',
                        help='Path to parsed log file (json)')
    parser.add_argument('--output_dir', type=str, default='../data/sequences/',
                        help='Directory to save sequences')
    parser.add_argument('--window_size', type=int, default=20,
                        help='Window size for sliding window')
    
    args = parser.parse_args()
    
    # Resolve paths relative to this script if needed, currently assumes running from project root or sequence folder
    # Adjusting to be robust
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_path = args.input_file
    if not os.path.isabs(input_path):
        input_path = os.path.normpath(os.path.join(script_dir, input_path))
        
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(script_dir, output_dir))
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Process
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    data = load_data(input_path)
    sequences, event2idx, idx2event = build_sequences(data, window_size=args.window_size)
    
    # Save sequences
    seq_file = os.path.join(output_dir, 'sequences.pkl')
    with open(seq_file, 'wb') as f:
        pickle.dump(sequences, f)
    print(f"Saved sequences to {seq_file}")
    
    # Save mappings (crucial for inference/decoding)
    map_file = os.path.join(output_dir, 'event_mapping.json')
    mapping = {
        'event2idx': event2idx,
        'idx2event': idx2event
    }
    with open(map_file, 'w') as f:
        json.dump(mapping, f, indent=4)
    print(f"Saved event mapping to {map_file}")
    
    # Statistics
    print("\n--- Statistics ---")
    print(f"Total Sequences: {len(sequences)}")
    print(f"Sequence Length: {args.window_size}")
    print(f"Total Unique Events: {len(event2idx)}")
    
    # Optional: Save a small sample to check
    sample_file = os.path.join(output_dir, 'sample_sequences.txt')
    with open(sample_file, 'w') as f:
        for s in sequences[:5]:
            f.write(str(s) + '\n')
    print(f"Saved sample sequences to {sample_file}")

if __name__ == '__main__':
    main()

