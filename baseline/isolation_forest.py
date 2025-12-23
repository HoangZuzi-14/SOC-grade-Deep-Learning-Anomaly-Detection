import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def load_sequences(pkl_path):
    """
    Load sequences from pickle file.
    Each sequence is a fixed-length list of template IDs.
    """
    with open(pkl_path, "rb") as f:
        sequences = pickle.load(f)
    return np.array(sequences)


def train_isolation_forest(X, contamination=0.01):
    """
    Train Isolation Forest on sequence-level data.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    return model, scaler


def compute_anomaly_score(model, scaler, X):
    """
    Compute anomaly scores for sequences.
    Higher score = more anomalous.
    """
    X_scaled = scaler.transform(X)

    # decision_function: higher = more normal
    scores = -model.decision_function(X_scaled)
    return scores


def main():
    # Path to sequence data
    sequence_path = "data/sequences/sequences.pkl"

    print("[+] Loading sequences...")
    X = load_sequences(sequence_path)
    print(f"[+] Loaded {len(X)} sequences with shape {X.shape}")

    print("[+] Training Isolation Forest baseline...")
    model, scaler = train_isolation_forest(X)

    print("[+] Computing anomaly scores...")
    scores = compute_anomaly_score(model, scaler, X)

    print("[+] Example anomaly scores:")
    print(scores[:10])

    # Optional: save scores for later comparison
    # with open("baseline_if_scores.pkl", "wb") as f:
    #     pickle.dump(scores, f)

    print("[+] Baseline Isolation Forest completed.")


if __name__ == "__main__":
    main()
