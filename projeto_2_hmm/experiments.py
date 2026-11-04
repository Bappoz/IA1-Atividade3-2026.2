"""Experimentos do HMM com dados sintéticos controlados."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from hmm import build_hmm

OUT = Path(__file__).parent / "results"


def run(seed=19):
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    model = build_hmm()
    true_states, observations = model.sample(180, rng)
    decoded, _ = model.viterbi(observations)
    metrics = {
        "accuracy": float(accuracy_score(true_states, decoded)),
        "macro_f1": float(f1_score(true_states, decoded, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(true_states, decoded, labels=range(5)).tolist(),
        "log_likelihood": model.log_likelihood(observations),
    }
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.step(np.arange(len(true_states)), true_states, where="post", label="estado real", lw=2)
    ax.step(np.arange(len(decoded)), decoded, where="post", label="Viterbi", alpha=.8)
    ax.set_yticks(range(5), ["normal", "sobrecarga", "desgaste", "pré-falha", "crítica"])
    ax.set_xlabel("Instante"); ax.set_ylabel("Estado oculto")
    ax.set_title("Estados reais e estados inferidos")
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / "state_comparison.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = np.array(metrics["confusion_matrix"])
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, matrix[i, j], ha="center", va="center")
    ax.set_xlabel("Estimado"); ax.set_ylabel("Real"); ax.set_title("Matriz de confusão")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    fig.colorbar(image, ax=ax); fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=300); plt.close(fig)

    sequences = [model.sample(100, rng)[1] for _ in range(8)]
    learned = build_hmm()
    history = learned.baum_welch(sequences, iterations=40)
    metrics["baum_welch_iterations"] = len(history)
    metrics["baum_welch_initial_log_likelihood"] = float(history[0])
    metrics["baum_welch_final_log_likelihood"] = float(history[-1])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history, marker="o", ms=3)
    ax.set_xlabel("Iteração EM"); ax.set_ylabel("Log-verossimilhança")
    ax.set_title("Convergência do Baum–Welch"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "likelihood_analysis.png", dpi=300); plt.close(fig)
    with (OUT / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
