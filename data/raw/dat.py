from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Charge le CSV HAM10000 et vérifie la présence de la colonne label."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Fichier introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("Le CSV doit contenir une colonne 'label'.")

    return df


def compute_distribution(df: pd.DataFrame) -> pd.Series:
    """Calcule la distribution des labels triée par valeur de label."""
    return df["label"].value_counts().sort_index()


def filter_and_remap_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime certaines classes, fusionne d'autres classes et remappe les labels."""
    # Conversion en int pour éviter les problèmes de labels stockés comme chaîne
    df = df.copy()
    df["label"] = df["label"].astype(int)

    # 1. Supprimer les classes non désirées: akiec(0), df(3), vasc(5)
    classes_supprimees = {0, 3, 5}
    df = df[~df["label"].isin(classes_supprimees)]

    # 2. Fusionner mel(6) et bcc(1) dans la nouvelle classe 2
    mapping = {4: 0, 2: 1, 1: 2, 6: 2}
    df["label"] = df["label"].map(mapping)

    if df["label"].isna().any():
        raise ValueError("Certaines valeurs de label ne sont pas couvertes par le remappage.")

    df["label"] = df["label"].astype(int)

    # Vérifier qu'il reste exactement 3 classes
    classes_apres = sorted(df["label"].unique())
    if classes_apres != [0, 1, 2]:
        raise AssertionError(
            f"Le prétraitement doit produire exactement les classes [0, 1, 2], mais a trouvé {classes_apres}."
        )

    return df


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Sauvegarde le dataset traité au format CSV."""
    df.to_csv(output_path, index=False)
    print(f"Dataset sauvegardé sous: {output_path}")


def plot_distribution(counts: pd.Series, title: str, save_path: Path) -> None:
    """Trace et sauvegarde un histogramme de distribution des classes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.values, color=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Label prétraité", fontsize=12)
    ax.set_ylabel("Nombre d'échantillons", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for x, y in zip(counts.index, counts.values):
        ax.text(x, y + max(counts.values) * 0.01, str(int(y)), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Graphique enregistré: {save_path}")


def plot_comparison(before_counts: pd.Series, after_counts: pd.Series, save_path: Path) -> None:
    """Trace une figure de comparaison avant/après pour chaque label."""
    summary = pd.DataFrame({"Avant prétraitement": before_counts, "Après prétraitement": after_counts}).fillna(0)
    summary.index = summary.index.astype(str)

    fig, ax = plt.subplots(figsize=(9, 6))
    width = 0.35
    x = np.arange(len(summary.index))
    ax.bar(x - width / 2, summary["Avant prétraitement"].values, width=width, label="Avant", color="#4c72b0")
    ax.bar(x + width / 2, summary["Après prétraitement"].values, width=width, label="Après", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index)
    ax.set_title("Comparaison des distributions des classes avant / après prétraitement", fontsize=14)
    ax.set_xlabel("Label original ou remappé", fontsize=12)
    ax.set_ylabel("Nombre d'échantillons", fontsize=12)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Graphique comparatif enregistré: {save_path}")


def plot_correlation_matrix(df: pd.DataFrame, save_path: Path, max_features: int = 50, sample_size: int = 2000) -> None:
    """Calcule et trace une matrice de corrélation sur un échantillon de pixels."""
    pixel_cols = [col for col in df.columns if col != "label"]
    pixel_cols = sorted(pixel_cols)
    if len(pixel_cols) == 0:
        raise ValueError("Aucune colonne de pixel trouvée pour la corrélation.")

    # Limiter le nombre de colonnes pour garder la figure lisible
    selected_cols = pixel_cols[:min(max_features, len(pixel_cols))]
    sample = df[selected_cols].sample(n=min(sample_size, len(df)), random_state=42)
    corr = sample.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(selected_cols)))
    ax.set_yticks(range(len(selected_cols)))
    ax.set_xticklabels(selected_cols, rotation=90, fontsize=6)
    ax.set_yticklabels(selected_cols, fontsize=6)
    ax.set_title("Matrice de corrélation des pixels (échantillon)", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Matrice de corrélation enregistrée: {save_path}")


def save_report(before_counts: pd.Series, after_counts: pd.Series, output_path: Path) -> None:
    """Sauvegarde un rapport textuel des distributions avant et après."""
    lines = [
        "Rapport de transformation HAM10000 en 3 classes",
        "---------------------------------------------",
        "Distribution avant prétraitement:",
    ]
    lines += [f"  Classe {idx}: {count}" for idx, count in before_counts.items()]
    lines += ["", "Distribution après prétraitement:",]
    lines += [f"  Classe {idx}: {count}" for idx, count in after_counts.items()]
    lines += [
        "",
        f"Nombre de classes avant prétraitement: {len(before_counts)}",
        f"Nombre de classes après prétraitement: {len(after_counts)}",
        "Remappage appliqué: 4->0, 2->1, 1->2, 6->2",
        "Classes supprimées: 0, 3, 5",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapport textuel enregistré: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """
            Prétraite le dataset HAM10000 pour réduire le problème de classification à 3 classes.

            - Supprime les labels 0, 3, 5
            - Fusionne les labels 6 et 1 dans la classe 2
            - Remappe 4 -> 0, 2 -> 1, {1, 6} -> 2
            - Sauvegarde le nouveau CSV et crée des visualisations
            """
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "hmnist_28_28_RGB.csv",
        help="Chemin vers le CSV HAM10000 d'origine",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "hmnist_3classes.csv",
        help="Chemin de sortie du CSV résultant",
    )
    parser.add_argument(
        "--visualisations",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "visualisations",
        help="Dossier de sortie pour les graphiques et le rapport",
    )
    args = parser.parse_args()

    print("Chargement du dataset...")
    df_original = load_csv(args.input)
    before_counts = compute_distribution(df_original)
    print("Distribution avant prétraitement:")
    print(before_counts.to_string())

    print("Application du prétraitement des labels...")
    df_processed = filter_and_remap_labels(df_original)
    after_counts = compute_distribution(df_processed)
    print("Distribution après prétraitement:")
    print(after_counts.to_string())

    if len(after_counts) != 3:
        raise AssertionError(
            f"Le dataset traité doit contenir exactement 3 classes, obtenu {len(after_counts)}."
        )

    save_csv(df_processed, args.output)
    args.visualisations.mkdir(parents=True, exist_ok=True)

    plot_distribution(
        before_counts,
        title="Distribution des classes avant prétraitement",
        save_path=args.visualisations / "distribution_avant_pretraitement.png",
    )
    plot_distribution(
        after_counts,
        title="Distribution des classes après prétraitement",
        save_path=args.visualisations / "distribution_apres_pretraitement.png",
    )
    plot_comparison(
        before_counts,
        after_counts,
        save_path=args.visualisations / "comparaison_distribution_avant_apres.png",
    )
    plot_correlation_matrix(
        df_original,
        save_path=args.visualisations / "matrice_correlation_pixels.png",
        max_features=50,
        sample_size=2000,
    )
    save_report(
        before_counts,
        after_counts,
        output_path=args.visualisations / "rapport_distribution.txt",
    )

    


if __name__ == "__main__":
    main()
