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
    """Regroupe les 7 classes HAM10000 originales en 3 catégories cliniques.
    
    Justification du regroupement:
    - Classe 0: Melanocytic Nevi (nv) - lésions bénignes homogènes
    - Classe 1: Benign Keratosis-like (bkl), Dermatofibroma (df), Vascular (vasc) - lésions bénignes
    - Classe 2: Melanoma (mel), Basal Cell Carcinoma (bcc), Actinic Keratoses (akiec) - lésions malignes/précancéreuses
    
    Mapping des classes originales (0-6) vers les 3 nouvelles catégories:
    - akiec (0) -> 2 (malin)
    - bcc (1) -> 2 (malin)
    - bkl (2) -> 1 (bénin)
    - df (3) -> 1 (bénin)
    - nv (4) -> 0 (nevi)
    - vasc (5) -> 1 (bénin)
    - mel (6) -> 2 (malin)
    """
    # Conversion en int pour éviter les problèmes de labels stockés comme chaîne
    df = df.copy()
    df["label"] = df["label"].astype(int)

    # Remappage complet des 7 classes originales vers 3 catégories cliniques
    mapping = {
        0: 2,  # akiec (Actinic Keratoses) -> malin
        1: 2,  # bcc (Basal Cell Carcinoma) -> malin
        2: 1,  # bkl (Benign Keratosis-like) -> bénin
        3: 1,  # df (Dermatofibroma) -> bénin
        4: 0,  # nv (Melanocytic Nevi) -> nevi
        5: 1,  # vasc (Vascular Lesions) -> bénin
        6: 2,  # mel (Melanoma) -> malin
    }
    
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
        "Rapport de transformation HAM10000 en 3 catégories cliniques",
        "-----------------------------------------------------------",
        "Justification du regroupement des 7 classes HAM10000 en 3 catégories:",
        "",
        "Classe 0 (Melanocytic Nevi - nv):",
        "  - Lésions bénignes homogènes",
        "  - Classe la plus représentée du dataset",
        "  - Caractéristiques visuelles relativement homogènes",
        "",
        "Classe 1 (Lésions bénignes):",
        "  - Benign Keratosis-like Lesions (bkl)",
        "  - Dermatofibroma (df)",
        "  - Vascular Lesions (vasc)",
        "  - Propriétés morphologiques similaires",
        "",
        "Classe 2 (Lésions malignes/précancéreuses):",
        "  - Melanoma (mel)",
        "  - Basal Cell Carcinoma (bcc)",
        "  - Actinic Keratoses (akiec)",
        "  - Intérêt clinique majeur pour le diagnostic IA",
        "",
        "Distribution avant prétraitement:",
    ]
    lines += [f"  Classe {idx}: {count}" for idx, count in before_counts.items()]
    lines += ["", "Distribution après prétraitement:",]
    lines += [f"  Classe {idx}: {count}" for idx, count in after_counts.items()]
    lines += [
        "",
        f"Nombre de classes avant prétraitement: {len(before_counts)}",
        f"Nombre de classes après prétraitement: {len(after_counts)}",
        "",
        "Remappage appliqué:",
        "  0 (akiec) -> 2 (malin)",
        "  1 (bcc) -> 2 (malin)",
        "  2 (bkl) -> 1 (bénin)",
        "  3 (df) -> 1 (bénin)",
        "  4 (nv) -> 0 (nevi)",
        "  5 (vasc) -> 1 (bénin)",
        "  6 (mel) -> 2 (malin)",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapport textuel enregistré: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """
            Prétraite le dataset HAM10000 pour réduire le problème de classification à 3 catégories cliniques.

            Regroupement justifié par des considérations cliniques et statistiques:
            - Classe 0: Melanocytic Nevi (nv)
            - Classe 1: Benign Keratosis-like (bkl), Dermatofibroma (df), Vascular Lesions (vasc)
            - Classe 2: Melanoma (mel), Basal Cell Carcinoma (bcc), Actinic Keratoses (akiec)
            
            Remappage complet:
            - akiec (0) -> 2 (malin)
            - bcc (1) -> 2 (malin)
            - bkl (2) -> 1 (bénin)
            - df (3) -> 1 (bénin)
            - nv (4) -> 0 (nevi)
            - vasc (5) -> 1 (bénin)
            - mel (6) -> 2 (malin)
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
