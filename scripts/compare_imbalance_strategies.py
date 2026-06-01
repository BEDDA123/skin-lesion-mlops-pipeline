#!/usr/bin/env python
"""
Script de comparaison: SMOTE vs Class_Weight vs Augmentation

Teste trois stratégies de gestion du déséquilibre et compare les performances.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml

def run_experiment(config_name: str, imbalance_method: str, experiment_id: str) -> Dict[str, Any]:
    """
    Lance une expérience avec une configuration donnée.
    """
    print(f"\n{'='*70}")
    print(f"EXPERIENCE: {config_name}")
    print(f"Imbalance Method: {imbalance_method}")
    print(f"{'='*70}")

    # Charger params.yaml
    params_file = Path('params.yaml')
    with open(params_file) as f:
        params = yaml.safe_load(f)

    # Modifier la stratégie
    params['preprocess']['imbalance_method'] = imbalance_method

    # Sauvegarder temporairement
    with open(params_file, 'w') as f:
        yaml.dump(params, f)

    print("\n[1/3] Forcage du recalcul des données...")
    result = subprocess.run(
        [sys.executable, 'src/train.py', '--prepare-only', '--force-prepare'],
        cwd=Path.cwd(),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"ERREUR: {result.stderr}")
        return None

    print("[2/3] Entraînement du modèle...")
    result = subprocess.run(
        [sys.executable, 'src/train.py', '--train-only'],
        cwd=Path.cwd(),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"ERREUR: {result.stderr}")
        return None

    print("[3/3] Récupération des métriques...")

    # Chercher le derniers mlruns
    mlruns_dir = Path('mlruns/406321544010892008')
    runs = sorted(mlruns_dir.glob('*/'), key=lambda p: p.stat().st_mtime, reverse=True)

    if not runs:
        print("ERREUR: Aucun run MLflow trouvé")
        return None

    latest_run = runs[0]
    metrics_file = latest_run / 'metrics.json'

    if not metrics_file.exists():
        print(f"ERREUR: metrics.json non trouvé dans {latest_run}")
        return None

    # Charger metrics
    with open(metrics_file) as f:
        metrics = json.load(f)

    # Charger les predictions pour classification report
    pred_file = Path('models/predictions.json')
    if pred_file.exists():
        with open(pred_file) as f:
            predictions = json.load(f)
    else:
        predictions = None

    return {
        'experiment_id': experiment_id,
        'config_name': config_name,
        'imbalance_method': imbalance_method,
        'metrics': metrics,
        'predictions': predictions,
        'run_dir': str(latest_run)
    }


def compare_results(results: list) -> None:
    """
    Compare les résultats de plusieurs expériences.
    """
    print(f"\n\n{'='*70}")
    print("COMPARAISON DES RESULTATS")
    print(f"{'='*70}\n")

    # Créer un dataframe pour chaque métrique
    df_data = []

    for r in results:
        if r is None:
            continue

        metrics = r['metrics']
        row = {
            'Config': r['config_name'],
            'Imbalance': r['imbalance_method'],
        }

        # Métriques clés
        for key in ['accuracy', 'f1_macro', 'f1_weighted', 'balanced_accuracy']:
            if key in metrics:
                row[key.upper()] = f"{metrics[key]:.4f}"

        # Métriques par classe
        for key in ['precision_class_0', 'precision_class_1', 'precision_class_2']:
            if key in metrics:
                row[f"PREC_{key[-1]}"] = f"{metrics[key]:.4f}"

        for key in ['recall_class_0', 'recall_class_1', 'recall_class_2']:
            if key in metrics:
                row[f"REC_{key[-1]}"] = f"{metrics[key]:.4f}"

        for key in ['f1_class_0', 'f1_class_1', 'f1_class_2']:
            if key in metrics:
                row[f"F1_{key[-1]}"] = f"{metrics[key]:.4f}"

        df_data.append(row)

    if not df_data:
        print("Aucun résultat à comparer")
        return

    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))

    # Sauvegarder la comparaison
    df.to_csv('COMPARISON_RESULTS.csv', index=False)
    print("\nResultats sauvegardés dans COMPARISON_RESULTS.csv")


def main():
    """
    Lance les expériences de comparaison.
    """
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║           COMPARAISON: SMOTE vs Class_Weight vs Aucun              ║
    ║                                                                    ║
    ║  Cette script teste l'impact des différentes stratégies           ║
    ║  de gestion du déséquilibre sur les performances du modèle        ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Définir les expériences
    experiments = [
        ("SMOTE+Tomek (Current)", "smote_tomek", "exp_smote"),
        ("Class Weight Only", "none", "exp_classweight"),
        # ("Augmentation", "augmentation", "exp_augmentation"),  # À implémenter
    ]

    results = []
    for config_name, imbalance_method, exp_id in experiments:
        result = run_experiment(config_name, imbalance_method, exp_id)
        results.append(result)

    # Comparer les résultats
    compare_results(results)

    print("\n" + "="*70)
    print("RECOMMANDATION")
    print("="*70)

    if results[0] and results[1]:
        smote_f1 = float(results[0]['metrics'].get('f1_macro', 0))
        cw_f1 = float(results[1]['metrics'].get('f1_macro', 0))

        print(f"\nSMOTE+Tomek - F1 macro: {smote_f1:.4f}")
        print(f"Class Weight - F1 macro: {cw_f1:.4f}")

        if cw_f1 > smote_f1:
            print(f"\n✅ Class Weight est MEILLEUR de {(cw_f1-smote_f1)*100:.2f}%")
            print("   Recommandation: Désactiver SMOTE et utiliser class_weight uniquement")
        elif smote_f1 > cw_f1:
            print(f"\n❌ SMOTE est MEILLEUR de {(smote_f1-cw_f1)*100:.2f}%")
            print("   Recommandation: Garder SMOTE (mais analyser why)")
        else:
            print("\n⚖️  Performances similaires")


if __name__ == '__main__':
    main()
