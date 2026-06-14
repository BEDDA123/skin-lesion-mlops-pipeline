# Justification du regroupement des classes HAM10000 en 3 catégories

## Contexte
Le dataset HAM10000 (HAM = Human Against Machine) contient initialement 7 classes de lésions cutanées. Afin de simplifier le problème de classification et de réduire le déséquilibre entre les classes, un regroupement en 3 catégories cliniques a été effectué.

## Regroupement retenu

### Classe 0: Melanocytic Nevi (nv)
- **Classe originale:** nv (4)
- **Caractérisation:** Lésions bénignes homogènes
- **Justification:**
  - Constitue la catégorie la plus représentée du dataset
  - Caractéristiques visuelles relativement homogènes
  - Permet une catégorie de référence solide

### Classe 1: Lésions bénignes (Benign)
- **Classes originales regroupées:**
  - Benign Keratosis-like Lesions (bkl) - classe 2
  - Dermatofibroma (df) - classe 3
  - Vascular Lesions (vasc) - classe 5
- **Justification:**
  - Partagent plusieurs propriétés morphologiques similaires
  - Fusion justifiée cliniquement (lésions bénignes)
  - Augmente le nombre d'échantillons disponibles
  - Améliore la stabilité de l'apprentissage

### Classe 2: Lésions malignes/précancéreuses (Malignant)
- **Classes originales regroupées:**
  - Melanoma (mel) - classe 6
  - Basal Cell Carcinoma (bcc) - classe 1
  - Actinic Keratoses (akiec) - classe 0
- **Justification:**
  - Regroupent des lésions malignes ou précancéreuses
  - Intérêt clinique majeur dans le cadre du diagnostic assisté par IA
  - Permet une meilleure discrimination entre bénin et malin
  - Harmonise les considérations médicales et statistiques

## Tableau de remappage

| Classe originale | Label | Nouvelle classe | Catégorie |
|---|---|---|---|
| akiec | 0 | 2 | Malin |
| bcc | 1 | 2 | Malin |
| bkl | 2 | 1 | Bénin |
| df | 3 | 1 | Bénin |
| nv | 4 | 0 | Nevi |
| vasc | 5 | 1 | Bénin |
| mel | 6 | 2 | Malin |

## Avantages de cette stratégie

1. **Réduction de la complexité:** Passage de 7 classes à 3 facilite l'apprentissage du modèle
2. **Équilibre amélioré:** Les classes bénignes sont regroupées, augmentant leur représentation
3. **Signification médicale:** Distinction claire entre bénin/malin, pertinente cliniquement
4. **Stabilité accrue:** Plus d'échantillons par catégorie améliore la stabilité des modèles
5. **Pertinence clinique:** La séparation bénin/malin est l'enjeu majeur du diagnostic

## Implication pour le modèle

Cette stratégie permet:
- Une classification plus robuste avec moins de risque de surapprentissage
- Une meilleure généralisation sur de nouvelles données
- Une interprétabilité améliorée (3 catégories vs 7 spécialisées)
- Un équilibre meilleur entre les classes d'apprentissage

## Référence d'implémentation

Le remappage est implémenté dans `data/raw/dat.py` dans la fonction `filter_and_remap_labels()`.
Le dataset traité est sauvegardé dans `data/raw/hmnist_3classes.csv`.

---
*Document préparé pour le projet de classification HAM10000 avec apprentissage machine*
