# Philomathia 

> *Projet B2 IA*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas)
![SymPy](https://img.shields.io/badge/SymPy-1.12-3B5526)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557c)

---

## Contexte du Projet : Pourquoi des maths en Data Science ?

Au début de notre parcours en Data Science, la tentation est grande de sauter directement sur des modèles d'IA complexes comme `Scikit-Learn` ou `PyTorch`. Cependant, utiliser ces outils sans comprendre le moteur sous-jacent revient à conduire une voiture les yeux fermés.

**Philomathia** est notre projet de transition vers le **Machine Learning**. Son but ? Démystifier la "boîte noire" en pratiquant les trois piliers mathématiques fondamentaux avec Python :

1. **L'Algèbre Linéaire :** C'est le langage des données. Vos datasets ne sont rien d'autre que des matrices géantes, et chaque observation est un vecteur.
2. **Les Probabilités & Statistiques :** C'est notre boussole pour analyser le bruit, résumer des milliers d'informations en quelques chiffres clés et mesurer l'incertitude.
3. **Le Calcul de Dérivée :** C'est le moteur d'apprentissage. Pour qu'une IA "apprenne", elle doit calculer sa marge d'erreur et ajuster ses paramètres via le calcul de dérivées (la fameuse *Descente de Gradient*).

---

## Veille Théorique & Glossaire Vulgarisé

Avant de coder dans le notebook, voici notre guide de révision. Pour chaque notion, nous combinons la **définition formelle**, une **image/analogie intuitive** et un **exemple concret du quotidien**.

---

### 1. Vecteur
* **Définition rigoureuse :** Un vecteur est une liste ordonnée de nombres $v = (v_1, v_2, \dots, v_n)$ représentant un point ou une direction dans un espace à $n$ dimensions.
* **Analogie :** La **fiche de profil** d'un individu. Si tu décris une personne par son âge, sa taille et son poids, tu construis un vecteur à 3 dimensions.
* **Exemple concret :** Dans un système de recommandation, le profil d'un utilisateur Spotify s'écrit sous forme de vecteur : `[âge=22, écoutes_pop=85%, écoutes_rock=15%]`.

---

### 2. Matrice
* **Définition rigoureuse :** Une matrice est un tableau bidimensionnel composé de $m$ lignes et $n$ colonnes, noté $M \in \mathbb{R}^{m \times n}$.
* **Analogie :** Le **tableau Excel complet**. Si un vecteur est la fiche d'une seule personne, la matrice est la collection des fiches de tout le monde.
* **Exemple concret :** Un tableau de 1000 appartements (1000 lignes) avec 4 colonnes (`Surface`, `Pièces`, `Année`, `Prix`) forme une matrice $1000 \times 4$.

---

### 3. Probabilité & Loi de probabilité
* **Définition rigoureuse :** La probabilité $P(E) \in [0, 1]$ mesure la chance qu'un événement se réalise. La **loi de probabilité** est la fonction qui décrit la répartition des probabilités sur l'ensemble des résultats possibles d'une variable aléatoire.
* **Analogie :** La probabilité est la météo de demain (*"80% de chance de soleil"*). La loi de probabilité est le **climat annuel complet** montrant la fréquence de chaque type de temps.
* **Exemple concret :** Le lancer d'un dé à 6 faces suit une *loi uniforme discrète* où chaque face a une probabilité égale de $\frac{1}{6} \approx 16.7\%$.

---

### 4. Variables indépendantes
* **Définition rigoureuse :** Deux variables $X$ et $Y$ sont indépendantes si le résultat de l'une ne donne aucune information sur le résultat de l'autre : $P(X \cap Y) = P(X) \times P(Y)$.
* **Analogie :** Le **hasard total sans lien**. La couleur de tes chaussettes le matin n'a aucun impact sur la note que tu auras à ton examen l'après-midi.
* **Exemple concret :** Si tu lances deux fois d'affilée une pièce de monnaie, le premier résultat (*Pile*) n'influence absolument pas le second lancer.

---

### 5. Espérance, Variance & Écart-type
* **Définition rigoureuse :** 
  * **Espérance $\mathbb{E}[X]$ :** La valeur moyenne théorique.
  * **Variance $\text{Var}(X)$ :** La moyenne des carrés des écarts à la moyenne.
  * **Écart-type $\sigma$ :** La racine carrée de la variance ($\sigma = \sqrt{\text{Var}(X)}$), exprimée dans l'unité d'origine.
* **Analogie :** Pense à un **tir à l'arc**. L'espérance est le centre du mille. L'écart-type mesure à quel point tes flèches sont éparpillées autour du centre.
* **Exemple concret :** Deux classes ont 10/20 de moyenne. La classe A a des notes entre 9 et 11 (écart-type très faible $\approx 0.8$). La classe B n'a que des 0 et des 20 (écart-type très élevé $= 10$).

---

### 6. Corrélation linéaire
* **Définition rigoureuse :** Indicateur (souvent le coefficient $r$ de Pearson, compris entre $-1$ et $+1$) mesurant l'intensité et le sens de la relation linéaire entre deux variables.
* **Analogie :** Deux **danseurs synchronisés**. S'ils montent ensemble ($r \approx +1$), s'ils font l'opposé ($r \approx -1$), ou s'ils dansent chacun de leur côté sans lien ($r \approx 0$).
* **Exemple concret :** La température extérieure et la vente de glaces ont une forte corrélation positive ($r \approx +0.9$).

---

### 7. Moyenne, Médiane, Maximum & Minimum
* **Définition rigoureuse :**
  * **Moyenne :** Somme des valeurs divisée par l'effectif total.
  * **Médiane :** Valeur centrale du jeu de données trié (50% au-dessus, 50% en dessous).
  * **Min / Max :** Valeurs extrêmes observées.
* **Analogie :** Imagine une pièce avec 9 étudiants et Bill Gates. La **moyenne** des revenus devient absurde (tout le monde est "millionnaire"), alors que la **médiane** donne une vraie image du salaire de l'étudiant du milieu.
* **Exemple concret :** Série de prix `[10€, 12€, 15€, 18€, 500€]` $\rightarrow$ Médiane = $15€$ (très représentatif), Moyenne = $111€$ (fossée par la valeur extrême de 500€).

---

### 8. Quartiles
* **Définition rigoureuse :** Les trois valeurs ($Q_1, Q_2, Q_3$) qui coupent un ensemble de données triées en 4 parts égales (25% des données dans chaque tranche).
* **Analogie :** Le découpage d'une **pizza en 4 parts égales** avec 3 coups de couteau.
* **Exemple concret :** Lors d'un examen, si $Q_1 = 8/20$, cela veut dire que 25% des élèves les moins bien notés ont eu 8 ou moins. Si $Q_3 = 15/20$, 25% des meilleurs ont eu plus de 15.

---

### 9. Boxplot (Boîte à moustaches)
* **Définition rigoureuse :** Représentation graphique résumant une distribution en 5 valeurs ($Min, Q_1, \text{Médiane}, Q_3, Max$) tout en faisant ressortir les valeurs aberrantes (*outliers*).
* **Analogie :** Le **scanner de santé** instantané de tes données en une seule image.
* **Visualisation :**
```text
               Écart Interquartile (IQR = Q3 - Q1)
                       ┌─────────┬─────────┐
    |------------------|         |         |------------------|      o
  Minimum            Q1       Médiane     Q3               Maximum Outlier
```

### 10. Histogramme en statistique
* **Définition rigoureuse :** Représentation graphique sous forme de barres contiguës dont la surface est proportionnelle à la fréquence des données regroupées dans des intervalles de valeurs adjacents et disjoints (*bins*).
* **Analogy / Vulgarisation :** Pense à du **tri de linges ou de pièces**. Tu prends une montagne de pièces en vrac et tu les empiles dans des casiers selon leur valeur : les casiers les plus remplis forment les plus hautes piles.
* **Exemple concret :** Répartir les clients d'un magasin par tranches d'âge (`[18-25 ans]`, `[26-35 ans]`, etc.) pour identifier rapidement quelle est la tranche d'âge la plus représentée.

---

### 11. Le Théorème Central Limit (TCL)
* **Définition rigoureuse :** Théorème fondamental stipulant que la moyenne d'un grand nombre d'échantillons indépendants et identiquement distribués (i.i.d.) tend vers une **Loi Normale** (courbe en cloche), quelle que soit la loi de probabilité d'origine de la population.
* **Analogy / Vulgarisation :** **La magie du retour à l'ordre**. Même si le comportement individuel de chaque élément est totalement anarchique ou bizarre, la *moyenne d'un grand groupe* finit **toujours** par dessiner une belle courbe symétrique en cloche.
* **Exemple concret :** Si tu lances 100 fois un dé très truqué et que tu calcules la moyenne des lancers, puis que tu répètes cette expérience 10 000 fois, la distribution de toutes tes moyennes formera une loi normale parfaite.

---

### 12. Une dérivée
* **Définition rigoureuse :** La limite du taux d'accroissement d'une fonction lorsque l'intervalle tend vers zéro : $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$. Elle mesure la variation instantanée et correspond géométriquement à la pente de la tangente.
* **Analogy / Vulgarisation :** Le **compteur de vitesse instantanée** de ta voiture. La fonction de départ te donne la distance totale parcourue depuis le début du trajet, tandis que la dérivée te donne ta vitesse exacte à un instant précis.
* **Exemple concret :** En Machine Learning, la dérivée de la fonction d'erreur (Loss) indique à l'algorithme de combien et dans quel sens il doit ajuster ses paramètres ($w$) pour réduire ses fautes au minimum (*Descente de Gradient*).

## 🚀 Installation & Guide d'Exécution

Pour exécuter le projet **Philomathia** dans les meilleures conditions (que ce soit sur VS Code, Jupyter Notebook ou Google Colab), suis les étapes ci-dessous.

---

### 1. Préparation de l'environnement local

Ouvre ton terminal (ou l'invite de commande) et lance les instructions suivantes :

```bash
# 1. Récupérer le dépôt du projet
git clone [https://github.com/votre-utilisateur/philomathia.git](https://github.com/votre-utilisateur/philomathia.git)
cd philomathia

# 2. Créer un environnement virtuel (recommandé pour isoler les dépendances)
python -m venv venv

# 3. Activer l'environnement virtuel
# Sur macOS/Linux :
source venv/bin/activate
# Sur Windows (Command Prompt) :
venv\Scripts\activate.bat
# Sur Windows (PowerShell) :
.\venv\Scripts\Activate.ps1

# 4. Mettre à jour pip et installer les bibliothèques
pip install --upgrade pip
pip install -r requirements.txt
```

### Auteur

Projet réalisé dans le cadre de La Plateforme par Noémie FERAUD
