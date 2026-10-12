# Aide à la décision statistique pour le trading intraday de turbos

Outil en Python (package + application Streamlit) qui, pour un indice dont tu fournis l'historique et pour le
contexte de la séance du jour, calcule :

- l'amplitude attendue de la séance et le régime de volatilité ;
- la distribution de la variation ouverture vers clôture, avec intervalles de confiance ;
- la probabilité de knock-out par levier et par sens (long / short) ;
- l'espérance, le risque et la mise de chaque produit testé (levier x stop), nette de coûts ;
- un verdict : un produit se démarque, ou **ne pas trader** si aucun avantage n'est statistiquement distinguable de zéro.

Ce n'est pas un conseil en investissement. Le passé ne garantit pas l'avenir, et un turbo peut perdre 100 % de la mise.

## Ce que l'outil ne prétend pas faire

Il ne donne pas « la meilleure décision possible ». Sur le Nasdaq-100 (voir la validation plus bas), l'information
exploitable est dans le **risque** (amplitude, KO), pas dans la **direction**. L'outil le dit en toute transparence :
quand rien n'est significatif après correction pour les tests multiples, le verdict est « ne pas trader ».

## Installation et lancement

```
cd decision_tool
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

En ligne de commande, sans interface :

```
python -m decision_tool.cli --file data/nasdaq100.xlsx --date 2026-10-12 --validate
python -m decision_tool.cli --file data/nasdaq100.xlsx --date 2026-10-12 --open 31000   # ouverture connue
```

Tests : `python -m unittest discover -s tests -t .`

## Utilisation au quotidien

1. Mets à jour le fichier de données du sous-jacent (dossier `data/`, voir `data/LISEZ-MOI.txt`), ou colle les
   séances manquantes dans la barre latérale.
2. Avant l'ouverture : laisse « Ouverture du jour » à 0. L'outil travaille en mode pré-ouverture.
3. Après l'ouverture : saisis le prix d'ouverture. L'outil intègre le gap (mode post-ouverture), plus précis sur l'amplitude.
4. Règle ton levier, tes stops, ton coût et ta perte maximale par trade, puis lance l'analyse.

Pour ajouter un sous-jacent (DAX, CAC 40, S&P 500), dépose son fichier dans `data/` : il apparaît dans la liste.

## Méthode

1. **Qualité des données** (`data.py`). Détection automatique des barres plates (High = Low) et des ouvertures
   recopiées de la clôture de la veille, qui faussent silencieusement le gap et le trajet depuis l'ouverture. Les
   calculs d'ouverture ne démarrent qu'à la première année saine (2001 pour le Nasdaq-100).
2. **Prévision de volatilité** (`volatility.py`). Modèle HAR par moindres carrés sur le log de la volatilité de
   Parkinson : régresseurs = volatilité de la veille, moyenne sur 5 jours, moyenne sur 22 jours (et |gap| en mode
   post-ouverture). On obtient une échelle `s` par séance.
3. **Simulation historique filtrée** (`conditional.py`). Chaque séance passée est exprimée en multiples de son échelle
   `s` : rendement ouverture-clôture, excursions défavorables long et short, amplitude. Pour la séance du jour, on
   multiplie ces résultats standardisés par l'échelle prévue : des milliers de scénarios réalistes, qui conservent le
   lien entre le trajet (KO) et le résultat final. C'est ainsi que toute l'histoire sert, quel que soit le régime.
4. **Analogues** (`conditional.py`). Noyau gaussien sur le contexte (rendement de la veille, gap, éventuellement régime
   de volatilité), avec rétrécissement vers la distribution inconditionnelle quand les analogues sont rares :
   `lambda = ESS / (ESS + 200)`.
5. **Produit** (`turbo.py`). Barrière à `1/levier`, perte totale si touchée, stop optionnel, coût aller-retour,
   taille de position pour une perte maximale fixée, fraction de Kelly (à ne prendre qu'au quart).
6. **Inférence** (`decision.py`). Intervalles de confiance par bootstrap pondéré (taille = ESS, donc honnête sur
   l'information disponible), p-values par bootstrap, correction de Holm-Bonferroni sur l'ensemble de la grille de
   produits. Un produit n'est retenu que si son espérance est positive ET significative après correction.
7. **Validation** (`validation.py`). Walk-forward : pour chaque année de test, tout est ré-estimé avec le seul passé.

## Validation obtenue sur le Nasdaq-100 (tests de 2006 à 2026, 5 223 séances)

| Question | Résultat |
|---|---|
| La prévision de volatilité bat-elle des benchmarks ? | Carré de la corrélation avec le réalisé : HAR 0,476 ; EWMA 0,94 : 0,412 ; moyenne 22 j : 0,394 ; veille seule : 0,367 |
| Les probabilités de KO sont-elles calibrées ? | Oui en moyenne (par exemple KO 2 % long : 7,3 % prévu contre 7,6 % réalisé) ; gain de Brier de 12 à 22 % contre la fréquence brute |
| Le contexte améliore-t-il la probabilité de hausse ? | Non : gain de Brier de -0,10 % (p = 0,21) avec la veille, +0,06 % (p = 0,55) avec veille et gap |

Point faible connu : pour les probabilités de KO très faibles (moins de 1 %, régimes très calmes), le modèle les
sous-estime d'un facteur 2 à 4 (par exemple 0,30 % prévu pour 1,25 % réalisé en barrière 2 % long). Lis ces cas comme
« faible mais pas négligeable ».

## Limites

- Données journalières : les stratégies intraday fines (breakout de début de séance, VWAP, momentum) demandent des
  barres de 1 à 5 minutes, non gérées en version 0.1.
- Le modèle du produit suppose une barrière à `1/levier`, un coût fixe en points de base, une sortie du stop au prix
  du stop (sans glissement) et aucun financement. Un vrai turbo a sa barrière et son spread.
- Les résultats supposent que le futur ressemble statistiquement au passé de l'échantillon. Les ruptures de régime
  (crise, changement de structure de marché) ne sont pas anticipées.
- L'interface Streamlit n'a pas pu être lancée dans l'environnement de développement (Streamlit absent) : seul son code
  a été exécuté avec un module simulé. Si un écran se comporte mal, le coeur de l'outil reste utilisable en ligne de commande.

## Structure

```
decision_tool/
  data.py          chargement, nettoyage, contrôle qualité
  features.py      rendements, gap, excursions, volatilité de Parkinson
  volatility.py    HAR, EWMA
  conditional.py   scénarios standardisés, analogues, bootstrap
  turbo.py         simulation du produit, Holm, Kelly
  decision.py      analyse complète et verdict
  validation.py    walk-forward : volatilité, calibration du KO, compétence directionnelle
  report.py        mise en forme
  cli.py           ligne de commande
app/streamlit_app.py
tests/
data/
```
