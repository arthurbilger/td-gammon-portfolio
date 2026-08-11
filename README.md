# From TD-Gammon to Portfolio Allocation

Reproducing Tesauro's TD-Gammon algorithm (1992) and applying the same
Temporal Difference learning framework to dynamic portfolio allocation.

## Project Status
Work in progress - Phase 1: RL foundations

## Stack
- Python 3.14
- NumPy (from-scratch neural network)
- PyTorch (production network)
- Gymnasium (RL environment API)

## Structure
- src/ : Production code (environment, agent, training)
- notebooks/ : Pedagogical notebooks
- data/ : Market data (not tracked)
- outputs/ : Saved models, charts
- tests/ : Unit tests

## References
- Tesauro, G. (1995). Temporal Difference Learning and TD-Gammon. Communications of the ACM.
- Sutton, R. & Barto, A. (2018). Reinforcement Learning: An Introduction.

## Author
M1 Finance, Universite Paris Dauphine

## Résultats & limites

L'agent TD(λ) transposé du backgammon vers l'allocation de portefeuille bat les
benchmarks classiques (60/40, équipondéré) sur toutes les métriques ajustées du risque,
mais n'atteint pas le Calmar ratio du S&P 500 seul sur la période testée (2020-2024).
Une recherche systématique d'hyperparamètres (protocole de comparaison contrôlée,
même méthode que pour le tuning TD(λ) du projet backgammon) a permis d'identifier
précisément cette limite : l'espace d'actions discret (8 profils statiques) manque de
la granularité nécessaire pour égaler un indice pur sur un régime haussier soutenu.
Cette limite est documentée plutôt que dissimulée — elle illustre une compréhension
du compromis rendement/risque en gestion de portefeuille, davantage qu'un chiffre
de performance isolé.
