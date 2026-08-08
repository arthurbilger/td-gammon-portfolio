"""
portfolio_env.py — Environnement RL pour l'allocation dynamique de portefeuille.

Formalisation alignee sur le standard professionnel decrit dans :
  Halperin, Kolm & Ritter (2025), "Reinforcement Learning and Inverse
  Reinforcement Learning: A Practitioner's Guide for Investment Management",
  in AI in Asset Management, CFA Institute Research Foundation.

  - State  : poids actuels + rendements recents + volatilites + correlations
  - Action : vecteur de poids cibles (espace discretise)
  - Reward : rendement ajuste du risque, net des couts de transaction

Architecture identique a BackgammonEnv (reset / step / etat) pour permettre
le transfert direct de l'agent TD(lambda) du projet TD-Gammon.
"""

import numpy as np
import pandas as pd


# ============================================================
# 1. FEATURE ENGINEERING
# ============================================================

def calculer_features(prix, fenetre_vol=20, fenetre_corr=60, fenetre_momentum=60,
                      lambda_ewma=0.94):
    """
    Transforme une matrice de prix en features d'etat pour l'agent.

    Parametres
    ----------
    prix : DataFrame (dates x actifs) de prix ajustes.
    lambda_ewma : facteur de lissage EWMA (0.94 = standard RiskMetrics).

    Retourne
    --------
    features : DataFrame indexe par date, aligne sur prix (NaN au debut, a purger).
    rendements : DataFrame des log-rendements journaliers.

    Notes financieres
    -----------------
    - Log-rendements : additifs dans le temps (somme sur T jours = rendement
      cumule), propriete indispensable pour des rewards additifs en RL.
    - Volatilite EWMA (RiskMetrics) : var_t = lam*var_{t-1} + (1-lam)*r_t^2.
      Reagit plus vite qu'une fenetre glissante simple aux changements de
      regime — exactement la feature qui restaure la propriete de Markov.
    - Correlations glissantes : capturent le "correlation breakdown" en stress.
    """
    rendements = np.log(prix / prix.shift(1))

    feats = {}
    actifs = list(prix.columns)

    # Rendements moyens court terme et momentum par actif
    for a in actifs:
        r = rendements[a]
        feats[f"ret5_{a}"] = r.rolling(5).mean()
        feats[f"ret20_{a}"] = r.rolling(20).mean()
        feats[f"mom_{a}"] = np.log(prix[a] / prix[a].shift(fenetre_momentum))
        # Volatilite EWMA annualisee (RiskMetrics lambda = 0.94)
        var_ewma = (r ** 2).ewm(alpha=1 - lambda_ewma).mean()
        feats[f"vol_{a}"] = np.sqrt(var_ewma * 252)

    # Correlations glissantes par paire
    for i in range(len(actifs)):
        for j in range(i + 1, len(actifs)):
            a, b = actifs[i], actifs[j]
            feats[f"corr_{a}_{b}"] = rendements[a].rolling(fenetre_corr).corr(rendements[b])

    features = pd.DataFrame(feats, index=prix.index)
    return features, rendements


def normaliser_features(features_train, features):
    """
    Z-score : (x - moyenne_train) / ecart_type_train.
    CRUCIAL : les statistiques de normalisation viennent UNIQUEMENT du train
    set — utiliser celles du test set serait du look-ahead bias.
    """
    mu = features_train.mean()
    sigma = features_train.std().replace(0, 1.0)
    return (features - mu) / sigma


# ============================================================
# 2. ESPACE D'ACTIONS DISCRET
# ============================================================

def generer_actions(n_actifs=3, pas=0.25):
    """
    Genere tous les vecteurs de poids possibles par pas de 25%,
    sommant a 1.0 (contrainte fully-invested, pas de cash ni de levier).
    Pour 3 actifs et pas=0.25 : 15 allocations possibles.

    La discretisation permet de reutiliser l'architecture value-based
    TD(lambda) du projet TD-Gammon (le monographe CFA note que les poids
    continus exigeraient des methodes policy-gradient, plus complexes).
    """
    n_pas = int(round(1.0 / pas))
    actions = []

    def recurse(restant, courant):
        if len(courant) == n_actifs - 1:
            actions.append(courant + [restant])
            return
        for k in range(restant + 1):
            recurse(restant - k, courant + [k])

    recurse(n_pas, [])
    return np.array(actions, dtype=float) * pas


# ============================================================
# 3. ENVIRONNEMENT MDP
# ============================================================

class PortfolioEnv:
    """
    Environnement d'allocation dynamique.

    Episode = fenetre de `duree_episode` jours de bourse tiree du jeu de
    donnees. A chaque pas : l'agent choisit un vecteur de poids cible,
    paie les couts de transaction sur le turnover, puis le marche bouge.

    Reward par pas (utilite moyenne-variance, cf. QLBS / Halperin 2020) :
        r_t = 100 * [ ret_net_t - aversion_risque * ret_net_t^2 ]
    ou ret_net_t = rendement log du portefeuille net des couts.
    Le facteur 100 met les rewards a une echelle adaptee au reseau.
    """

    def __init__(self, rendements, features_norm, actions,
                 duree_episode=60, cout_transaction=0.0005,
                 aversion_risque=1.0, seed=None):
        # Alignement strict features / rendements, purge des NaN initiaux
        communes = features_norm.dropna().index.intersection(rendements.dropna().index)
        self.rendements = rendements.loc[communes].to_numpy()
        self.features = features_norm.loc[communes].to_numpy()
        self.dates = communes
        self.actions = actions
        self.n_actifs = self.rendements.shape[1]
        self.duree_episode = duree_episode
        self.cout_transaction = cout_transaction
        self.aversion_risque = aversion_risque
        self.rng = np.random.default_rng(seed)

        self.n_features_marche = self.features.shape[1]
        # Etat = features marche + poids courants + temps restant normalise
        self.dim_etat = self.n_features_marche + self.n_actifs + 1

    # ---------- API identique a BackgammonEnv ----------

    def reset(self, t_debut=None):
        """Demarre un episode. t_debut aleatoire dans le train par defaut."""
        t_max = len(self.rendements) - self.duree_episode - 1
        self.t_debut = self.rng.integers(0, t_max) if t_debut is None else t_debut
        self.t = self.t_debut
        self.pas_restants = self.duree_episode
        self.poids = np.ones(self.n_actifs) / self.n_actifs  # depart equiponderee
        return self._etat(self.poids)

    def _etat(self, poids):
        """Vecteur d'etat : features marche du jour + poids + temps restant."""
        return np.concatenate([
            self.features[self.t],
            poids,
            [self.pas_restants / self.duree_episode],
        ])

    def etats_candidats(self):
        """
        AFTERSTATES : pour chaque action possible, l'etat resultant si on
        l'applique MAINTENANT (avant le mouvement de marche) + le cout
        certain du rebalancement. Transposition directe du mecanisme
        TD-Gammon : evaluer la position apres son coup, avant les des.
        """
        X = np.empty((len(self.actions), self.dim_etat))
        couts = np.empty(len(self.actions))
        for k, w in enumerate(self.actions):
            X[k] = self._etat(w)
            couts[k] = self.cout_transaction * np.abs(w - self.poids).sum()
        return X, couts

    def step(self, indice_action):
        """
        Applique l'action, fait avancer le marche d'un jour.
        Retourne (nouvel_etat, reward, termine).
        """
        w_cible = self.actions[indice_action]
        turnover = np.abs(w_cible - self.poids).sum()
        cout = self.cout_transaction * turnover

        # Rendement du portefeuille sur le jour qui suit
        r_actifs = self.rendements[self.t + 1]
        r_portefeuille = float(w_cible @ r_actifs) - cout

        # Utilite moyenne-variance par pas (echelle x100)
        reward = 100.0 * (r_portefeuille - self.aversion_risque * r_portefeuille ** 2)

        # Derive des poids avec le marche (buy-and-hold intra-pas)
        croissance = np.exp(r_actifs)
        w_derive = w_cible * croissance
        self.poids = w_derive / w_derive.sum()

        self.t += 1
        self.pas_restants -= 1
        termine = self.pas_restants == 0

        return self._etat(self.poids), reward, termine

    def verifier_invariant(self):
        """Somme des poids = 1, toujours (equivalent des 3 pions du backgammon)."""
        assert abs(self.poids.sum() - 1.0) < 1e-8, f"Poids somment a {self.poids.sum()}"


# ============================================================
# 4. METRIQUES PROFESSIONNELLES (tearsheet)
# ============================================================

def tearsheet(rendements_log, taux_sans_risque=0.02, jours_an=252):
    """
    Calcule les metriques standard d'une strategie a partir de ses
    log-rendements journaliers. Retourne un dict.

    - CAGR      : croissance annualisee composee
    - Vol       : ecart-type annualise
    - Sharpe    : (CAGR - rf) / Vol
    - Sortino   : comme Sharpe mais volatilite des seuls jours negatifs
                  (ne penalise pas la volatilite haussiere)
    - Max DD    : perte maximale depuis un sommet (peak-to-trough)
    - Calmar    : CAGR / |Max DD| — rendement par unite de pire perte
    """
    r = np.asarray(rendements_log, dtype=float)
    n = len(r)
    equity = np.exp(np.cumsum(r))

    cagr = float(equity[-1] ** (jours_an / n) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(jours_an))
    sharpe = (cagr - taux_sans_risque) / vol if vol > 0 else np.nan

    r_neg = r[r < 0]
    vol_neg = float(r_neg.std(ddof=1) * np.sqrt(jours_an)) if len(r_neg) > 1 else np.nan
    sortino = (cagr - taux_sans_risque) / vol_neg if vol_neg and vol_neg > 0 else np.nan

    sommets = np.maximum.accumulate(equity)
    drawdowns = equity / sommets - 1
    max_dd = float(drawdowns.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino,
            "MaxDD": max_dd, "Calmar": calmar}