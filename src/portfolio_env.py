"""
portfolio_env_v2.py — Allocation multi-actifs par selection de profils.

REFONTE v2 — trois corrections majeures par rapport a v1 :

1. REWARD RECALIBREE. En v1, reward = 100*(r - aversion*r^2) avec r decimal :
   a frequence journaliere (r ~ 0.01), la penalite ne pesait que 1-3% du
   signal. L'agent maximisait de facto le rendement brut -> 100% SPY.
   En v2, le rendement passe en pourcents AVANT la penalite quadratique :
   reward = r_pct - aversion * r_pct^2, avec aversion ~ 0.15 : sur un jour
   a +/-1%, la penalite pese ~15% du signal — une vraie aversion au risque.

2. ACTIONS = PROFILS D'ALLOCATION, plus une grille de poids bruts.
   Avec 10 actifs, une grille par pas de 25% donnerait 715 corners a
   explorer — ingerables. A la place, l'agent choisit parmi 8 profils
   professionnels (du plus defensif au plus agressif + 2 profils DYNAMIQUES
   dont les poids s'adaptent aux conditions courantes). L'agent devient un
   "selecteur de regime" — le pattern hierarchique decrit dans le monographe
   CFA (Halperin, Kolm & Ritter 2025) : agent strategique de haut niveau.
   Bonus : chaque profil etant deja diversifie, les corner-solutions a 100%
   sur un titre deviennent structurellement impossibles.

3. FEATURES DE CORRELATION COMPACTEES : avec 10 actifs, les 45 paires
   satureraient l'etat. On garde 2 agregats : correlation moyenne du marche
   (mesure du regime risk-on/risk-off) et correlation actions-obligations
   (SPY-TLT, LE signal de diversification).

Univers (10 ETFs liquides, historique complet depuis 2010) :
  SPY (US large cap)   QQQ (US tech)      IWM (US small cap)
  EFA (dev. internat.) EEM (emergents)    TLT (Treasuries 20y+)
  IEF (Treasuries 7-10y) LQD (credit IG)  GLD (or)  VNQ (REITs US)
"""

import numpy as np
import pandas as pd

TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "VNQ"]


# ============================================================
# 1. FEATURES
# ============================================================

def calculer_features(prix, fenetre_corr=60, fenetre_momentum=60, lambda_ewma=0.94):
    """
    Features par actif : ret5, ret20, momentum 60j, vol EWMA (RiskMetrics).
    Features de marche : correlation moyenne inter-actifs + correlation SPY-TLT.
    Retourne aussi vols et momentum BRUTS (pour les profils dynamiques).
    """
    rendements = np.log(prix / prix.shift(1))
    actifs = list(prix.columns)

    feats = {}
    vols_brutes = {}
    for a in actifs:
        r = rendements[a]
        feats[f"ret5_{a}"] = r.rolling(5).mean()
        feats[f"ret20_{a}"] = r.rolling(20).mean()
        feats[f"mom_{a}"] = np.log(prix[a] / prix[a].shift(fenetre_momentum))
        var_ewma = (r ** 2).ewm(alpha=1 - lambda_ewma).mean()
        vols_brutes[a] = np.sqrt(var_ewma * 252)
        feats[f"vol_{a}"] = vols_brutes[a]

    # Correlation moyenne du marche (regime risk-on/off)
    corr_glissante = rendements.rolling(fenetre_corr).corr()
    corr_moyenne = corr_glissante.groupby(level=0).apply(
        lambda m: (m.values.sum() - len(m)) / (len(m) ** 2 - len(m)))
    feats["corr_moyenne"] = corr_moyenne

    # Correlation actions-obligations (SPY-TLT) : LE signal de diversification
    feats["corr_SPY_TLT"] = rendements["SPY"].rolling(fenetre_corr).corr(rendements["TLT"])

    features = pd.DataFrame(feats, index=prix.index)
    vols = pd.DataFrame(vols_brutes, index=prix.index)
    momentum = pd.DataFrame({a: feats[f"mom_{a}"] for a in actifs}, index=prix.index)
    return features, rendements, vols, momentum


def normaliser_features(features_train, features):
    """Z-score sur statistiques du TRAIN uniquement (anti look-ahead)."""
    mu = features_train.mean()
    sigma = features_train.std().replace(0, 1.0)
    return (features - mu) / sigma


# ============================================================
# 2. PROFILS D'ALLOCATION (l'espace d'actions)
# ============================================================
# Ordre des actifs : SPY QQQ IWM EFA EEM | TLT IEF LQD | GLD | VNQ

PROFILS_STATIQUES = {
    # nom : poids sur les 10 actifs (somment a 1.0)
    "Ultra-defensif": [0.10, 0.00, 0.00, 0.00, 0.00, 0.25, 0.30, 0.15, 0.15, 0.05],
    "Defensif":       [0.15, 0.05, 0.00, 0.10, 0.00, 0.20, 0.25, 0.10, 0.10, 0.05],
    "Equilibre":      [0.25, 0.10, 0.00, 0.10, 0.05, 0.15, 0.10, 0.10, 0.10, 0.05],
    "Croissance":     [0.30, 0.20, 0.05, 0.10, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05],
    "Agressif":       [0.35, 0.25, 0.10, 0.10, 0.10, 0.00, 0.00, 0.00, 0.05, 0.05],
    "Equipondere":    [0.10] * 10,
}
NOMS_PROFILS = list(PROFILS_STATIQUES.keys()) + ["Risk-parity", "Momentum"]


def poids_risk_parity(vols_jour):
    """Inverse-vol : poids proportionnels a 1/volatilite (risk parity naif)."""
    inv = 1.0 / np.maximum(vols_jour, 1e-6)
    return inv / inv.sum()


def poids_momentum(momentum_jour):
    """Poids proportionnels au momentum 60j positif ; equipondere si tout <= 0."""
    m = np.maximum(momentum_jour, 0.0)
    if m.sum() < 1e-9:
        return np.ones(len(m)) / len(m)
    return m / m.sum()


# ============================================================
# 3. ENVIRONNEMENT v2
# ============================================================

class PortfolioEnvV2:
    """
    A chaque pas, l'agent choisit UN PROFIL parmi les 8. Les poids effectifs
    sont ceux du profil (statiques, ou calcules dynamiquement pour
    Risk-parity et Momentum a partir des conditions courantes).
    Reward = r_pct_net - aversion * r_pct_net^2  (r en POURCENTS).
    """

    def __init__(self, rendements, features_norm, vols, momentum,
                 duree_episode=60, cout_transaction=0.0005,
                 aversion_risque=0.15, seed=None):
        communes = (features_norm.dropna().index
                    .intersection(rendements.dropna().index)
                    .intersection(vols.dropna().index)
                    .intersection(momentum.dropna().index))
        self.rendements = rendements.loc[communes].to_numpy()
        self.features = features_norm.loc[communes].to_numpy()
        self.vols = vols.loc[communes].to_numpy()
        self.momentum = momentum.loc[communes].to_numpy()
        self.dates = communes
        self.n_actifs = self.rendements.shape[1]
        self.n_actions = len(NOMS_PROFILS)
        self.duree_episode = duree_episode
        self.cout_transaction = cout_transaction
        self.aversion_risque = aversion_risque
        self.rng = np.random.default_rng(seed)

        self._profils_statiques = np.array(list(PROFILS_STATIQUES.values()))
        # Etat = features marche + poids courants + temps restant
        self.dim_etat = self.features.shape[1] + self.n_actifs + 1

    def poids_du_profil(self, k, t=None):
        """Poids effectifs du profil k aux conditions du jour t."""
        t = self.t if t is None else t
        if k < len(self._profils_statiques):
            return self._profils_statiques[k]
        if NOMS_PROFILS[k] == "Risk-parity":
            return poids_risk_parity(self.vols[t])
        return poids_momentum(self.momentum[t])

    def reset(self, t_debut=None):
        t_max = len(self.rendements) - self.duree_episode - 1
        self.t_debut = self.rng.integers(0, t_max) if t_debut is None else t_debut
        self.t = self.t_debut
        self.pas_restants = self.duree_episode
        self.poids = np.ones(self.n_actifs) / self.n_actifs
        return self._etat(self.poids)

    def _etat(self, poids):
        return np.concatenate([self.features[self.t], poids,
                               [self.pas_restants / self.duree_episode]])

    def etats_candidats(self):
        """Afterstates : etat resultant de chaque profil + cout certain du switch."""
        X = np.empty((self.n_actions, self.dim_etat))
        couts = np.empty(self.n_actions)
        for k in range(self.n_actions):
            w = self.poids_du_profil(k)
            X[k] = self._etat(w)
            couts[k] = self.cout_transaction * np.abs(w - self.poids).sum()
        return X, couts

    def step(self, k):
        w = self.poids_du_profil(k)
        turnover = np.abs(w - self.poids).sum()
        cout = self.cout_transaction * turnover

        r_actifs = self.rendements[self.t + 1]
        r_net = float(w @ r_actifs) - cout

        # REWARD v2 : passage en pourcents AVANT la penalite quadratique
        r_pct = 100.0 * r_net
        reward = r_pct - self.aversion_risque * r_pct ** 2

        croissance = np.exp(r_actifs)
        w_derive = w * croissance
        self.poids = w_derive / w_derive.sum()

        self.t += 1
        self.pas_restants -= 1
        termine = self.pas_restants == 0
        return self._etat(self.poids), reward, termine

    def verifier_invariant(self):
        assert abs(self.poids.sum() - 1.0) < 1e-8, f"Poids somment a {self.poids.sum()}"


# ============================================================
# 4. TEARSHEET (identique v1)
# ============================================================

def tearsheet(rendements_log, taux_sans_risque=0.02, jours_an=252):
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
    max_dd = float((equity / sommets - 1).min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino,
            "MaxDD": max_dd, "Calmar": calmar}