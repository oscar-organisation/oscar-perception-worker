from perception_worker.focus import nombre_cible, score_focus, selectionner
from perception_worker.schemas import Detection


def det(x, y, w, h, i=0):
    return Detection(detection_id=str(i), label="product", class_id=0, confidence=0.8,
                     x=x, y=y, width=w, height=h)


def test_sans_reglage_tout_passe():
    ds = [det(0.1 * i, 0.1, 0.05, 0.05, i) for i in range(8)]
    assert selectionner(ds, 0) == ds
    assert nombre_cible({}) == 0


def test_on_garde_le_nombre_demande():
    ds = [det(0.02 * i, 0.3, 0.05, 0.08, i) for i in range(40)]
    assert len(selectionner(ds, 6)) == 6


def test_un_produit_au_centre_lemporte_sur_un_plus_gros_au_bord():
    """Le critere retenu : proche du point vise, pas seulement grand."""
    centre = det(0.45, 0.45, 0.10, 0.10, 1)
    bord = det(0.88, 0.0, 0.12, 0.30, 2)   # plus grand, colle au coin
    assert score_focus(centre) > score_focus(bord)


def test_a_distance_egale_le_plus_proche_lemporte():
    petit = det(0.45, 0.45, 0.05, 0.05, 1)
    grand = det(0.425, 0.425, 0.15, 0.15, 2)
    assert selectionner([petit, grand], 1)[0].detection_id == "2"


def test_la_selection_precedente_resiste_a_un_candidat_voisin():
    """Sans continuite, deux scores proches alternaient et la vue clignotait."""
    garde = det(0.40, 0.40, 0.10, 0.10, 1)    # centre (0,45 ; 0,45)
    rival = det(0.46, 0.46, 0.10, 0.10, 2)    # centre (0,51 ; 0,51) : plus central
    assert selectionner([garde, rival], 1)[0].detection_id == "2"
    assert selectionner([garde, rival], 1, precedentes=[garde])[0].detection_id == "1"


def test_reglage_invalide_ou_hors_bornes():
    assert nombre_cible({"focus_max": "six"}) == 0
    assert nombre_cible({"focus_max": 500}) == 50
    assert nombre_cible({"focus_max": 6}) == 6
