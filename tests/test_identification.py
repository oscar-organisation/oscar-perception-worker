import numpy as np

from perception_worker.adapters.identification import Identite, appliquer_identites, decider, recadrer
from perception_worker.schemas import Detection


def det(x, y, w, h, i="d"):
    return Detection(detection_id=i, label="product", class_id=0, confidence=0.8, x=x, y=y, width=w, height=h)


PANZANI = Identite("3038350013804", "Spaghetti n°5", "Panzani", "en:spaghetti", 0.91)


def test_un_nom_nest_retenu_quau_dessus_du_seuil():
    scores = np.array([[0.93, 0.80], [0.70, 0.60]])
    indices = np.array([[4, 9], [2, 3]])
    assert decider(scores, indices, seuil=0.85, marge=0.02) == [4, None]


def test_deux_candidats_trop_proches_ne_donnent_pas_de_nom():
    """Si la galerie hesite entre deux produits, afficher l'un serait un pari."""
    scores = np.array([[0.92, 0.915]])
    assert decider(scores, np.array([[1, 2]]), seuil=0.85, marge=0.02) == [None]


def test_le_libelle_ajoute_la_marque_sans_la_repeter():
    assert PANZANI.libelle == "Panzani · Spaghetti n°5"
    assert Identite("1", "Nutella pâte à tartiner", "Nutella", "", 0.9).libelle == "Nutella pâte à tartiner"


def test_un_libelle_long_est_tronque():
    long = Identite("1", "x" * 80, "", "", 0.9)
    assert len(long.libelle) == 40 and long.libelle.endswith("…")


def test_un_produit_trop_petit_nest_pas_recadre():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert recadrer(image, 0.5, 0.5, 0.02, 0.02) is None          # ~13 px
    r = recadrer(image, 0.4, 0.4, 0.1, 0.2)
    assert r is not None and r.flags["C_CONTIGUOUS"]


def test_le_recadrage_reste_dans_limage():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    r = recadrer(image, 0.95, 0.9, 0.05, 0.1)
    assert r is not None and r.shape[0] <= 480 and r.shape[1] <= 640


def test_le_nom_suit_la_boite_qui_recouvre():
    memoire = [(det(0.40, 0.40, 0.10, 0.20), PANZANI, 100.0)]
    courant = [det(0.41, 0.41, 0.10, 0.20, "a"), det(0.80, 0.10, 0.08, 0.15, "b")]
    nommees = appliquer_identites(courant, memoire, maintenant=101.0)
    assert nommees[0].label == "Panzani · Spaghetti n°5" and nommees[0].product_code == "3038350013804"
    assert nommees[1].label == "product" and nommees[1].product_code is None


def test_un_nom_perime_nest_plus_applique():
    memoire = [(det(0.40, 0.40, 0.10, 0.20), PANZANI, 100.0)]
    assert appliquer_identites([det(0.40, 0.40, 0.10, 0.20)], memoire, maintenant=104.0)[0].label == "product"


def test_le_paquet_transporte_lidentite():
    import json

    from perception_worker.schemas import OverlayPacket

    d = appliquer_identites([det(0.40, 0.40, 0.10, 0.20)], [(det(0.40, 0.40, 0.10, 0.20), PANZANI, 1.0)], 1.5)
    paquet = OverlayPacket(robot_id="r", room="room", frame_timestamp_us=1, frame_width=640, frame_height=480,
                           model_id="m", model_name="n", model_version="1", task="product_detection", detections=d)
    corps = json.loads(paquet.fit_wire()[0])["detections"][0]
    assert corps["label"] == "Panzani · Spaghetti n°5" and corps["brand"] == "Panzani"


def test_un_produit_trop_petit_en_hauteur_nest_pas_identifie():
    """A 80 px de haut, le bon produit n'arrive en tete qu'une fois sur trois (banc)."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert recadrer(image, 0.4, 0.4, 0.1, 0.15, hauteur_min=140) is None     # ~78 px
    assert recadrer(image, 0.4, 0.2, 0.1, 0.35, hauteur_min=140) is not None  # ~180 px
