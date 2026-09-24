import pytest

from perception_worker.exclusion import filtrer, lire_zones
from perception_worker.schemas import Detection


def det(x, y, w, h, label="bottle"):
    return Detection(detection_id=label, label=label, class_id=0, confidence=0.9,
                     x=x, y=y, width=w, height=h)


PINCE = lire_zones("0,0.72,1,1")


def test_une_boite_dessinee_sur_la_pince_disparait():
    """Le cas mesure : 'dirty_floor' sur toute la bande basse."""
    assert filtrer([det(0.15, 0.72, 0.82, 0.28, "dirty_floor")], PINCE, 0.6) == []


def test_un_objet_juste_devant_les_machoires_reste():
    # 25 % de la boite deborde sur la bande : l'objet a saisir doit rester visible.
    garde = filtrer([det(0.40, 0.50, 0.10, 0.293)], PINCE, 0.6)
    assert len(garde) == 1


def test_un_objet_loin_de_la_pince_nest_pas_touche():
    assert len(filtrer([det(0.58, 0.45, 0.05, 0.16)], PINCE, 0.6)) == 1


def test_sans_zone_rien_nest_filtre():
    ds = [det(0.1, 0.8, 0.5, 0.2)]
    assert filtrer(ds, lire_zones(""), 0.6) == ds


def test_plusieurs_zones():
    zones = lire_zones("0,0.72,1,1; 0,0,0.1,1")
    assert filtrer([det(0.0, 0.2, 0.08, 0.3)], zones, 0.6) == []


@pytest.mark.parametrize("brut", ["0,0.7,1", "0,0.8,1,0.7", "-0.1,0,1,1", "a,b,c,d"])
def test_une_zone_invalide_fait_echouer_le_demarrage(brut):
    with pytest.raises(ValueError):
        lire_zones(brut)
