import pytest
from pydantic import ValidationError

from perception_worker.schemas import Detection, OverlayPacket, WorkerSession


def test_overlay_packet_uses_normalized_coordinates():
    packet = OverlayPacket(
        robot_id="robot-1", room="oscar-robot-1", frame_timestamp_us=123,
        frame_width=640, frame_height=480, model_id="model-1", model_name="Retail",
        model_version="1.0.0", task="product_detection",
        detections=[Detection(
            detection_id="d1", label="produit", class_id=0, confidence=0.91,
            x=0.1, y=0.2, width=0.3, height=0.4,
        )],
    )
    assert b"oscar.vision.overlay.v1" in packet.wire_bytes()


def test_detection_rejects_out_of_frame_box():
    with pytest.raises(ValidationError):
        Detection(
            detection_id="d1", label="bad", class_id=0, confidence=0.9,
            x=1.2, y=0.2, width=0.3, height=0.4,
        )

    with pytest.raises(ValidationError):
        Detection(
            detection_id="d2", label="wide", class_id=0, confidence=0.9,
            x=0.8, y=0.2, width=0.3, height=0.4,
        )


def test_worker_session_contract():
    session = WorkerSession(
        livekit_url="wss://stream.example.test",
        room="oscar-robot-1",
        identity="vision-robot-1",
        token="signed-token",
        expires_in_seconds=86400,
    )
    assert session.room == "oscar-robot-1"


def test_lossy_overlay_packet_stays_under_mtu():
    detections = [Detection(
        detection_id=f"detection-{index}", label=f"produit-{index}", class_id=index,
        confidence=0.5 + index / 100, x=0.1, y=0.1, width=0.2, height=0.2,
    ) for index in range(20)]
    packet = OverlayPacket(
        robot_id="robot-1", room="oscar-robot-1", frame_timestamp_us=123,
        frame_width=640, frame_height=480, model_id="model-1", model_name="Retail",
        model_version="1.0.0", task="product_detection", detections=detections,
    )
    payload = packet.lossy_wire_bytes()
    assert len(payload) <= 1200
    assert b'"detections_total":20' in payload


def test_lurl_locale_prime_sur_celle_annoncee_par_lapi():
    """Le worker tournant a cote du SFU doit joindre le media en direct.

    Par le nom public, depuis le meme hote, la signalisation aboutit — le
    participant apparait actif cote serveur — mais l'abonnement aux pistes
    n'arrive jamais. Le jeton reste emis par l'API ; seule l'URL est locale.
    """
    import inspect

    from perception_worker import worker

    source = inspect.getsource(worker.PerceptionWorker.run)
    assert "self.config.livekit_url or session.livekit_url" in source


def test_la_room_est_creee_dans_la_boucle():
    """Regression : une room construite hors boucle ne recoit aucun evenement.

    `rtc.Room` capture `asyncio.get_event_loop()` a sa creation. Construire le
    worker avant `asyncio.run` l'attache a une boucle qui ne tournera jamais :
    la connexion reussit, le participant apparait actif cote serveur, et aucune
    piste n'est jamais analysee.
    """
    import ast
    import inspect

    from perception_worker import main as module

    arbre = ast.parse(inspect.getsource(module))
    fonction = next(n for n in arbre.body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
    appels = [n for n in ast.walk(fonction) if isinstance(n, ast.Call)]
    constructions = [n for n in appels
                     if isinstance(n.func, ast.Name) and n.func.id == "PerceptionWorker"]
    assert not constructions, (
        "PerceptionWorker doit etre instancie dans une coroutine, pas dans main()"
    )


def test_une_piste_republiee_remplace_lanalyse_en_cours():
    """Regression : apres une reconnexion, la nouvelle piste etait ignoree.

    L'ancienne tache attendait sur un flux mort ; le worker restait connecte et
    abonne sans plus rien analyser. Mesure en production le 13 septembre :
    detections arretees a 10:13:12, « Ignoring additional video track ».
    """
    import inspect

    from perception_worker import worker

    source = inspect.getsource(worker.PerceptionWorker._wire_events)
    assert "Ignoring additional video track" not in source
    assert "self._track_task.cancel()" in source
    assert '"track_unsubscribed"' in source


def test_les_modeles_dune_box_tournent_en_parallele():
    import inspect

    from perception_worker import worker

    source = inspect.getsource(worker.PerceptionWorker._infer_frame)
    assert "asyncio.gather" in source


@pytest.mark.parametrize("config, attendu", [
    ({}, (640, 640)),
    ({"imgsz": [480, 640]}, (480, 640)),
    ({"imgsz": [500, 650]}, (480, 640)),     # ramene au multiple de 32
    ({"imgsz": [10, 99999]}, (160, 1280)),   # borne
    ({"imgsz": "grand"}, (640, 640)),        # invalide : valeur du manifeste
])
def test_une_box_regle_la_resolution_dinference(config, attendu):
    pytest.importorskip("ultralytics")
    from perception_worker.adapters.yolo import resolution_inference
    from perception_worker.schemas import ModelManifest

    manifeste = ModelManifest.model_validate({
        "id": "m", "name": "m", "version": "1", "task": "object_detection", "runtime": "ultralytics",
        "sha256": "0" * 64, "artifact_name": "m.pt", "artifact_path": "/a",
        "input": {"width": 640, "height": 640}, "output": {}, "labels": [],
        "inference_fps": 5, "confidence": 0.35, "iou_threshold": 0.45,
        "overlay_enabled": True, "incident_enabled": False, "camera": "primary",
        "config": config,
    })
    assert resolution_inference(manifeste) == attendu


def _paquet(n, model_id="modele-rayon"):
    import uuid
    return OverlayPacket(
        robot_id="r", room="room", frame_timestamp_us=1, frame_width=640, frame_height=480,
        model_id=model_id, model_name="SKU-110K", model_version="1", task="product_detection",
        detections=[Detection(detection_id=str(uuid.uuid4()), label="product", class_id=0,
                              confidence=0.5 + (i % 40) / 100, x=0.123456789, y=0.23456789,
                              width=0.0345678, height=0.0456789) for i in range(n)],
    )


def test_une_image_de_rayon_passe_en_entier():
    """Regression : 1 200 octets ne portaient que 5 ou 6 detections sur ~90."""
    import json
    payload, fiable = _paquet(90).fit_wire()
    corps = json.loads(payload)
    assert len(corps["detections"]) == 90
    assert fiable is True
    assert len(payload) < 14_000


def test_peu_de_detections_restent_sur_le_canal_lossy():
    payload, fiable = _paquet(3).fit_wire()
    assert fiable is False and len(payload) <= 1200


def test_le_contrat_lu_par_le_visualiseur_est_inchange():
    import json
    corps = json.loads(_paquet(2).fit_wire()[0])
    assert corps["schema"] == "oscar.vision.overlay.v1"
    assert set(corps["detections"][0]) >= {"detection_id", "label", "confidence", "x", "y", "width", "height"}


def test_les_identifiants_ne_se_chevauchent_pas_entre_modeles():
    import json
    a = json.loads(_paquet(2, "modeleA-xxx").fit_wire()[0])["detections"]
    b = json.loads(_paquet(2, "modeleB-yyy").fit_wire()[0])["detections"]
    assert not {d["detection_id"] for d in a} & {d["detection_id"] for d in b}


def test_au_dela_de_la_limite_on_garde_les_plus_sures():
    import json
    payload, fiable = _paquet(500).fit_wire()
    corps = json.loads(payload)
    assert fiable is True and len(payload) <= 14_000
    assert corps["detections_total"] == 500
    confiances = [d["confidence"] for d in corps["detections"]]
    assert min(confiances) >= 0.5 and confiances == sorted(confiances, reverse=True)
