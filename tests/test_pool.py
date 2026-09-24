import asyncio
import threading
import time
from pathlib import Path

import numpy as np

from perception_worker.config import WorkerConfig
from perception_worker.schemas import ModelManifest, RuntimeManifest
from perception_worker.worker import RobotSession, WorkerPool


def configuration(tmp_path: Path) -> WorkerConfig:
    return WorkerConfig(
        central_api_url="https://admin.example.test",
        worker_api_key="secret",
        worker_id="worker-test",
        worker_capacity=4,
        livekit_url="ws://livekit:7880",
        model_cache=tmp_path,
        manifest_refresh_seconds=10,
    )


def modele(model_id: str = "modele-partage", sha: str = "a" * 64) -> ModelManifest:
    return ModelManifest(
        id=model_id,
        name=model_id,
        version="1.0.0",
        task="product_detection",
        runtime="ultralytics",
        sha256=sha,
        artifact_name=f"{model_id}.pt",
        artifact_path=f"/models/{model_id}",
        inference_fps=5,
        overlay_enabled=False,
    )


def manifeste(robot_id: str, models: list[ModelManifest]) -> RuntimeManifest:
    return RuntimeManifest(
        schema_version="1.1",
        robot_id=robot_id,
        room=f"oscar-{robot_id}",
        models=models,
    )


class RegistreFactice:
    def __init__(self, manifests=None):
        self.manifests = manifests or {}

    async def artifact(self, model):
        return Path(f"/tmp/{model.artifact_name}")

    async def manifest(self, robot_id):
        return self.manifests[robot_id]


class AdaptateurMesure:
    def __init__(self):
        self.appels = 0
        self.simultanes = 0
        self.maximum_simultane = 0
        self._verrou = threading.Lock()

    def infer(self, frame, manifest):
        with self._verrou:
            self.simultanes += 1
            self.maximum_simultane = max(self.maximum_simultane, self.simultanes)
        time.sleep(0.02)
        with self._verrou:
            self.simultanes -= 1
            self.appels += 1
        return []


def test_deux_sessions_partageant_un_modele_gardent_leur_cadence(tmp_path):
    """Une horloge sur le modèle ferait sauter l'inférence du second robot."""
    async def scenario():
        adaptateur = AdaptateurMesure()
        pool = WorkerPool(
            configuration(tmp_path),
            registry=RegistreFactice(),
            adapter_factory=lambda model, artifact: adaptateur,
        )
        model = modele()
        charge = await pool.acquire_model(model)
        await pool.acquire_model(model)

        sessions = [RobotSession(pool, robot_id) for robot_id in ("robot-1", "robot-2")]
        for session in sessions:
            session.models = {model.id: charge}
            session.model_manifests = {model.id: model}
            session.runtime = manifeste(session.robot_id, [model])

        image = np.zeros((32, 32, 3), dtype=np.uint8)
        await asyncio.gather(*(
            session._infer_frame(image, 32, 32, 1) for session in sessions
        ))
        assert adaptateur.appels == 2
        assert all(session._prochaine_inference[model.id] > 0 for session in sessions)

        # L'adaptateur lourd est partagé, mais ses appels ne s'entrelacent pas.
        assert adaptateur.maximum_simultane == 1

        await asyncio.gather(*(
            session._infer_frame(image, 32, 32, 2) for session in sessions
        ))
        assert adaptateur.appels == 2

    asyncio.run(scenario())


def test_deux_modeles_distincts_dune_box_restent_paralleles(tmp_path):
    class Mesure:
        simultanes = 0
        maximum = 0
        verrou = threading.Lock()

    class Adaptateur:
        def infer(self, frame, manifest):
            with Mesure.verrou:
                Mesure.simultanes += 1
                Mesure.maximum = max(Mesure.maximum, Mesure.simultanes)
            time.sleep(0.02)
            with Mesure.verrou:
                Mesure.simultanes -= 1
            return []

    async def scenario():
        pool = WorkerPool(
            configuration(tmp_path),
            registry=RegistreFactice(),
            adapter_factory=lambda model, artifact: Adaptateur(),
        )
        modeles = [modele("modele-a", "a" * 64), modele("modele-b", "b" * 64)]
        charges = [await pool.acquire_model(model) for model in modeles]
        session = RobotSession(pool, "robot-1")
        session.models = {model.id: charge for model, charge in zip(modeles, charges)}
        session.model_manifests = {model.id: model for model in modeles}
        session.runtime = manifeste("robot-1", modeles)

        await session._infer_frame(np.zeros((32, 32, 3), dtype=np.uint8), 32, 32, 1)
        assert Mesure.maximum == 2
        await session._release_all_models()

    asyncio.run(scenario())


def test_une_trame_non_due_nest_pas_convertie(tmp_path):
    class Adaptateur:
        def infer(self, frame, manifest):
            return []

    class ImageInterdite:
        def convert(self, format):
            raise AssertionError("la conversion RGB ne devait pas avoir lieu")

    class Flux:
        def __aiter__(self):
            async def evenements():
                yield type("Evenement", (), {"frame": ImageInterdite()})()
            return evenements()

    async def scenario():
        pool = WorkerPool(
            configuration(tmp_path), registry=RegistreFactice(),
            adapter_factory=lambda model, artifact: Adaptateur(),
        )
        model = modele()
        charge = await pool.acquire_model(model)
        session = RobotSession(pool, "robot-1")
        session.models = {model.id: charge}
        session.model_manifests = {model.id: model}
        session.runtime = manifeste("robot-1", [model])
        session._prochaine_inference[model.id] = time.monotonic() + 60

        await session._analyse_stream(Flux())
        await session._release_all_models()

    asyncio.run(scenario())


def test_un_modele_est_decharge_apres_la_derniere_session(tmp_path):
    class Adaptateur:
        ferme = False

        def close(self):
            self.ferme = True

    async def scenario():
        adaptateur = Adaptateur()
        pool = WorkerPool(
            configuration(tmp_path),
            registry=RegistreFactice(),
            adapter_factory=lambda model, artifact: adaptateur,
        )
        model = modele()
        premier = await pool.acquire_model(model)
        second = await pool.acquire_model(model)
        assert premier is second and premier.references == 2

        await pool.release_model(premier)
        assert premier.key in pool.models and not adaptateur.ferme
        await pool.release_model(second)
        assert premier.key not in pool.models and adaptateur.ferme

    asyncio.run(scenario())


def test_perdre_un_bail_ne_ferme_que_la_session_concernee(tmp_path):
    creees = {}

    class SessionFactice:
        def __init__(self, pool, robot_id):
            self.robot_id = robot_id
            self.fermee = False
            self._fin = asyncio.Event()
            creees[robot_id] = self

        async def run(self):
            await self._fin.wait()

        def stop(self):
            self.fermee = True
            self._fin.set()

    async def scenario():
        pool = WorkerPool(
            configuration(tmp_path),
            registry=RegistreFactice(),
            session_factory=SessionFactice,
        )
        await pool.reconcile(["robot-1", "robot-2"])
        await asyncio.sleep(0)
        await pool.reconcile(["robot-2"])

        assert creees["robot-1"].fermee
        assert not creees["robot-2"].fermee
        assert set(pool.sessions) == {"robot-2"}
        await pool.reconcile([])

    asyncio.run(scenario())


def test_un_modele_invalide_nempeche_pas_les_autres_de_charger(tmp_path):
    bon = modele("bon", "b" * 64)
    mauvais = modele("mauvais", "c" * 64)
    registre = RegistreFactice({"robot-1": manifeste("robot-1", [mauvais, bon])})

    class Adaptateur:
        def infer(self, frame, manifest):
            return []

    def fabrique(model, artifact):
        if model.id == "mauvais":
            raise ValueError("artefact volontairement invalide")
        return Adaptateur()

    async def scenario():
        pool = WorkerPool(
            configuration(tmp_path), registry=registre, adapter_factory=fabrique
        )
        session = RobotSession(pool, "robot-1")
        await session.refresh_models()

        assert set(session.models) == {"bon"}
        assert len(pool.models) == 1
        await session._release_all_models()

    asyncio.run(scenario())
