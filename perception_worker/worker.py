from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from livekit import rtc

from .adapters import ModelAdapter, create_adapter
from .adapters.identification import appliquer_identites, recadrer
from .config import WorkerConfig
from .exclusion import filtrer, lire_zones
from .focus import nombre_cible, selectionner
from .registry import ModelRegistryClient
from .schemas import ModelManifest, OverlayPacket, RuntimeManifest

IDENTIFICATION = "product_identification"

log = logging.getLogger("oscar.perception")


@dataclass
class LoadedModel:
    """Adaptateur lourd partagé entre toutes les sessions qui utilisent l'artefact."""

    key: tuple[str, str]
    manifest: ModelManifest
    adapter: ModelAdapter
    references: int = 0
    # Les objets Ultralytics et TorchScript ne garantissent pas que deux appels
    # entrelacés sur la même instance soient sûrs. Les modèles distincts restent
    # parallèles, mais un même modèle est sérialisé entre les robots.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class WorkerPool:
    """Répartit les baux et mutualise les modèles entre plusieurs robots."""

    def __init__(
        self,
        config: WorkerConfig,
        registry: ModelRegistryClient | None = None,
        adapter_factory: Callable[[ModelManifest, Path], ModelAdapter] = create_adapter,
        session_factory: Callable[["WorkerPool", str], "RobotSession"] | None = None,
    ):
        self.config = config
        self.registry = registry or ModelRegistryClient(config)
        self.models: dict[tuple[str, str], LoadedModel] = {}
        self.sessions: dict[str, RobotSession] = {}
        self._session_tasks: dict[str, asyncio.Task] = {}
        self._adapter_factory = adapter_factory
        self._session_factory = session_factory or RobotSession
        self._models_lock = asyncio.Lock()
        self._stop = asyncio.Event()

    async def run(self) -> None:
        attente = 0
        try:
            while not self._stop.is_set():
                if attente:
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=attente)
                        break
                    except asyncio.TimeoutError:
                        pass
                try:
                    baux = await self.registry.leases()
                    await self.reconcile(baux.robots)
                    attente = baux.renouveler_dans
                except Exception:
                    # Une panne temporaire du registre ne coupe pas les flux
                    # déjà actifs. Le serveur fera foi au prochain heartbeat.
                    log.exception("Lease heartbeat failed; active robot sessions are kept")
                    attente = 5
        finally:
            await self._fermer_toutes_les_sessions()
            try:
                await self.registry.release_leases()
            except Exception:
                log.exception("Worker leases could not be released cleanly")
            await self.registry.close()

    def stop(self) -> None:
        self._stop.set()

    async def reconcile(self, robot_ids: list[str]) -> None:
        """Fait converger les sessions vers la liste exacte rendue par l'API."""
        attendus = set(robot_ids)
        for robot_id in set(self.sessions) - attendus:
            await self._fermer_session(robot_id)

        # Une déconnexion LiveKit ne rend pas le bail. Si la tâche s'est
        # terminée mais que l'API maintient l'attribution, on recrée uniquement
        # cette session au heartbeat suivant.
        for robot_id in attendus:
            task = self._session_tasks.get(robot_id)
            if task is not None and task.done():
                await self._fermer_session(robot_id)
            if robot_id not in self.sessions:
                self._demarrer_session(robot_id)

    async def acquire_model(self, manifest: ModelManifest) -> LoadedModel:
        key = (manifest.id, manifest.sha256)
        async with self._models_lock:
            current = self.models.get(key)
            if current is None:
                artifact = await self.registry.artifact(manifest)
                adapter = await asyncio.to_thread(self._adapter_factory, manifest, artifact)
                current = LoadedModel(key=key, manifest=manifest, adapter=adapter)
                self.models[key] = current
                log.info("Loaded shared model %s v%s (%s)",
                         manifest.name, manifest.version, manifest.runtime)
            current.references += 1
            return current

    async def release_model(self, loaded: LoadedModel) -> None:
        async with self._models_lock:
            current = self.models.get(loaded.key)
            if current is not loaded:
                return
            current.references = max(0, current.references - 1)
            if current.references:
                return
            async with current.lock:
                self.models.pop(current.key, None)
                close = getattr(current.adapter, "close", None)
                if close:
                    resultat = close()
                    if inspect.isawaitable(resultat):
                        await resultat
            log.info("Unloaded shared model %s v%s",
                     current.manifest.name, current.manifest.version)

    def _demarrer_session(self, robot_id: str) -> None:
        session = self._session_factory(self, robot_id)
        self.sessions[robot_id] = session
        task = asyncio.create_task(session.run(), name=f"perception-{robot_id}")
        self._session_tasks[robot_id] = task
        task.add_done_callback(lambda fini, rid=robot_id: self._journaliser_fin(rid, fini))

    async def _fermer_session(self, robot_id: str) -> None:
        session = self.sessions.pop(robot_id, None)
        task = self._session_tasks.pop(robot_id, None)
        if session is not None:
            session.stop()
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=10)
            except asyncio.TimeoutError:
                log.warning("Robot session %s did not stop in time; cancelling it", robot_id)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            except Exception:
                # La fin anormale a déjà été journalisée par le callback.
                pass

    async def _fermer_toutes_les_sessions(self) -> None:
        await asyncio.gather(
            *(self._fermer_session(robot_id) for robot_id in list(self.sessions)),
            return_exceptions=True,
        )

    @staticmethod
    def _journaliser_fin(robot_id: str, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        erreur = task.exception()
        if erreur is not None:
            log.error("Robot session %s stopped unexpectedly", robot_id,
                      exc_info=(type(erreur), erreur, erreur.__traceback__))


class RobotSession:
    """Connexion LiveKit et état d'inférence propres à un robot."""

    def __init__(self, pool: WorkerPool, robot_id: str):
        self.pool = pool
        self.config = pool.config
        self.registry = pool.registry
        self.robot_id = robot_id
        self.zones_exclusion = lire_zones(self.config.exclusion_zones)
        self._focus_precedent: dict[str, list] = {}
        self._identites: dict[str, list] = {}
        self._identification_en_cours: set[str] = set()
        self._identification_tasks: set[asyncio.Task] = set()
        self._prochaine_inference: dict[str, float] = {}
        self.room = rtc.Room()
        self.runtime: RuntimeManifest | None = None
        self.models: dict[str, LoadedModel] = {}
        self.model_manifests: dict[str, ModelManifest] = {}
        self._stop = asyncio.Event()
        self._track_task: asyncio.Task | None = None
        self._track_sid: str | None = None
        if self.zones_exclusion:
            log.info("Exclusion zones active for %s: %s (overlap >= %.0f%%)",
                     robot_id, self.zones_exclusion, self.config.exclusion_overlap * 100)

    async def run(self) -> None:
        refresh_task: asyncio.Task | None = None
        try:
            await self.refresh_models()
            session = await self.registry.session(self.robot_id)
            if self.runtime and session.room != self.runtime.room:
                raise RuntimeError("La session LiveKit et le manifeste ciblent des rooms différentes")
            # L'URL locale prime sur celle annoncée par l'API, le jeton restant
            # émis pour ce robot précis par l'API centrale.
            livekit_url = self.config.livekit_url or session.livekit_url
            self._wire_events()
            await self.room.connect(livekit_url, session.token)
            await self.room.local_participant.set_metadata(json.dumps({
                "role": "oscar-perception-worker",
                "worker_id": self.config.worker_id,
                "robot_id": self.robot_id,
                "overlay_topic": self.runtime.overlay_topic if self.runtime else "oscar.vision.overlay",
            }, separators=(",", ":")))
            refresh_task = asyncio.create_task(self._refresh_loop())
            await self._stop.wait()
        finally:
            if refresh_task:
                refresh_task.cancel()
            if self._track_task:
                self._track_task.cancel()
            for task in self._identification_tasks:
                task.cancel()
            await asyncio.gather(
                *(task for task in [refresh_task, self._track_task,
                                    *self._identification_tasks] if task is not None),
                return_exceptions=True,
            )
            await self._release_all_models()
            try:
                await self.room.disconnect()
            except Exception:
                log.debug("Room %s was already disconnected", self.robot_id, exc_info=True)

    def stop(self) -> None:
        self._stop.set()

    def _wire_events(self) -> None:
        @self.room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):  # noqa: ANN001
            if track.kind != rtc.TrackKind.KIND_VIDEO:
                return
            if self._track_task and not self._track_task.done():
                log.info("Switching %s analysis to video track %s from %s",
                         self.robot_id, publication.name, participant.identity)
                self._track_task.cancel()
            else:
                log.info("Analyzing %s video track %s from %s",
                         self.robot_id, publication.name, participant.identity)
            self._track_sid = publication.sid
            self._track_task = asyncio.create_task(self._consume_video(track))

        @self.room.on("track_unsubscribed")
        def on_track_unsubscribed(track, publication, participant):  # noqa: ANN001
            if publication.sid != self._track_sid:
                return
            log.info("Video track %s from %s unsubscribed for %s",
                     publication.name, participant.identity, self.robot_id)
            if self._track_task and not self._track_task.done():
                self._track_task.cancel()
            self._track_sid = None

        @self.room.on("disconnected")
        def on_disconnected(reason):  # noqa: ANN001
            log.warning("LiveKit disconnected for %s: %s", self.robot_id, reason)
            self._stop.set()

    async def refresh_models(self) -> None:
        manifest = await self.registry.manifest(self.robot_id)
        if manifest.robot_id != self.robot_id:
            raise RuntimeError("Le manifeste retourné ne correspond pas au robot demandé")

        suivants: dict[str, LoadedModel] = {}
        manifests_suivants: dict[str, ModelManifest] = {}
        for model in manifest.models:
            current = self.models.get(model.id)
            if current and current.key == (model.id, model.sha256):
                suivants[model.id] = current
                manifests_suivants[model.id] = model
                continue
            try:
                suivants[model.id] = await self.pool.acquire_model(model)
                manifests_suivants[model.id] = model
            except Exception:
                log.exception("Model %s could not be loaded for %s; other models remain active",
                              model.id, self.robot_id)
                # Une nouvelle version défectueuse ne retire pas la version qui
                # fonctionnait déjà sur ce robot.
                if current is not None:
                    suivants[model.id] = current
                    manifests_suivants[model.id] = self.model_manifests[model.id]

        for model_id, loaded in list(self.models.items()):
            if suivants.get(model_id) is not loaded:
                await self.pool.release_model(loaded)
        self.runtime = manifest
        self.models = suivants
        self.model_manifests = manifests_suivants
        self._prochaine_inference = {
            model_id: self._prochaine_inference.get(model_id, 0.0)
            for model_id in suivants
        }

    async def _release_all_models(self) -> None:
        charges = list(self.models.values())
        self.models = {}
        self.model_manifests = {}
        self._prochaine_inference = {}
        await asyncio.gather(*(self.pool.release_model(model) for model in charges),
                             return_exceptions=True)

    async def _refresh_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self.config.manifest_refresh_seconds)
            try:
                await self.refresh_models()
            except Exception:
                log.exception("Manifest refresh failed for %s; loaded models are kept",
                              self.robot_id)

    async def _consume_video(self, track) -> None:  # noqa: ANN001
        stream = rtc.VideoStream(track)
        try:
            await self._analyse_stream(stream)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Video analysis stopped unexpectedly for %s", self.robot_id)
        finally:
            try:
                await stream.aclose()
            except Exception:
                pass

    async def _analyse_stream(self, stream) -> None:  # noqa: ANN001
        async for event in stream:
            if not self.models or not self.runtime:
                continue
            # La grande majorité des trames ne doit pas être convertie : chaque
            # session consulte ici sa propre horloge d'inférence.
            maintenant = time.monotonic()
            if all(self._prochaine_inference.get(model_id, 0.0) > maintenant
                   for model_id in self.models):
                continue
            frame = event.frame.convert(rtc.VideoBufferType.RGB24)
            rgb = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
            await self._infer_frame(rgb, frame.width, frame.height, int(time.time_ns() / 1000))

    async def _infer_frame(self, frame: np.ndarray, width: int, height: int,
                           timestamp_us: int) -> None:
        maintenant = time.monotonic()
        dus: list[tuple[LoadedModel, ModelManifest]] = []
        for model_id, loaded in list(self.models.items()):
            manifest = self.model_manifests[model_id]
            if manifest.task == IDENTIFICATION:
                continue
            if maintenant < self._prochaine_inference.get(model_id, 0.0):
                continue
            self._prochaine_inference[model_id] = maintenant + 1.0 / manifest.inference_fps
            dus.append((loaded, manifest))
        # Les modèles distincts d'une Box tournent en parallèle. Le verrou du
        # LoadedModel ne sérialise que deux robots utilisant la même instance.
        await asyncio.gather(*(
            self._infer_one(loaded, manifest, frame, width, height, timestamp_us)
            for loaded, manifest in dus
        ))

    async def _infer_one(self, loaded: LoadedModel, manifest: ModelManifest,
                         frame: np.ndarray, width: int, height: int,
                         timestamp_us: int) -> None:
        try:
            async with loaded.lock:
                detections = await asyncio.to_thread(loaded.adapter.infer, frame, manifest)
            detections = filtrer(detections, self.zones_exclusion, self.config.exclusion_overlap)
            cible = nombre_cible(manifest.config)
            if cible:
                detections = selectionner(
                    detections, cible, self._focus_precedent.get(manifest.id)
                )
                self._focus_precedent[manifest.id] = detections
            detections = self._nommer(manifest.id, detections, frame)
            if not manifest.overlay_enabled or self.runtime is None:
                return
            packet = OverlayPacket(
                robot_id=self.robot_id,
                room=self.runtime.room,
                frame_timestamp_us=timestamp_us,
                frame_width=width,
                frame_height=height,
                model_id=manifest.id,
                model_name=manifest.name,
                model_version=manifest.version,
                task=manifest.task,
                detections=detections,
            )
            payload, fiable = packet.fit_wire()
            await self.room.local_participant.publish_data(
                payload, reliable=fiable, topic=self.runtime.overlay_topic,
            )
        except Exception:
            log.exception("Inference failed for model %s on %s", manifest.id, self.robot_id)

    def _nommer(self, detecteur_id: str, detections: list, frame: np.ndarray) -> list:
        """Applique les noms connus et relance une identification si elle est due."""
        paire = next((
            (self.models[model_id], manifest)
            for model_id, manifest in self.model_manifests.items()
            if manifest.task == IDENTIFICATION
        ), None)
        if paire is None or not detections:
            return detections
        identifieur, manifest = paire
        maintenant = time.monotonic()
        nommees = appliquer_identites(detections, self._identites.get(detecteur_id), maintenant)
        prochaine = self._prochaine_inference.get(manifest.id, 0.0)
        if maintenant >= prochaine and detecteur_id not in self._identification_en_cours:
            self._prochaine_inference[manifest.id] = maintenant + 1.0 / manifest.inference_fps
            reglage = getattr(identifieur.adapter, "hauteur_min", 0)
            hauteur_min = reglage(manifest) if callable(reglage) else reglage
            lot = [
                (d, recadrer(frame, d.x, d.y, d.width, d.height,
                             hauteur_min=hauteur_min))
                for d in detections
            ]
            self._identification_en_cours.add(detecteur_id)
            task = asyncio.create_task(
                self._identifier(detecteur_id, identifieur, manifest, lot)
            )
            self._identification_tasks.add(task)
            task.add_done_callback(self._identification_tasks.discard)
        return nommees

    async def _identifier(self, detecteur_id: str, identifieur: LoadedModel,
                          manifest: ModelManifest, lot: list) -> None:
        try:
            valides = [(d, r) for d, r in lot if r is not None]
            async with identifieur.lock:
                identites = await asyncio.to_thread(
                    identifieur.adapter.identifier,
                    [r for _, r in valides],
                    manifest,
                )
            instant = time.monotonic()
            self._identites[detecteur_id] = [
                (d, identite, instant)
                for (d, _), identite in zip(valides, identites)
            ]
        except Exception:
            log.exception("Identification failed for detector %s on %s",
                          detecteur_id, self.robot_id)
        finally:
            self._identification_en_cours.discard(detecteur_id)


# Compatibilité du nom public avec les intégrations qui importaient la classe.
PerceptionWorker = RobotSession
