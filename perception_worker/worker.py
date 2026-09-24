from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import numpy as np
from livekit import rtc

from .adapters import ModelAdapter, create_adapter
from .config import WorkerConfig
from .exclusion import filtrer, lire_zones
from .focus import nombre_cible, selectionner
from .adapters.identification import appliquer_identites, recadrer

IDENTIFICATION = "product_identification"
from .registry import ModelRegistryClient
from .schemas import ModelManifest, OverlayPacket, RuntimeManifest

log = logging.getLogger("oscar.perception")


@dataclass
class LoadedModel:
    manifest: ModelManifest
    adapter: ModelAdapter
    next_inference_at: float = 0.0


class PerceptionWorker:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.registry = ModelRegistryClient(config)
        self.zones_exclusion = lire_zones(config.exclusion_zones)
        # Derniere selection « ciblee » par modele, pour la stabilite de la vue.
        self._focus_precedent: dict[str, list] = {}
        # Noms connus par modele detecteur : [(Detection, Identite | None, instant)].
        self._identites: dict[str, list] = {}
        self._identification_en_cours: set[str] = set()
        if self.zones_exclusion:
            log.info("Exclusion zones active: %s (overlap >= %.0f%%)",
                     self.zones_exclusion, config.exclusion_overlap * 100)
        self.room = rtc.Room()
        self.runtime: RuntimeManifest | None = None
        self.models: dict[str, LoadedModel] = {}
        self._stop = asyncio.Event()
        self._track_task: asyncio.Task | None = None
        self._track_sid: str | None = None

    async def run(self) -> None:
        await self.refresh_models()
        if self.config.livekit_url and self.config.livekit_token:
            livekit_url = self.config.livekit_url
            livekit_token = self.config.livekit_token
        else:
            session = await self.registry.session()
            if self.runtime and session.room != self.runtime.room:
                raise RuntimeError("La session LiveKit et le manifeste ciblent des rooms différentes")
            # L'URL locale prime sur celle annoncée par l'API, le jeton restant
            # émis par elle. Quand le worker tourne à côté du SFU, le nom public
            # est le mauvais chemin pour le média : la signalisation aboutit,
            # le participant apparaît actif, et l'abonnement aux pistes n'arrive
            # jamais. Mesuré : sur l'URL interne l'abonnement est immédiat.
            livekit_url = self.config.livekit_url or session.livekit_url
            livekit_token = session.token
        self._wire_events()
        await self.room.connect(livekit_url, livekit_token)
        await self.room.local_participant.set_metadata(json.dumps({
            "role": "oscar-perception-worker",
            "robot_id": self.config.robot_id,
            "overlay_topic": self.runtime.overlay_topic if self.runtime else "oscar.vision.overlay",
        }, separators=(",", ":")))
        refresh_task = asyncio.create_task(self._refresh_loop())
        try:
            await self._stop.wait()
        finally:
            refresh_task.cancel()
            if self._track_task:
                self._track_task.cancel()
            await self.registry.close()
            await self.room.disconnect()

    def _wire_events(self) -> None:
        @self.room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):  # noqa: ANN001
            if track.kind != rtc.TrackKind.KIND_VIDEO:
                return
            # La piste la plus recente l'emporte. Une nouvelle piste video dans
            # la room d'un robot est presque toujours la meme camera republiee
            # apres une reconnexion : l'ancienne tache attend alors sur un flux
            # mort dont l'iterateur ne se termine jamais. Ignorer la nouvelle
            # piste, comme avant, laissait le worker connecte, abonne, et
            # n'analysant plus rien — sans une ligne d'erreur.
            if self._track_task and not self._track_task.done():
                log.info("Switching analysis to video track %s from %s (previous stream released)",
                         publication.name, participant.identity)
                self._track_task.cancel()
            else:
                log.info("Analyzing video track %s from %s", publication.name, participant.identity)
            self._track_sid = publication.sid
            self._track_task = asyncio.create_task(self._consume_video(track))

        @self.room.on("track_unsubscribed")
        def on_track_unsubscribed(track, publication, participant):  # noqa: ANN001
            if publication.sid != self._track_sid:
                return
            log.info("Video track %s from %s unsubscribed; waiting for the next one",
                     publication.name, participant.identity)
            if self._track_task and not self._track_task.done():
                self._track_task.cancel()
            self._track_sid = None

        @self.room.on("disconnected")
        def on_disconnected(reason):  # noqa: ANN001
            log.warning("LiveKit disconnected: %s", reason)
            self._stop.set()

    async def refresh_models(self) -> None:
        manifest = await self.registry.manifest()
        next_models: dict[str, LoadedModel] = {}
        for model in manifest.models:
            current = self.models.get(model.id)
            if current and current.manifest.sha256 == model.sha256:
                current.manifest = model
                next_models[model.id] = current
                continue
            try:
                artifact = await self.registry.artifact(model)
                adapter = await asyncio.to_thread(create_adapter, model, artifact)
                next_models[model.id] = LoadedModel(model, adapter)
                log.info("Loaded model %s v%s (%s)", model.name, model.version, model.runtime)
            except Exception:
                log.exception("Model %s could not be loaded; media and control remain unaffected", model.id)
        self.runtime = manifest
        self.models = next_models

    async def _refresh_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self.config.manifest_refresh_seconds)
            try:
                await self.refresh_models()
            except Exception:
                log.exception("Manifest refresh failed; keeping currently loaded models")

    async def _consume_video(self, track) -> None:  # noqa: ANN001
        stream = rtc.VideoStream(track)
        try:
            await self._analyse_stream(stream)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Video analysis stopped unexpectedly; waiting for the next track")
        finally:
            try:
                await stream.aclose()
            except Exception:  # noqa: BLE001 - un flux deja mort peut refuser la fermeture
                pass

    async def _analyse_stream(self, stream) -> None:  # noqa: ANN001
        async for event in stream:
            if not self.models or not self.runtime:
                continue
            # La piste arrive a ~30 images/s, l'inference est cadencee a 5 : la
            # grande majorite des trames n'a aucune raison d'etre convertie.
            # Sans ce test, chaque trame etait transcodee en RGB24 puis copiee
            # dans un tableau numpy pour rien, et la file interne debordait
            # (« native video stream queue overflow »).
            maintenant = time.monotonic()
            if all(charge.next_inference_at > maintenant for charge in self.models.values()):
                continue
            frame = event.frame.convert(rtc.VideoBufferType.RGB24)
            rgb = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
            await self._infer_frame(rgb, frame.width, frame.height, int(time.time_ns() / 1000))

    async def _infer_frame(self, frame: np.ndarray, width: int, height: int, timestamp_us: int) -> None:
        now = time.monotonic()
        dus = []
        for loaded in list(self.models.values()):
            if loaded.manifest.task == IDENTIFICATION:
                continue  # second etage : il travaille sur les boites, pas sur l'image
            if now < loaded.next_inference_at:
                continue
            loaded.next_inference_at = now + 1.0 / loaded.manifest.inference_fps
            dus.append(loaded)
        # Les modeles d'une Box tournent en parallele. En sequence, le retard
        # de l'overlay sur l'image affichee valait la somme des inferences :
        # la boite arrivait quand la camera avait deja tourne. torch libere le
        # GIL pendant l'inference, les threads se repartissent donc les coeurs.
        await asyncio.gather(*(self._infer_one(loaded, frame, width, height, timestamp_us)
                               for loaded in dus))

    async def _infer_one(self, loaded: LoadedModel, frame: np.ndarray, width: int, height: int,
                         timestamp_us: int) -> None:
        try:
            detections = await asyncio.to_thread(loaded.adapter.infer, frame)
            detections = filtrer(detections, self.zones_exclusion, self.config.exclusion_overlap)
            cible = nombre_cible(loaded.manifest.config)
            if cible:
                detections = selectionner(detections, cible, self._focus_precedent.get(loaded.manifest.id))
                self._focus_precedent[loaded.manifest.id] = detections
            detections = self._nommer(loaded.manifest.id, detections, frame)
            if not loaded.manifest.overlay_enabled:
                return
            packet = OverlayPacket(
                robot_id=self.config.robot_id,
                room=self.runtime.room,
                frame_timestamp_us=timestamp_us,
                frame_width=width,
                frame_height=height,
                model_id=loaded.manifest.id,
                model_name=loaded.manifest.name,
                model_version=loaded.manifest.version,
                task=loaded.manifest.task,
                detections=detections,
            )
            payload, fiable = packet.fit_wire()
            await self.room.local_participant.publish_data(
                payload, reliable=fiable, topic=self.runtime.overlay_topic,
            )
        except Exception:
            log.exception("Inference failed for model %s", loaded.manifest.id)

    def _nommer(self, detecteur_id: str, detections: list, frame: np.ndarray) -> list:
        """Applique les noms connus et relance une identification si elle est due."""
        identifieur = next((m for m in self.models.values() if m.manifest.task == IDENTIFICATION), None)
        if identifieur is None or not detections:
            return detections
        maintenant = time.monotonic()
        nommees = appliquer_identites(detections, self._identites.get(detecteur_id), maintenant)
        if maintenant >= identifieur.next_inference_at and detecteur_id not in self._identification_en_cours:
            identifieur.next_inference_at = maintenant + 1.0 / identifieur.manifest.inference_fps
            # Recadrages copies tout de suite : le tampon de l'image est reutilise
            # par le flux video des l'image suivante.
            hauteur_min = getattr(identifieur.adapter, "hauteur_min", 0)
            lot = [(d, recadrer(frame, d.x, d.y, d.width, d.height, hauteur_min=hauteur_min))
                   for d in detections]
            self._identification_en_cours.add(detecteur_id)
            asyncio.create_task(self._identifier(detecteur_id, identifieur, lot))
        return nommees

    async def _identifier(self, detecteur_id: str, identifieur: LoadedModel, lot: list) -> None:
        try:
            valides = [(d, r) for d, r in lot if r is not None]
            identites = await asyncio.to_thread(identifieur.adapter.identifier, [r for _, r in valides])
            instant = time.monotonic()
            self._identites[detecteur_id] = [(d, i, instant) for (d, _), i in zip(valides, identites)]
        except Exception:
            log.exception("Identification failed for detector %s", detecteur_id)
        finally:
            self._identification_en_cours.discard(detecteur_id)

