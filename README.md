# OSCAR Perception Worker

Participant LiveKit indépendant qui transforme un flux vidéo robot en paquets
`oscar.vision.overlay.v1`. Il ne publie aucune commande et ne fait partie ni du
watchdog, ni de la chaîne ROS : son mode de panne est volontairement fail-open.

## Flux

1. Lecture du manifeste actif du robot auprès de l'API centrale.
2. Téléchargement vérifié par SHA-256 des artefacts activés.
3. Abonnement à la piste vidéo LiveKit de la room du robot.
4. Échantillonnage à la cadence définie par déploiement, 5 FPS par défaut.
5. Inférence via l'adaptateur du runtime.
6. Publication non fiable des boîtes normalisées sur `oscar.vision.overlay`.

Les détections sont triées par confiance et le paquet est borné à 1 200 octets,
sous la recommandation LiveKit de 1 300 octets pour éviter la fragmentation des
messages lossy. `detections_total` indique combien de résultats existaient avant
la réduction éventuelle.

Le worker recharge le manifeste toutes les 10 secondes. Un toggle depuis la
Sandbox ajoute ou retire donc un modèle sans redémarrer la vidéo ni le robot.

## Variables requises

```dotenv
OSCAR_CENTRAL_API_URL=https://admin.oscar-bot.com
OSCAR_PERCEPTION_WORKER_KEY=change-me
OSCAR_ROBOT_ID=<uuid-api-central>
OSCAR_MODEL_CACHE=/var/lib/oscar/models
OSCAR_MANIFEST_REFRESH_SECONDS=10
```

Au démarrage, le worker échange sa clé privée contre un jeton LiveKit court,
limité à l'abonnement vidéo et à la publication de data. La clé worker n'est
jamais envoyée au navigateur. `OSCAR_LIVEKIT_URL` et `OSCAR_LIVEKIT_TOKEN`
restent disponibles comme surcharge locale, à fournir ensemble.

## Formats

Le registre accepte `.pt`, `.onnx`, `.engine`, `.torchscript` et `.tflite`.
L'image actuelle exécute directement les modèles de détection Ultralytics `.pt`.
Les autres formats sont enregistrables pour portabilité, mais leur adaptateur de
sortie doit être ajouté et testé avant activation en production.

```bash
docker build -t oscar/perception-worker .
docker run --rm --env-file worker.env oscar/perception-worker
```
