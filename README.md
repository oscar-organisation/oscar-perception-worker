# OSCAR Perception Worker

Pool de participants LiveKit qui transforme les flux vidéo des robots en
paquets `oscar.vision.overlay.v1`. Il ne publie aucune commande et ne fait
partie ni du watchdog, ni de la chaîne ROS : son mode de panne est
volontairement fail-open.

## Flux

1. Annonce de la capacité du worker et acquisition de baux de 45 secondes.
2. Ouverture d'une session LiveKit indépendante pour chaque robot attribué.
3. Lecture de son manifeste actif auprès de l'API centrale.
4. Téléchargement vérifié par SHA-256 des artefacts activés.
5. Échantillonnage à la cadence définie par déploiement, 5 FPS par défaut.
6. Inférence via un adaptateur mutualisé entre les robots utilisant le même artefact.
7. Publication des boîtes normalisées sur `oscar.vision.overlay`.

Les détections sont triées par confiance et le paquet est borné à 1 200 octets,
sous la recommandation LiveKit de 1 300 octets pour éviter la fragmentation des
messages lossy. `detections_total` indique combien de résultats existaient avant
la réduction éventuelle.

Chaque session recharge son manifeste toutes les 10 secondes. Un toggle depuis
la Sandbox ajoute ou retire donc un modèle sans redémarrer la vidéo ni le
robot. Le heartbeat de 15 secondes renouvelle les baux et fait converger les
sessions. La perte d'un bail ne ferme que le robot concerné.

Les modèles sont chargés une fois par processus et comptés par références. La
cadence, le focus et l'identification restent propres à chaque robot. Les
appels à une même instance de modèle sont sérialisés ; les modèles distincts
d'une Box continuent de tourner en parallèle.

## Variables requises

```dotenv
OSCAR_CENTRAL_API_URL=https://admin.oscar-bot.com
OSCAR_PERCEPTION_WORKER_KEY=change-me
OSCAR_WORKER_CAPACITY=4
OSCAR_MODEL_CACHE=/var/lib/oscar/models
OSCAR_MANIFEST_REFRESH_SECONDS=10
```

`OSCAR_WORKER_ID` est facultatif : le nom d'hôte unique du conteneur est utilisé
par défaut. Au démarrage de chaque session robot, le worker échange sa clé
privée contre un jeton LiveKit court, limité à l'abonnement vidéo et à la
publication de data. La clé worker n'est jamais envoyée au navigateur.
`OSCAR_LIVEKIT_URL` peut surcharger l'URL publique pour joindre directement le
SFU depuis son réseau Docker ; le jeton reste toujours émis par l'API pour le
robot concerné.

`OSCAR_EXCLUSION_ZONES_BY_ROBOT` accepte un objet JSON associant un identifiant
robot à ses zones d'exclusion. Il prime sur `OSCAR_EXCLUSION_ZONES`, conservée
comme valeur de repli pour les installations homogènes.

## Formats

Le registre accepte `.pt`, `.onnx`, `.engine`, `.torchscript` et `.tflite`.
L'image actuelle exécute directement les modèles de détection Ultralytics `.pt`.
Les autres formats sont enregistrables pour portabilité, mais leur adaptateur de
sortie doit être ajouté et testé avant activation en production.

```bash
docker compose up --build --detach
```

Compose utilise le projet `oscar-perception`, lance deux répliques et leur fait
partager le volume de cache `perception-models`.
