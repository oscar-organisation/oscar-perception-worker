import asyncio
import logging

from .config import WorkerConfig
from .worker import PerceptionWorker


async def _executer() -> None:
    """Construit le worker DANS la boucle, puis le lance.

    `rtc.Room.__init__` fait `self._loop = loop or asyncio.get_event_loop()` :
    la room capture la boucle presente a sa creation. Instancier le worker avant
    `asyncio.run` l'attachait donc a une boucle qui ne tournerait jamais. Les
    appels awaites aboutissaient — connexion, metadonnees, rafraichissement du
    manifeste — mais les evenements remontes par le FFI etaient postes sur la
    boucle orpheline : `track_subscribed` ne se declenchait pas, aucune image
    n'etait analysee, et rien ne le signalait dans les journaux.
    """
    await PerceptionWorker(WorkerConfig.from_env()).run()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(_executer())


if __name__ == "__main__":
    main()
