from __future__ import annotations

import io
import logging
import os
from typing import TYPE_CHECKING

from PIL import Image, ImageOps

from panoptic.core.databases.data.models import DataCommit, Sha1Value
from panoptic.core.task.task import Task

from .engines import Engine

if TYPE_CHECKING:
    from .main import DescribePlugin

logger = logging.getLogger('PanopticDescribe')


class DescribeTask(Task):
    def __init__(self, plugin: DescribePlugin, sha1s: list[str], prop_id: int):
        super().__init__()
        self.plugin = plugin
        self.project = plugin.project
        self.sha1s = sha1s
        self.prop_id = prop_id
        self.name = 'Describe'

    def start(self) -> None:
        self.state.total = len(self.sha1s)
        self.set_step('Loading model', detail=Engine(self.plugin.params.engine).value)
        try:
            engine = self.plugin.get_engine()
        except Exception:
            logger.exception('Cannot load the description model')
            self.state.failed = len(self.sha1s)
            self.set_step('Cannot load the description model')
            return
        self.set_step('Describing images')

        for i in range(0, len(self.sha1s), engine.batch_size):
            if self.is_cancelled():
                break
            chunk = self.sha1s[i:i + engine.batch_size]
            images = {sha1: img for sha1 in chunk if (img := self._load_image(sha1)) is not None}
            try:
                texts = engine.describe(list(images.values())) if images else []
            except Exception:
                logger.exception('Description failed on %d images', len(images))
                texts = []
            values = [Sha1Value(property_id=self.prop_id, sha1=sha1, value=text)
                      for sha1, text in zip(images, texts) if text]
            if values:
                self.project.apply_commit(DataCommit(sha1_values=values))
            self.state.done += len(values)
            self.state.failed += len(chunk) - len(values)
            self._notify()

    def _load_image(self, sha1: str) -> Image.Image | None:
        # the stored 1024 px image is enough for a description and avoids reading large originals or IIIF downloads
        try:
            image = self._read_stored(sha1) or self._read_original(sha1)
            if image is None:
                logger.warning('No image found for %s', sha1)
                return None
            image = ImageOps.exif_transpose(image).convert('RGB')
            size = self.plugin.params.max_image_size
            if size > 0:
                image.thumbnail((size, size), Image.LANCZOS)
            return image
        except Exception:
            logger.exception('Cannot read image %s', sha1)
            return None

    def _read_stored(self, sha1: str) -> Image.Image | None:
        with self.project._media_db() as db:
            types = db.get_image_types()
            if not types:
                return None
            largest = max(types, key=lambda t: max(t.width or 0, t.height or 0))
            stored = db.get_images(type_id=largest.id, sha1=sha1)
        return Image.open(io.BytesIO(stored[0].data)) if stored else None

    def _read_original(self, sha1: str) -> Image.Image | None:
        # The plugin interface has no public image access yet, hence _data_reader().
        with self.project._data_reader() as r:
            ref = r.resolve_image_ref(sha1)
        if not ref:
            return None
        if ref['kind'] == 'local':
            return Image.open(ref['path']) if os.path.exists(ref['path']) else None
        if ref['kind'] == 'iiif':
            import requests
            res = requests.get(ref['url'], timeout=60)
            res.raise_for_status()
            return Image.open(io.BytesIO(res.content))
        return None
