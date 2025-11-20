from __future__ import annotations

import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from . import BLIPPlugin

from PIL import Image

from panoptic.core.task.task import Task
from panoptic.models import Instance, Property, ImageProperty, DbCommit

logger = logging.getLogger('PanopticBLIP:BLIPTask')


class ComputeBLIPTask(Task):
    def __init__(self, plugin: BLIPPlugin, instance: Instance, prop:Property):
        super().__init__()
        self.plugin = plugin
        self.project = plugin.project
        self.instance = instance
        self.name = f'Compute BLIP'
        self.prop = prop

    async def run(self):
        instance = self.instance
        image = self.preprocess_image(instance.url)
        text = await self.run_async(self.compute_label, image)
        commit = DbCommit()
        commit.image_values.extend([ImageProperty(property_id=self.prop.id, sha1=instance.sha1, value=text)])
        await self.project.do(commit)

    def compute_label(self, image: Image):
        inputs = self.plugin.processor(image, return_tensors="pt").to(self.plugin.device)

        out = self.plugin.model.generate(**inputs)
        text = self.plugin.processor.decode(out[0], skip_special_tokens=True)
        text.replace('arafed', '')

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image

