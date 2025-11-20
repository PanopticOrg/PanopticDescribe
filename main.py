import torch
from panoptic.core.plugin.plugin import APlugin
from panoptic.models import ActionContext, PropertyType, PropertyMode, DbCommit, Instance, ImageProperty, Property
from panoptic.models.results import ActionResult, Notif, NotifType
from panoptic.core.plugin.plugin_project_interface import PluginProjectInterface

from .task import ComputeBLIPTask


class BLIPPlugin(APlugin):
    def __init__(self, project: PluginProjectInterface, plugin_path: str, name: str):
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        self.add_action_easy(self.labelize, ['execute'])
        self.model = None
        self.processor = None
        self.device = resolve_device()

    async def labelize(self, context: ActionContext):
        commit = DbCommit()

        prop = await self.project.get_or_create_property('blip_description', PropertyType.string, PropertyMode.sha1)
        commit.properties.append(prop)
        res = await self.project.do(commit)
        real_prop = res.properties[0]
        instances = await self.project.get_instances(ids=context.instance_ids)
        unique_sha1 = list({i.sha1: i for i in instances}.values())

        self.model, self.processor = await self._init_model()

        [await self.labelize_task(i, real_prop) for i in unique_sha1]

        notif = Notif(
            type=NotifType.INFO,
            name='compute_labels',
            message=f'BLIP Labelize started on {len(unique_sha1)} images'
        )
        return ActionResult(notifs=[notif])

    async def labelize_task(self, instance, prop):
        task = ComputeBLIPTask(self, instance, prop)
        self.project.add_task(task)

    async def _init_model(self):
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)
        return model, processor


def resolve_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # TODO: when silicon bugs are working again put it back
    # elif torch.backends.mps.is_available():
    #     device = 'mps'
    return device