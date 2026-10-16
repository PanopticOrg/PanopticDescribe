# PanopticDescribe

Plugin that describes the content of the selected images in natural language and stores it in a text property (`description` by default), to search, filter or group the images by their content (with PanopticText for instance).

## Usage

The **describe** function (execute button) runs in a background task on the selected images, or on the whole project when nothing is selected. Images that already have a description are skipped, unless `overwrite` is checked. The model is downloaded on first use.

## Engines

Choose the engine in the plugin params:

| Engine | Quality | Speed | Notes |
|---|---|---|---|
| `florence2` (default) | good, English only | ~0.1 s/image on GPU, 1 to 4 s/image on CPU | [Florence-2](https://huggingface.co/florence-community/Florence-2-base) (0.23B, MIT), ~0.5 GB downloaded |
| `vlm` | best, any language, follows a free prompt | ~1 s/image on GPU, GPU recommended | [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) (Apache 2.0), ~4.5 GB downloaded |

- `florence2`: `detail` sets the length of the description: `short` (one line), `detailed` (default, two or three sentences) or `very_detailed` (a paragraph). `florence-community/Florence-2-large` (0.77B) can be set in `florence_model`.
- `vlm`: the `prompt` param is sent to the model with each image, e.g. `Décris cette image en une phrase.` for French descriptions, or `List the objects visible in this image, separated by commas.` `vlm_model` accepts other Hugging Face image-text-to-text models (`Qwen/Qwen3.5-4B` for better quality, `Qwen/Qwen3.5-0.8B` is faster but makes more mistakes).

Images are read from the 1024 px copy stored by Panoptic (the original file is used if there is none) and resized to `max_image_size`.

Warning: describing many images can take a long time without a GPU.
