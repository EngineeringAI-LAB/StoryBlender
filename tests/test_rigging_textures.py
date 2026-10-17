"""Rigging must use GLB material references, not texture names or ordering."""

import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo
from pygltflib import (GLTF2, Buffer, BufferView, Image, Texture, TextureInfo,
                       Material, PbrMetallicRoughness, NormalMaterialTexture)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from operators.animator_operators.rigging import _extract_textures, rig_model
from operators.animator_operators.restore_texture import restore_textures


def make_model(path, with_color=True):
    blob = bytearray()
    views = []
    for color in [(128, 128, 255), (0, 255, 0), (255, 0, 0)]:
        data = io.BytesIO()
        metadata = PngInfo()
        metadata.add_text('original', 'preserve texture bytes')
        PILImage.new('RGB', (2, 2), color).save(data, format='PNG', pnginfo=metadata)
        views.append(BufferView(buffer=0, byteOffset=len(blob), byteLength=len(data.getvalue())))
        blob.extend(data.getvalue())
        blob.extend(b'\0' * (-len(blob) % 4))
    model = GLTF2(
        buffers=[Buffer(byteLength=len(blob))], bufferViews=views,
        images=[Image(name=name, mimeType='image/png', bufferView=i)
                for i, name in enumerate(['normal', 'Image_0', 'Image_1'])],
        textures=[Texture(source=i) for i in [1, 2, 0]],
        materials=[Material(pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorTexture=TextureInfo(index=1) if with_color else None,
            metallicRoughnessTexture=TextureInfo(index=0)),
            normalTexture=NormalMaterialTexture(index=2))])
    model.set_binary_blob(bytes(blob))
    model.save_binary(str(path))


class RiggingTextureTests(unittest.TestCase):
    def test_meshy_receives_color_map_even_when_normal_image_is_first(self):
        for with_color in [True, False]:
            with self.subTest(with_color=with_color), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / 'character.glb'
                make_model(source, with_color)
                session = Mock()
                session.post.return_value.json.return_value = {'result': 'test-rig'}
                session.get.return_value.json.return_value = {'id': 'test-rig', 'status': 'SUCCEEDED'}
                with patch('operators.animator_operators.rigging.time.sleep'):
                    rig_model(str(source), meshy_api_key='dummy', session=session)
                payload = session.post.call_args.kwargs['json']
                if with_color:
                    encoded = payload['texture_image_url'].split(',', 1)[1]
                    image = PILImage.open(io.BytesIO(base64.b64decode(encoded)))
                    self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
                else:
                    self.assertNotIn('texture_image_url', payload)

    def test_material_manifest_restores_all_maps_and_replaces_wrong_cloud_color(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / 'source.glb'
            make_model(source)
            textures = folder / 'textures'
            _extract_textures(str(source), str(textures))
            original = GLTF2.load(str(source))
            view = original.bufferViews[2]
            self.assertEqual((textures / 'texture_2.png').read_bytes(),
                             original.binary_blob()[view.byteOffset:view.byteOffset + view.byteLength])
            manifest = json.loads((textures / 'texture_manifest.json').read_text())
            self.assertEqual(manifest, {'base_color': 'texture_2.png',
                                        'metallic_roughness': 'texture_1.png',
                                        'normal': 'texture_0.png'})
            # An old extraction folder must not override explicit material roles.
            PILImage.new('RGB', (2, 2), 'magenta').save(textures / 'old_color.png')
            cloud = GLTF2.load(str(source))
            cloud.materials[0].pbrMetallicRoughness.baseColorTexture.index = 2
            raw = folder / 'raw.glb'
            cloud.save_binary(str(raw))
            output = folder / 'restored.glb'
            restore_textures(str(raw), str(textures), str(output))
            restored = GLTF2.load(str(output))
            mat = restored.materials[0]
            for info, color in [(mat.pbrMetallicRoughness.baseColorTexture, (255, 0, 0)),
                                (mat.pbrMetallicRoughness.metallicRoughnessTexture, (0, 255, 0)),
                                (mat.normalTexture, (128, 128, 255))]:
                image = restored.images[restored.textures[info.index].source]
                decoded = PILImage.open(io.BytesIO(base64.b64decode(image.uri.split(',', 1)[1])))
                self.assertEqual(decoded.getpixel((0, 0)), color)


if __name__ == '__main__':
    unittest.main()
