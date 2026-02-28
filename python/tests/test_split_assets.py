from importlib import resources
from pathlib import Path

from hama import G2PModel


def test_split_assets_load_and_predict():
    assets = resources.files("hama.assets")
    encoder_path = assets.joinpath("encoder.onnx")
    decoder_step_path = assets.joinpath("decoder_step.onnx")

    assert encoder_path.is_file(), f"Missing split asset: {encoder_path}"
    assert decoder_step_path.is_file(), f"Missing split asset: {decoder_step_path}"

    model = G2PModel(
        encoder_model_path=Path(str(encoder_path)),
        decoder_step_model_path=Path(str(decoder_step_path)),
    )
    result = model.predict("hello world")
    assert result.ipa
    assert result.alignments
