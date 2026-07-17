from voice_digester.config import DigestConfig


def test_loads_on_device_yaml():
    c = DigestConfig.from_yaml("sarvam-translate-4b.yaml")
    assert c.role == "on-device"
    assert c.model_name == "sarvamai/sarvam-translate"
    assert c.gguf_repo == "mradermacher/sarvam-translate-i1-GGUF"
    assert c.gguf_file.endswith("Q4_K_M.gguf")
    assert c.stt_model_name == "ai4bharat/indic-conformer-600m-multilingual"
    assert "{target_language}" in c.system_prompt
    assert "English" in c.rendered_system_prompt()


def test_loads_reference_yaml():
    c = DigestConfig.from_yaml("sarvam-30b-gguf.yaml")
    assert c.role == "reference"
    assert c.gguf_repo == "sarvamai/sarvam-30b-gguf"
