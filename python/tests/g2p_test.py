from hama import Phonemizer

def test_g2p_english():
    with Phonemizer() as phonemizer:
        result = phonemizer.to_ipa("Hello world")
        print(result)


