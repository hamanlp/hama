from hama import G2PModel


def main() -> None:
    model = G2PModel()
    result = model.predict("안녕하세요")
    print("IPA:", result.ipa)
    print("Alignments:", result.alignments)


if __name__ == "__main__":
    main()
