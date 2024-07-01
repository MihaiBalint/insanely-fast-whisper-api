def diarization(hf_token):
    from .diarization_pipeline import build_pipeline

    build_pipeline(hf_token)


def main():
    print("Loading weights")
    from .app import pipe, hf_token

    print(f"Loaded {pipe.type}")


if __name__ == "__main__":
    main()
