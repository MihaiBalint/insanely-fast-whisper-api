from .diarization_pipeline import build_pipeline


def main():
    print("Loading weights")
    from .app import pipe, hf_token

    build_pipeline(hf_token)
    print(f"Loaded {pipe.type}")


if __name__ == "__main__":
    main()
