def main():
    print("Loading weights")
    from .app import pipe
    print(f"Loaded {pipe.type}")

if __name__ == "__main__":
    main()
