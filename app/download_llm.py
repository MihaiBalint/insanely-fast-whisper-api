from vllm import LLM, SamplingParams


def main():
    print("Loading language model weights")
    # llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1")
    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    outputs = llm.generate(["Hello dear!"], sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("Loaded language model weights")


if __name__ == "__main__":
    main()
