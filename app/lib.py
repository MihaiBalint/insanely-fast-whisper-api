import time


def process_dividing_batch_size(
    url: str,
    task: str,
    language: str,
    batch_size: int,
    timestamp: str,
    pipeline_function,
):
    if not batch_size or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive, non-zero integer, received `{batch_size}`")

    generate_kwargs = {
        "task": task,
        "language": None if language == "None" else language,
    }
    current_batch_size = batch_size
    batch_size_one_attempt_no = 0
    batch_size_one_max_retries = 5
    while batch_size_one_attempt_no <= batch_size_one_max_retries:
        try:
            outputs = pipeline_function(
                url,
                chunk_length_s=30,
                batch_size=current_batch_size,
                generate_kwargs=generate_kwargs,
                return_timestamps="word" if timestamp == "word" else True,
            )
            if batch_size != current_batch_size:
                print(
                    f"Transcript generated successfully with batch_size={current_batch_size} requested_batch_size={batch_size}"
                )
            return outputs
        except Exception as e:
            errorMessage = str(e)
            if "out of memory" not in errorMessage.lower():
                raise e

            if current_batch_size > 1:
                print(f"Reducing batch size following CUDA OOM at batch_size={current_batch_size}")
                current_batch_size = current_batch_size // 2
                continue
            elif batch_size_one_attempt_no < batch_size_one_max_retries:
                current_batch_size = 1
                batch_size_one_attempt_no += 1
                delay = 2.0 * batch_size_one_attempt_no
                print(
                    f"Retrying following CUDA OOM at batch_size={current_batch_size} attempts={batch_size_one_attempt_no} max_attempts={batch_size_one_max_retries} delay={delay:0.1f}s"
                )
                time.sleep(delay)
            else:
                print(
                    f"Giving up following CUDA OOM at batch_size={current_batch_size} attempts={batch_size_one_attempt_no}"
                )
                raise e
