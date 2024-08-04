import pytest
import app.lib as lib


def test_process_dividing_batch_size_no_retries():
    def pipe(*args, batch_size=None, **kwargs):
        return {"status": "ok"}

    outputs = lib.process_dividing_batch_size(
        url="http://example.com",
        task="test",
        language=None,
        timestamp=None,
        batch_size=1,
        pipeline_function=pipe,
    )
    assert outputs == {"status": "ok"}


def test_process_dividing_batch_size_retry_until_failure():
    def pipe(*args, **kwargs):
        raise ValueError("CUDA out of memory. Tried to allocate 40.00 MiB.")

    with pytest.raises(ValueError):
        lib.process_dividing_batch_size(
            url="http://example.com",
            task="test",
            language=None,
            timestamp=None,
            batch_size=8,
            pipeline_function=pipe,
        )


def test_process_dividing_batch_size_retry_until_batch_size_one():
    def pipe(*args, batch_size=None, **kwargs):
        if batch_size > 1:
            raise ValueError("CUDA out of memory. Tried to allocate 40.00 MiB.")
        else:
            return {"status": "ok"}

    outputs = lib.process_dividing_batch_size(
        url="http://example.com",
        task="test",
        language=None,
        timestamp=None,
        batch_size=8,
        pipeline_function=pipe,
    )
    assert outputs == {"status": "ok"}
