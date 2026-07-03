from types import SimpleNamespace

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2


class _FakeKVCache:
    def __init__(self, num_committed_tokens: int):
        self.num_committed_tokens = num_committed_tokens
        self.committed_tokens = None
        self.stopped_committing = False
        self.history_length = 8
        self.capacity = 8

    def commit(self, tokens, save_ssm_snapshot=False):
        self.committed_tokens = tokens
        self.num_committed_tokens += len(tokens)

    def stop_committing(self):
        self.stopped_committing = True


def test_try_commit_blocks_commits_uncommitted_tokens_and_stops_at_context_end():
    request = SimpleNamespace(
        py_request_id=1,
        is_dummy_request=False,
        is_dummy=False,
        context_current_position=8,
        context_remaining_length=0,
        prompt_len=8,
        get_tokens=lambda beam_id: list(range(10)),
        block_reuse_commit_limit=lambda: 8,
        should_save_ssm_snapshot=lambda commit_end: False,
    )
    kv_cache = _FakeKVCache(num_committed_tokens=4)
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = True
    manager.is_draft = False
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._block_reuse_committed_request_ids = set()
    manager._block_reuse_committed_request_count = 0
    manager._block_reuse_committed_token_count = 0
    manager._augment_tokens_for_block_reuse = lambda tokens, request, start, end: tokens[start:end]

    manager.try_commit_blocks(request)

    assert kv_cache.committed_tokens == [4, 5, 6, 7]
    assert kv_cache.num_committed_tokens == 8
    assert kv_cache.stopped_committing
