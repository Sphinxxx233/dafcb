import random
from typing import List

_perm_pool: List[int] = []
_rng = random.Random()


def _get_perm_chunk(num_clients: int, clients_per_round: int) -> List[int]:
    """Return a chunk from a shuffled pool, refilling to cover all clients before repeat."""
    global _perm_pool

    # If not enough left for this round, top up with a fresh shuffled permutation
    if len(_perm_pool) < clients_per_round:
        refill = list(range(num_clients))
        _rng.shuffle(refill)
        _perm_pool.extend(refill)

    chunk = _perm_pool[:clients_per_round]
    _perm_pool = _perm_pool[clients_per_round:]
    return chunk


def select_clients(
    round_idx: int,
    num_clients: int,
    clients_per_round: int,
    warmup_rounds: int = 10,
) -> List[int]:
    """Select client indices for a given round.

    - 前 warmup_rounds 轮：随机但确保每个客户端至少被选中过一次（打乱后分块）。
    - 之后：独立随机抽样。
    """
    if round_idx <= warmup_rounds:
        return _get_perm_chunk(num_clients, clients_per_round)
    return random.sample(range(num_clients), clients_per_round)
