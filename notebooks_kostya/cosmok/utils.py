import numpy as np
from typing import Union, Optional


def make_random_state(
        seed: Optional[Union[int, np.random.Generator]],
) -> Union[np.random.Generator, np.random.RandomState]:
    if seed is None:
        return np.random
    if isinstance(seed, np.random.Generator) or isinstance(seed, np.random.RandomState):
        return seed
    return np.random.default_rng(seed=seed)


if __name__ == "__main__":
    ...
