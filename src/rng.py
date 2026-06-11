import numpy as np


P_SIGMA_SEED_OFFSET = 2_000_003


def create_rng(generator_name: str, seed: int):
    if generator_name == "pcg64":
        return np.random.default_rng(seed)

    if generator_name == "mt19937":
        bit_generator = np.random.MT19937(seed)
        return np.random.Generator(bit_generator)

    raise ValueError(f"Unknown generator: {generator_name}")


def create_p_sigma_rng(generator_name: str, seed: int):
    return create_rng(generator_name, seed + P_SIGMA_SEED_OFFSET)
