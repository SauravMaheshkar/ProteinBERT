from proteinbert import ProteinBERT
import jax
from jax import random


def test():

    seq = jax.random.randint(
        key=random.PRNGKey(0), minval=0, maxval=21, shape=(2, 2048)
    )
    annotation = jax.random.randint(
        key=random.PRNGKey(0), minval=0, maxval=1, shape=(2, 8943)
    )

    init_rngs = {"params": random.PRNGKey(0), "layers": random.PRNGKey(1)}

    ProteinBERT().init(init_rngs, seq, annotation)


if __name__ == "__main__":
    test()
