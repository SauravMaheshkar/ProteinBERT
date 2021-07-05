import jax
from jax import random

from proteinbert import ProteinBERT


def test_model():

    seq = jax.random.randint(
        key=random.PRNGKey(0), minval=0, maxval=21, shape=(2, 2048)
    )
    annotation = jax.random.randint(
        key=random.PRNGKey(0), minval=0, maxval=1, shape=(2, 8943)
    )

    init_rngs = {"params": random.PRNGKey(0), "layers": random.PRNGKey(1)}

    ProteinBERT().init(init_rngs, seq, annotation)
