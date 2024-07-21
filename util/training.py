from collections import namedtuple
import os
import pickle
import numpy as np
from jax.tree_util import tree_map, tree_leaves, tree_flatten, tree_unflatten

TrainState = namedtuple(
    "TrainState",
    [
        "opt_statef",
        "model_statef",
        "paramsf",
        "params_emaf",
        "opt_stateb",
        "model_stateb",
        "paramsb",
        "params_emab",
        "ema_rate",
        "step",
        "rng",
    ],
)


def save(ckpt_dir: str, state) -> None:
    with open(os.path.join(ckpt_dir, "arrays.npy"), "wb") as f:
        for x in tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = tree_map(lambda t: 0, state)
    with open(os.path.join(ckpt_dir, "tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)


def restore(ckpt_dir):
    with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = tree_flatten(tree_struct)
    with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return tree_unflatten(treedef, flat_state)