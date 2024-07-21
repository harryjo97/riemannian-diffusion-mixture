import os
import hydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("int", lambda x: int(x), replace=True)
OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)
OmegaConf.register_new_resolver("str", lambda x: str(x), replace=True)
OmegaConf.register_new_resolver("prod", lambda x: np.prod(x), replace=True)
OmegaConf.register_new_resolver(
    "where", lambda condition, x, y: x if condition else y, replace=True
)
OmegaConf.register_new_resolver("isequal", lambda x, y: x == y, replace=True)
OmegaConf.register_new_resolver("pi", lambda x: x * math.pi, replace=True)
OmegaConf.register_new_resolver("min", min, replace=True)

@hydra.main(config_path="config", config_name="main")
def main(cfg):
    os.environ["GEOMSTATS_BACKEND"] = "jax"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    from run import run

    return run(cfg)


if __name__ == "__main__":
    main()
