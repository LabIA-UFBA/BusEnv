"""Lets MARLlib find the BusEnv ("sunt_bus") configuration file.

MARLlib looks for sunt_bus.yaml only inside its own installation folder. The team's
scripts use a local MARLlib copy that has the file; the shared `marllib` conda
environment does not. make_sunt_env() uses MARLlib's normal path when the file is there,
and otherwise does the same steps itself with the copy kept in this repository
(src/configs/marllib/sunt_bus.yaml, copied from /mnt/ssd1/wesley/BusEnv/MARLlib).
"""
from __future__ import annotations

import os

import yaml

REPO_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "marllib", "sunt_bus.yaml")


def make_sunt_env(environment_name: str = "sunt_bus", map_name: str = "sunt_bus", force_coop: bool = False,
                  yaml_path: str = REPO_YAML):
    from marllib import marl
    from marllib.envs.base_env import ENV_REGISTRY
    from marllib.marl.common import dict_update
    from ray.tune import register_env

    pkg_yaml = os.path.join(os.path.dirname(marl.__file__), "../envs/base_env/config/{}.yaml".format(environment_name))
    if os.path.exists(pkg_yaml):
        return marl.make_env(environment_name=environment_name, map_name=map_name, force_coop=force_coop)

    # Same steps as marl.make_env (marllib/marl/__init__.py), reading the repo copy of the yaml
    try:
        from marllib.marl import set_ray
    except ImportError:  # older MARLlib versions
        from marllib.marl.common import set_ray  # type: ignore

    with open(yaml_path) as f:
        env_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    env_config_dict["env_args"] = dict_update(env_config_dict.get("env_args", {}), {}, True)
    env_config_dict["env_args"]["map_name"] = map_name
    env_config_dict["force_coop"] = force_coop
    env_config = set_ray(env_config_dict)
    if force_coop:
        raise NotImplementedError("force_coop is not used by BusEnv")
    env_reg_name = env_config["env"] + "_" + env_config["env_args"]["map_name"]
    register_env(env_reg_name, lambda _: ENV_REGISTRY[env_config["env"]](env_config["env_args"]))
    env = ENV_REGISTRY[env_config["env"]](env_config["env_args"])
    return env, env_config
