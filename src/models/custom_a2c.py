# custom_algos/custom_a2c.py
import gym
from typing import Optional, Dict

import ray
from ray.rllib.agents.ppo.ppo_torch_policy import ValueNetworkMixin
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch, Postprocessing
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import apply_grad_clipping, sequence_mask
from ray.rllib.utils.typing import TrainerConfigDict, TensorType, PolicyID, LocalOptimizer

from ray.rllib.agents.a3c.a2c import A2CTrainer, A2C_DEFAULT_CONFIG
from ray.rllib.agents.callbacks import DefaultCallbacks

torch, nn = try_import_torch()

# ======================
# 1. Policy Definition
# ======================
def actor_critic_loss(policy: Policy, model: ModelV2,
                      dist_class: ActionDistribution,
                      train_batch: SampleBatch) -> TensorType:
    logits, _ = model.from_batch(train_batch)
    values = model.value_function()

    if policy.is_recurrent():
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask_orig = sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
        valid_mask = torch.reshape(mask_orig, [-1])
    else:
        valid_mask = torch.ones_like(values, dtype=torch.bool)

    dist = dist_class(logits, model)
    log_probs = dist.logp(train_batch[SampleBatch.ACTIONS]).reshape(-1)

    pi_err = -torch.sum(
        torch.masked_select(log_probs * train_batch[Postprocessing.ADVANTAGES], valid_mask)
    )

    if policy.config["use_critic"]:
        value_err = 0.5 * torch.sum(
            torch.pow(
                torch.masked_select(
                    values.reshape(-1) - train_batch[Postprocessing.VALUE_TARGETS], valid_mask
                ),
                2.0,
            )
        )
    else:
        value_err = 0.0

    entropy = torch.sum(torch.masked_select(dist.entropy(), valid_mask))

    total_loss = (
        pi_err + value_err * policy.config["vf_loss_coeff"] - entropy * policy.config["entropy_coeff"]
    )

    model.tower_stats["entropy"] = entropy
    model.tower_stats["pi_err"] = pi_err
    model.tower_stats["value_err"] = value_err

    return total_loss


def loss_and_entropy_stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    return {
        "policy_entropy": torch.mean(torch.stack(policy.get_tower_stats("entropy"))),
        "policy_loss": torch.mean(torch.stack(policy.get_tower_stats("pi_err"))),
        "vf_loss": torch.mean(torch.stack(policy.get_tower_stats("value_err"))),
    }


def model_value_predictions(
    policy: Policy, input_dict: Dict[str, TensorType], state_batches, model: ModelV2,
    action_dist: ActionDistribution
) -> Dict[str, TensorType]:
    return {SampleBatch.VF_PREDS: model.value_function()}


def torch_optimizer(policy: Policy, config: TrainerConfigDict) -> LocalOptimizer:
    return torch.optim.Adam(policy.model.parameters(), lr=config["lr"])


def setup_mixins(policy: Policy, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, config: TrainerConfigDict) -> None:
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)


# create the custom A2C policy
CustomA2CPolicy = build_policy_class(
    name="CustomA2CPolicy",
    framework="torch",
    get_default_config=lambda: A2C_DEFAULT_CONFIG,
    loss_fn=actor_critic_loss,
    stats_fn=loss_and_entropy_stats,
    postprocess_fn=compute_gae_for_sample_batch,
    extra_action_out_fn=model_value_predictions,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=torch_optimizer,
    before_loss_init=setup_mixins,
    mixins=[ValueNetworkMixin],
)

# ======================
# 2. Trainer Definition
# ======================
CustomA2CPolicyTorch = CustomA2CPolicy.with_updates(
    name="CustomA2CPolicyTorch",
    get_default_config=lambda: A2C_DEFAULT_CONFIG,
)

def get_policy_class(config_):
    if config_["framework"] == "torch":
        return CustomA2CPolicyTorch

# ======================
# 3. Custom Callbacks para logar métricas por agente
# ======================

class CustomCallbacks(DefaultCallbacks):
    def on_train_result(self, *, trainer=None, algorithm=None, result: dict, **kwargs):
        algo = trainer if trainer is not None else algorithm

        # Exemplo: log rewards médios por política
        if "hist_stats" in result:
            for policy_id, rewards in result["hist_stats"].items():
                if isinstance(rewards, list) and rewards:
                    mean_r = sum(rewards) / len(rewards)
                    result.setdefault("custom_metrics", {})[f"{policy_id}_reward_mean"] = mean_r



# ======================
# 4. Trainer com callbacks embutido
# ======================
CustomA2CTrainer = A2CTrainer.with_updates(
    name="CustomA2CTrainer",
    default_policy=None,
    get_policy_class=get_policy_class,
    default_config={**A2C_DEFAULT_CONFIG, "callbacks": CustomCallbacks},
)