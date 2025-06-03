import os
os.environ["MUJOCO_GL"] = "osmesa"

import json
import random
import wandb
import numpy as np
from absl import app, flags
from flax.training import checkpoints
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax

from utils.env_utils import make_env_and_datasets
from utils.log_utils import get_exp_name, setup_wandb

FLAGS = flags.FLAGS

# ----------------------
# CLI flags (adjusted)
# ----------------------
flags.DEFINE_string("run_group", "ContrastiveCRL", "Run group name.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("env_name", "puzzle-3x3-noisy-v0", "Environment name.")
flags.DEFINE_string("save_dir", "./exp_contrastive", "Directory to save models and logs.")

flags.DEFINE_integer("train_steps", 200000, "Number of training steps.")         # increased from 5000
flags.DEFINE_integer("batch_size", 128, "Batch size.")
flags.DEFINE_integer("horizon", 10, "Maximum number of steps for positive pairs.")
flags.DEFINE_integer("embed_dim", 10, "Embedding dimension.")
flags.DEFINE_integer("proj_hidden", 128, "Projection-head hidden dimension.")
flags.DEFINE_integer("warmup_steps", 2000, "Number of warmup steps for LR schedule.")
flags.DEFINE_float("lr", 5e-4, "Peak learning rate.")
flags.DEFINE_float("end_lr_factor", 0.05, "Final LR = lr * end_lr_factor.")
flags.DEFINE_float("temperature", 0.5, "Contrastive temperature.")
flags.DEFINE_float("weight_decay", 1e-4, "Weight decay for regularization.")
flags.DEFINE_float("dropout_rate", 0.1, "Dropout rate in encoder.")
flags.DEFINE_float("noise_scale", 0.05, "Scale of noise for synthetic positive pairs.")
flags.DEFINE_float("pos_pair_ratio", 0.5, "Ratio of positive pairs (true positives only).")
flags.DEFINE_integer("log_interval", 500, "Logging interval (in steps).")

# ----------------------
# Encoder network
# ----------------------
class Encoder(nn.Module):
    hidden_dims: tuple = (256, 256)
    proj_hidden: int = 128
    embed_dim: int = 128
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training=True):
        # Backbone MLP: Dense -> ReLU -> LayerNorm -> Dropout
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x)
            if training:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Raw representation (could be used for downstream tasks)
        raw = nn.Dense(self.embed_dim)(x)

        # Projection head: Dense -> ReLU -> Dropout -> Dense
        z = nn.Dense(self.proj_hidden)(raw)
        z = nn.relu(z)
        if training:
            z = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(z)
        z = nn.Dense(self.embed_dim)(z)

        # L2-normalize the projection
        z = z / (jnp.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        return raw, z

# ----------------------
# BCE-style contrastive loss
# ----------------------
def bce_contrastive_loss(s0_proj, sg_pos_proj, sg_neg_proj):
    """
    BCE-style contrastive loss:
    - s0_proj: (B, D) anchor embeddings
    - sg_pos_proj: (B, D) positive embeddings
    - sg_neg_proj: (B, D) negative embeddings
    """
    pos_scores = jnp.sum(s0_proj * sg_pos_proj, axis=1)  # (B,)
    neg_scores = jnp.sum(s0_proj * sg_neg_proj, axis=1)  # (B,)
    pos_loss = jax.nn.log_sigmoid(pos_scores)            # log σ(f+)
    neg_loss = jax.nn.log_sigmoid(-neg_scores)           # log(1 - σ(f-)) = log σ(-f-)
    loss = -jnp.mean(pos_loss + neg_loss)
    metrics = {
        'avg_pos_score': jnp.mean(pos_scores),
        'avg_neg_score': jnp.mean(neg_scores),
        'pos_loss': -jnp.mean(pos_loss),
        'neg_loss': -jnp.mean(neg_loss),
    }
    return loss, metrics

# ----------------------
# Split flat OGbench arrays into per-episode lists (handles both compact and non-compact)
# ----------------------
def split_into_episodes(train_data):
    """
    Given train_data from OGbench, split into a list of episodes.
    Supports:
      - Non-compact format: train_data contains 'observations', 'next_observations', 'terminals'.
      - Compact format: train_data contains 'observations', 'terminals', 'valids'.
    Returns:
      episodes: a Python list of numpy arrays of shape (ep_len, obs_dim) each.
    """
    episodes = []

    # Non-compact: we have explicit next_observations and terminals
    if 'next_observations' in train_data:
        obs_flat       = train_data['observations']       # shape: (N, obs_dim)
        terminals_flat = train_data['terminals']          # shape: (N,)
        N = obs_flat.shape[0]
        start_idx = 0

        for i in range(N):
            if terminals_flat[i] == 1:
                # Episode runs from start_idx .. i (inclusive)
                ep_obs = obs_flat[start_idx : i + 1]  # (ep_len+1, obs_dim)
                episodes.append(np.asarray(ep_obs))
                start_idx = i + 1

    # Compact: we have only observations, terminals, and valids
    else:
        obs_flat       = train_data['observations']  # shape: (N, obs_dim)
        terminals_flat = train_data['terminals']     # shape: (N,)
        valids_flat    = train_data['valids']        # shape: (N,)
        N = obs_flat.shape[0]

        current_episode = []
        for i in range(N):
            current_episode.append(obs_flat[i])
            # If valids[i] == 0, no valid next_obs for i → episode ends here
            if valids_flat[i] == 0:
                ep_array = np.stack(current_episode, axis=0)  # (ep_len, obs_dim)
                episodes.append(ep_array)
                current_episode = []

        # If any leftover (though typically valids ensures segmentation), append it
        if len(current_episode) > 0:
            ep_array = np.stack(current_episode, axis=0)
            episodes.append(ep_array)

    return episodes

# ----------------------
# Sample a contrastive batch from episodes (no synthetic positives)
# ----------------------
def sample_bce_contrastive_batch(
    episodes, batch_size, H, obs_dim,
    dissim_threshold=1e-2,
    max_resample_attempts=10
):
    """
    For each anchor, sample a positive (within H steps in same episode) and a negative (from another episode).
    Returns: s0_array, sg_pos_array, sg_neg_array
    """
    s0_list, sg_pos_list, sg_neg_list = [], [], []
    num_episodes = len(episodes)

    for _ in range(batch_size):
        # Sample anchor episode and anchor state
        for _attempt in range(max_resample_attempts):
            ep_idx = np.random.randint(num_episodes)
            seq = episodes[ep_idx]
            traj_len = seq.shape[0]
            if traj_len < H + 2:
                continue
            t0 = np.random.randint(0, traj_len - H - 1)
            s0 = seq[t0]
            # Sample positive from same episode within H steps
            t1 = t0 + np.random.randint(1, H + 1)
            sg_pos = seq[t1]
            t2 = t0 + np.random.randint(H + 1, traj_len - t0)
            sg_neg = seq[t2]
            # Sample negative from a different episode
            # for _neg_attempt in range(max_resample_attempts):
            #     neg_ep_idx = np.random.randint(num_episodes)
            #     if neg_ep_idx == ep_idx:
            #         continue
            #     neg_seq = episodes[neg_ep_idx]
            #     if neg_seq.shape[0] == 0:
            #         continue
            #     t2 = np.random.randint(0, neg_seq.shape[0])
            #     sg_neg = neg_seq[t2]
            #     if np.linalg.norm(s0 - sg_neg) > dissim_threshold:
            #         break
            # else:
            #     continue  # Retry anchor if no good negative found
            # Append
            s0_list.append(s0)
            sg_pos_list.append(sg_pos)
            sg_neg_list.append(sg_neg)
            break

    s0_array = jnp.array(s0_list).reshape(batch_size, obs_dim)
    sg_pos_array = jnp.array(sg_pos_list).reshape(batch_size, obs_dim)
    sg_neg_array = jnp.array(sg_neg_list).reshape(batch_size, obs_dim)
    return s0_array, sg_pos_array, sg_neg_array

# ----------------------
# Main training loop
# ----------------------
def main(_):
    # 1) Logging and random seeds
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project="OGBench", group=FLAGS.run_group, name=exp_name)

    save_path = os.path.abspath(
        os.path.join(FLAGS.save_dir, FLAGS.env_name, FLAGS.run_group, exp_name)
    )
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving checkpoints & logs to: {save_path}")
    assert os.path.isabs(save_path), "Checkpoint directory must be absolute."

    with open(os.path.join(save_path, "flags.json"), "w") as f:
        json.dump(FLAGS.flag_values_dict(), f, indent=2)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # 2) Load OGbench dataset
    _, train_data, _ = make_env_and_datasets(
        FLAGS.env_name, frame_stack=1
    )

    # 3) Split into per-episode lists (handles both compact and non-compact)
    episodes = split_into_episodes(train_data)
    assert len(episodes) > 0, "No episodes found in the dataset!"

    # 4) Determine obs_dim from first episode
    obs_dim = episodes[0].shape[1]
    print(f"Found {len(episodes)} episodes. obs_dim = {obs_dim}")

    # 5) Initialize two encoders (initial-state and goal-state)
    key_init, key_goal = jax.random.split(jax.random.PRNGKey(FLAGS.seed))
    encoder_init = Encoder(
        hidden_dims=(256, 256),
        proj_hidden=FLAGS.proj_hidden,
        embed_dim=FLAGS.embed_dim,
        dropout_rate=FLAGS.dropout_rate
    )
    encoder_goal = Encoder(
        hidden_dims=(256, 256),
        proj_hidden=FLAGS.proj_hidden,
        embed_dim=FLAGS.embed_dim,
        dropout_rate=FLAGS.dropout_rate
    )

    dummy_input = jnp.ones((2, obs_dim))
    params_init = encoder_init.init(
        {"params": key_init, "dropout": jax.random.PRNGKey(0)},
        dummy_input,
        training=True
    )['params']
    params_goal = encoder_goal.init(
        {"params": key_goal, "dropout": jax.random.PRNGKey(1)},
        dummy_input,
        training=True
    )['params']

    print(f"Encoder networks initialized (obs_dim={obs_dim}, embed_dim={FLAGS.embed_dim}).")

    # 6) Learning rate schedule and optimizer
    total_decay = FLAGS.train_steps - FLAGS.warmup_steps
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=FLAGS.lr,
        warmup_steps=FLAGS.warmup_steps,
        decay_steps=total_decay,
        end_value=FLAGS.lr * FLAGS.end_lr_factor,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_schedule, weight_decay=FLAGS.weight_decay)
    )
    opt_state_init = optimizer.init(params_init)
    opt_state_goal = optimizer.init(params_goal)

    # 7) JIT-compiled training step
    @jax.jit
    def train_step(params_init, params_goal, s0_batch, sg_pos_batch, sg_neg_batch, opt_state_init, opt_state_goal, rng):
        rng1, rng2 = jax.random.split(rng, 2)

        def loss_fn(pi, pg):
            _, z0 = encoder_init.apply(
                {'params': pi}, s0_batch, training=True, rngs={'dropout': rng1}
            )
            _, zg_pos = encoder_goal.apply(
                {'params': pg}, sg_pos_batch, training=True, rngs={'dropout': rng2}
            )
            _, zg_neg = encoder_goal.apply(
                {'params': pg}, sg_neg_batch, training=True, rngs={'dropout': rng2}
            )
            loss, _ = bce_contrastive_loss(z0, zg_pos, zg_neg)
            return loss

        # Compute gradients using only the loss
        (loss_val), (grads_init, grads_goal) = jax.value_and_grad(loss_fn, argnums=(0, 1))(params_init, params_goal)
        
        # Compute metrics separately
        _, z0 = encoder_init.apply({'params': params_init}, s0_batch, training=False)
        _, zg_pos = encoder_goal.apply({'params': params_goal}, sg_pos_batch, training=False)
        _, zg_neg = encoder_goal.apply({'params': params_goal}, sg_neg_batch, training=False)
        _, metrics = bce_contrastive_loss(z0, zg_pos, zg_neg)
        
        updates_init, new_opt_state_init = optimizer.update(grads_init, opt_state_init, params_init)
        updates_goal, new_opt_state_goal = optimizer.update(grads_goal, opt_state_goal, params_goal)
        new_params_init = optax.apply_updates(params_init, updates_init)
        new_params_goal = optax.apply_updates(params_goal, updates_goal)
        return loss_val, new_params_init, new_params_goal, new_opt_state_init, new_opt_state_goal, metrics

    # 8) Training loop
    rng = jax.random.PRNGKey(FLAGS.seed)
    metrics = {
        'loss': [], 'steps': [],
        'avg_pos_score': [], 'avg_neg_score': [],
        'pos_loss': [], 'neg_loss': []
    }

    for step in range(1, FLAGS.train_steps + 1):
        rng, step_rng = jax.random.split(rng)

        # Sample a contrastive batch using episodes, plus track ep/timesteps
        s0_batch, sg_pos_batch, sg_neg_batch = sample_bce_contrastive_batch(
            episodes, FLAGS.batch_size, FLAGS.horizon, obs_dim
        )

        loss, params_init, params_goal, opt_state_init, opt_state_goal, metrics_step = train_step(
            params_init, params_goal,
            s0_batch, sg_pos_batch, sg_neg_batch,
            opt_state_init, opt_state_goal,
            step_rng
        )

        # Logging at intervals
        if step % FLAGS.log_interval == 0 or step == 1:
            metrics['loss'].append(loss.item())
            metrics['steps'].append(step)
            metrics['avg_pos_score'].append(metrics_step['avg_pos_score'].item())
            metrics['avg_neg_score'].append(metrics_step['avg_neg_score'].item())
            metrics['pos_loss'].append(metrics_step['pos_loss'].item())
            metrics['neg_loss'].append(metrics_step['neg_loss'].item())

            # Log additional debug metrics
            wandb.log({
                'loss': loss.item(),
                'avg_pos_score': metrics_step['avg_pos_score'].item(),
                'avg_neg_score': metrics_step['avg_neg_score'].item(),
                'pos_loss': metrics_step['pos_loss'].item(),
                'neg_loss': metrics_step['neg_loss'].item(),
                'learning_rate': lr_schedule(step),
            }, step=step)

            print(f"[Step {step:06d}] loss={loss:.4f}   "
                  f"avg_pos_score={metrics_step['avg_pos_score'].item():.3f}  avg_neg_score={metrics_step['avg_neg_score'].item():.3f}  "
                  f"pos_loss={metrics_step['pos_loss'].item():.4f}  neg_loss={metrics_step['neg_loss'].item():.4f}  lr={lr_schedule(step):.6f}")


    # 9) Save final checkpoint
    checkpoints.save_checkpoint(
        ckpt_dir=save_path,
        target={'encoder_init': params_init, 'encoder_goal': params_goal},
        step=FLAGS.train_steps,
        prefix="contrastive_",
        overwrite=True
    )
    print(f"Training complete. Encoders saved to {save_path}")


if __name__ == "__main__":
    app.run(main)