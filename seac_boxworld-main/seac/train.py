import glob
import os
import shutil
import time
from collections import deque
from dataclasses import dataclass
from os import path
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import utils
from a2c import A2C, AlgoConfig
from envs import make_vec_envs
from model import DRRLConfig
from wrappers import RecordEpisodeStatistics, SquashDones


@dataclass
class RunConfig:
    env_name: Optional[str] = None
    time_limit: Optional[int] = None
    wrappers: List = (
        RecordEpisodeStatistics,
        SquashDones,
    )
    use_dummy_vecenv: bool = False

    num_env_steps: int = int(100e6)

    eval_dir: str = "./results/video/{id}"
    loss_dir: str = "./results/loss/{id}"
    save_dir: str = "./results/trained_models/{id}"

    log_interval: int = 2000
    save_interval: int = int(1e6)
    eval_interval: int = int(1e6)
    episodes_per_eval: int = 8
    evaluation_steps: int = 8


configs = {}
for conf in glob.glob("configs/*.yaml"):
    name = f"{Path(conf).stem}"
    configs[name] = conf


def _squash_info(info):
    info = [i for i in info if i]
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    keys.discard("TimeLimit.truncated")
    for key in keys:
        mean = np.mean([np.array(d[key]).sum() for d in info if key in d])
        new_info[key] = mean
    return new_info


def evaluate(
        agents,
        monitor_dir,
        seed,
        _log,
        run_config,
        algo_config,
):
    eval_envs = make_vec_envs(
        env_name=run_config.env_name,
        seed=seed,
        dummy_vecenv=run_config.use_dummy_vecenv,
        parallel=run_config.episodes_per_eval,
        time_limit=run_config.time_limit,
        wrappers=run_config.wrappers,
        device=algo_config.device,
        monitor_dir=monitor_dir,
    )

    n_obs = eval_envs.reset()
    n_recurrent_hidden_states = [
        torch.zeros(
            run_config.episodes_per_eval, agent.model.recurrent_hidden_state_size, device=algo_config.device
        )
        for agent in agents
    ]
    masks = torch.zeros(run_config.episodes_per_eval, 1, device=algo_config.device)

    all_infos = []

    while len(all_infos) < run_config.episodes_per_eval:
        with torch.no_grad():
            _, n_action, _, n_recurrent_hidden_states = zip(
                *[
                    agent.model.act(n_obs[agent.agent_id], recurrent_hidden_states, masks)
                    for agent, recurrent_hidden_states in zip(agents, n_recurrent_hidden_states)
                ]
            )

        # Observe reward and next obs
        n_obs, _, done, infos = eval_envs.step(n_action)

        n_masks = torch.tensor(
            [[0] if d else [1] for d in done],
            dtype=torch.float32,
            device=algo_config.device,
        )
        all_infos.extend([i for i in infos if i])

    eval_envs.close()
    info = _squash_info(all_infos)
    _log.info(
        f"Evaluation using {len(all_infos)} episodes: "
        f"mean reward {info['episode_reward']:.5f} "
        f"mean episode length {info['episode_length']:.5f}\n"
    )


def run(
        run_id: int,
        _log,
        seed: int,
        run_config: RunConfig,
        algo_config: AlgoConfig,
        net_config: DRRLConfig,
):
    _log.info("starting")
    if run_config.loss_dir:
        loss_dir = path.expanduser(run_config.loss_dir.format(id=str(run_id)))
        utils.cleanup_log_dir(loss_dir)
        writer = SummaryWriter(loss_dir)
    else:
        writer = None

    eval_dir = path.expanduser(run_config.eval_dir.format(id=str(run_id)))
    save_dir = path.expanduser(run_config.save_dir.format(id=str(run_id)))

    utils.cleanup_log_dir(eval_dir)
    utils.cleanup_log_dir(save_dir)

    torch.set_num_threads(1)
    envs = make_vec_envs(
        env_name=run_config.env_name,
        seed=seed,
        dummy_vecenv=run_config.use_dummy_vecenv,
        parallel=algo_config.num_processes,
        time_limit=run_config.time_limit,
        wrappers=run_config.wrappers,
        device=algo_config.device,
    )

    agents = [
        A2C(i, obs_space, act_space, algo_config, net_config)
        for i, (obs_space, act_space) in enumerate(zip(envs.observation_space, envs.action_space))
    ]
    obs = envs.reset()

    for i in range(len(obs)):
        agents[i].storage.obs[0].copy_(obs[i])
        agents[i].storage.to(algo_config.device)

    start = time.time()
    num_updates = (run_config.num_env_steps // algo_config.num_steps // algo_config.num_processes)

    all_infos = deque(maxlen=10)

    for j in range(1, num_updates + 1):

        for step in range(algo_config.num_steps):
            # Sample actions
            with torch.no_grad():
                n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
                    *[
                        agent.model.act(
                            agent.storage.obs[step],
                            agent.storage.recurrent_hidden_states[step],
                            agent.storage.masks[step],
                        )
                        for agent in agents
                    ]
                )
            # Obs reward and next obs
            obs, reward, done, infos = envs.step(n_action)
            # envs.envs[0].render()

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0] if d else [1] for d in done])

            bad_masks = torch.FloatTensor(
                [[0] if info.get("TimeLimit.truncated", False) else [1] for info in infos]
            )
            for i in range(len(agents)):
                agents[i].storage.insert(
                    obs[i],
                    n_recurrent_hidden_states[i],
                    n_action[i],
                    n_action_log_prob[i],
                    n_value[i],
                    reward[:, i].unsqueeze(1),
                    masks,
                    bad_masks,
                )

            for info in infos:
                if info:
                    all_infos.append(info)

        # value_loss, action_loss, dist_entropy = agent.update(rollouts)
        for agent in agents:
            agent.compute_returns(**algo_config.__dict__)

        for agent in agents:
            loss = agent.update([a.storage for a in agents], **algo_config.__dict__)
            for k, v in loss.items():
                if writer:
                    writer.add_scalar(f"agent{agent.agent_id}/{k}", v, j)

        for agent in agents:
            agent.storage.after_update()

        if j % run_config.log_interval == 0 and len(all_infos) > 1:
            squashed = _squash_info(all_infos)

            total_num_steps = ((j + 1) * algo_config.num_processes * algo_config.num_steps)
            end = time.time()
            fps = int(total_num_steps / (end - start))
            _log.info(
                f"Updates {j}, num timesteps {total_num_steps}, FPS {fps}"
            )
            mean_reward = squashed['episode_reward'].sum()
            mean_episode_length = squashed['episode_length'].sum()
            _log.info(
                f"Last {len(all_infos)} training episodes mean reward {mean_reward:.3f} "
                f"mean episode length {mean_episode_length:.3f}"
            )

            # for k, v in squashed.items():
            #     _run.log_scalar(k, v, j)
            all_infos.clear()

        if run_config.save_interval is not None and (
                j > 0 and j % run_config.save_interval == 0 or j == num_updates
        ):
            cur_save_dir = path.join(save_dir, f"u{j}")
            for agent in agents:
                save_at = path.join(cur_save_dir, f"agent{agent.agent_id}")
                os.makedirs(save_at, exist_ok=True)
                agent.save(save_at)
            archive_name = shutil.make_archive(cur_save_dir, "xztar", save_dir, f"u{j}")
            shutil.rmtree(cur_save_dir)
            # _run.add_artifact(archive_name)

        if run_config.eval_interval is not None and (
                j > 0 and j % run_config.eval_interval == 0 or j == num_updates
        ):
            evaluate(
                agents=agents,
                monitor_dir=os.path.join(eval_dir, f"u{j}"),
                seed=seed,
                _log=_log,
                run_config=run_config,
                algo_config=algo_config
            )
            videos = glob.glob(os.path.join(eval_dir, f"u{j}") + "/*.mp4")
            # for i, v in enumerate(videos):
            # _run.add_artifact(v, f"u{j}.{i}.mp4")
    envs.close()
