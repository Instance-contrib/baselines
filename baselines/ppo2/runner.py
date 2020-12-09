import numpy as np
import tensorflow as tf
from baselines.common.runners import AbstractEnvRunner

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam, tb_logger, after_epoch_cb, schedule_gamma, schedule_gamma_after, schedule_gamma_value, model_id=None):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.model_id = model_id
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.schedule_gamma = schedule_gamma
        self.schedule_gamma_after = schedule_gamma_after
        self.schedule_gamma_value = schedule_gamma_value
        self.tb_logger = tb_logger
        self.after_epoch_cb = after_epoch_cb

    def run(self, n_episode):
        if self.schedule_gamma and n_episode >= self.schedule_gamma_after:
            self.gamma = self.schedule_gamma_value

        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        self.obs[:] = self.env.reset()  # TODO instance01
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            obs = tf.constant(self.obs)
            actions, values, self.states, neglogpacs = self.model.step(obs)
            actions = actions._numpy()
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values._numpy())
            mb_neglogpacs.append(neglogpacs._numpy())
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions, model_id=self.model_id)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)

        if self.tb_logger:
            self.tb_logger.log_summary(self.env, mb_rewards, n_episode)
            self.after_epoch_cb(n_episode)

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(tf.constant(self.obs))._numpy()

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs.reshape(-1, 1) + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones.reshape(-1, 1), mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
