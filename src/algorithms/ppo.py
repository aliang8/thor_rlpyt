import torch

from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo
from rlpyt.agents.base import AgentInputs, AgentInputsRnn
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs

from rlpyt.algos.pg.ppo import PPO

OptInfoCustom = namedarraytuple("OptInfoCustom", ["loss", "loss_b", "loss_p", "gradNorm", "entropy_b", "entropy_p", "perplexity_b", "perplexity_p"])

LossInputs = namedarraytuple("LossInputs",
    ["agent_inputs", "action", "return_", "advantage", "valid", "old_dist_info_b", "old_dist_info_p"])


class PPO_Custom(PPO):
    def __init__(
            self,
            discount=0.99,
            learning_rate=0.001,
            value_loss_coeff=1.,
            entropy_loss_coeff=0.01,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=1.,
            initial_optim_state_dict=None,
            gae_lambda=1,
            minibatches=4,
            epochs=4,
            ratio_clip=0.1,
            linear_lr_schedule=True,
            normalize_advantage=False,
            ):
        """Saves input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())

        self.opt_info_fields = tuple(f for f in OptInfoCustom._fields)  # copy

    def optimize_agent(self, itr, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)
        return_, advantage, valid = self.process_returns(samples)
        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_,
            advantage=advantage,
            valid=valid,
            old_dist_info_b=samples.agent.agent_info.dist_info_b,
            old_dist_info_p=samples.agent.agent_info.dist_info_p
        )
        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.
        T, B = samples.env.reward.shape[:2]
        opt_info = OptInfoCustom(*([] for _ in range(len(OptInfoCustom._fields))))
        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches
        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T

                xs = idxs if recurrent else idxs // T
                self.optimizer.zero_grad()
                rnn_state = init_rnn_state[B_idxs] if recurrent else None
                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                loss, loss_b, loss_p, entropy_b, entropy_p, perplexity_b, perplexity_p = self.loss(
                    *loss_inputs[T_idxs, B_idxs], rnn_state)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.clip_grad_norm)
                # print(grad_norm.item())
                self.optimizer.step()

                opt_info.loss.append(loss.item())
                opt_info.loss_b.append(loss_b.item())
                opt_info.loss_p.append(loss_p.item())

                opt_info.gradNorm.append(torch.tensor(grad_norm).item())  # backwards compatible
                opt_info.entropy_b.append(entropy_b.item())
                opt_info.entropy_p.append(entropy_p.item())

                opt_info.perplexity_b.append(perplexity_b.item())
                opt_info.perplexity_p.append(perplexity_p.item())

                self.update_counter += 1
        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr
        return opt_info

    def loss(self, agent_inputs, action, return_, advantage, valid, old_dist_info_b, old_dist_info_p, init_rnn_state=None):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info_b, dist_info_p, value, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info_b, dist_info_p, value = self.agent(*agent_inputs)

        dist_b, dist_p = self.agent.categorical_dist, self.agent.beta_dist

        # Base action policy loss
        if self.agent.recurrent:
            assert(len(action.shape) == 3)
            base = action[:,:,0].long()
            pointer = action[:,:,-2:]
        else:
            assert(len(action.shape) == 2)
            base = action[:,0].squeeze(-1).long()
            pointer = action[:,-2:]

        ratio = dist_b.likelihood_ratio(base, old_dist_info=old_dist_info_b, new_dist_info=dist_info_b)

        if len(ratio.shape) - len(advantage.shape) == 1: # TODO: this feels hacky ...
            advantage = advantage.unsqueeze(-1)

        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
            1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss_b = - valid_mean(surrogate, valid)

        # Pointer action policy loss
        ratio = dist_p.likelihood_ratio(pointer, old_dist_info=old_dist_info_p, new_dist_info=dist_info_p)

        if len(ratio.shape) - len(advantage.shape) == 1: # TODO: this feels hacky ...
            advantage = advantage.unsqueeze(-1)

        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
            1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss_p = - valid_mean(surrogate, valid)

        value_error = 0.5 * (value - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        # Base action entropy
        entropy_b = dist_b.mean_entropy(dist_info_b, valid)
        entropy_loss_b = -self.entropy_loss_coeff * entropy_b

        # Pointer action entropy
        entropy_p = dist_p.mean_entropy(dist_info_p, valid)
        entropy_loss_p = -self.entropy_loss_coeff * entropy_p

        loss_b = pi_loss_b + entropy_loss_b
        loss_p = pi_loss_p + entropy_loss_p
        loss = loss_b + loss_p + value_loss

        if torch.isnan(loss_b) or torch.isnan(loss_p) or torch.isnan(entropy_b):
            import ipdb; ipdb.set_trace()

        # print(f'loss b: {loss_b}, loss p: {loss_p}, entropy_b: {entropy_b}, entropy_p: {entropy_p}')

        perplexity_b = dist_b.mean_perplexity(dist_info_b, valid)
        perplexity_p = dist_p.mean_perplexity(dist_info_p, valid)

        return loss, loss_b, loss_p, entropy_b, entropy_p, perplexity_b, perplexity_p
