import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from IPython import embed
from PolicyNetworks import ContinuousPolicyNetwork
from gym.spaces import Box, Discrete

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)

        # CHANGE: Probably need to make this a more generic data structure to handle tuples. 
        # self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)

        # CHANGED: Making this a list (of arbitrary elements) of length = size. 
        # Since the buffer object itself is just length size for a particular epoch.
        self.act_buf = [[] for i in range(size)]

        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs

        # CHANGE: May possibly need to change this.   
        # CHANGED: Just need to torchify these actions before storing them. 
        # What we need to worry about is... torch taking up GPU space! Instead, storing them in normal RAM maybe better? 
        torch_act = [torch.as_tensor(x, dtype=torch.float32) for x in act]
        self.act_buf[self.ptr] = torch_act

        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        # The next step creates 
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)

        # Two options / ways to go about this - 
        # 1) Torchify everything here, and then store it in the same dictionary form. 
        # 2) Modify how data is stored. Instead of have it store an action tuple, make it explicitly store separate components of actions explicitly.       
        
        # Choose the first option. 
        # Assume the actions are torchified, and now torchify everything else. 
        return_dictionary = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items() if k != 'act'}
        # Now add the actions to the dictionary. 
        return_dictionary['act'] = data['act']

        # # Recreate dictionary with torch tensors for everything. 
        # return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

        # Now return the return_dictionary. 
        return return_dictionary

def hierarchical_ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10, args=None):
    
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    #######################################################
    # Setup code in general. 
    #######################################################

    if True:
        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()

        # Set up logger and save configuration
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())

        # Random seed
        seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Instantiate environment
        env = env_fn()
        obs_dim = env.observation_space.shape

        act_dim = env.action_space.shape
        

        # # CHANGED: actor_critic now refers to Hierarchical actor critic defined in core. 
        # # Create actor-critic module
        # ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
        
        #####################################################
        # Changing to implementing as an MLPactorcritic but with a few additional functions.. 
        #####################################################
        
        if True:
            latent_z_dimension = 16 
            # Creating a special action space. 
            action_space_bound = np.ones(latent_z_dimension)*np.inf
            action_space = Box(-action_space_bound, action_space_bound)

            ac = actor_critic(env.observation_space, action_space, **ac_kwargs)

        #####################################################
        # Also now instantiate low level policy. 
        #####################################################

        if True:
            # Set parameters. 

            hidden_size = 48
            number_layers = 4

            if args.data=='MIME':
                state_size = 16

            elif args.data in ['Roboturk','FullRoboturk']:
                state_size = 8

            input_size = 2*state_size
            output_size = state_size
            #  
            args.batch_size = 32
            args.z_dimensions = 16
            args.dropout = 0.

            # Instantiate policy based on parameters.    
            lowlevel_policy = ContinuousPolicyNetwork(input_size, hidden_size, output_size, args, number_layers).to(device)

            # If model file for lowlevel policy, load it. 
            if args.lowlevel_policy_model is not None:
                load_object = torch.load(args.lowlevel_policy_model)
                lowlevel_policy.load_state_dict(load_object['Policy_Network'])

        # Sync params across processes
        sync_params(ac)

        # Count variables
        var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
        logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

        # Set up experience buffer
        local_steps_per_epoch = int(steps_per_epoch / num_procs())
        buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    #######################################################
    # Set up function for computing PPO policy loss
    #######################################################

    def compute_loss_pi(data):

        # # Don't need to change this. 
        # obs, action_tuple, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # # Policy loss
        # # CHANGE: Don't need to change this code itself, but rather need to change forward of the actor critic policy to implement log probability correctly. 
        # # Now that we have transition from the buffer, we need to evaluate the likelihood of this action under the current estimate of the policy. 
        # # PPO uses both the log probability under the old policy and the new policy to do this. 
        # # This ac.pi(obs, act) call below just evaluates the logprobability and gets the distribution for the current policy with the previous action
        # # Must create an equivalent for the hierarchical policy to get the logprobability of hierarchical actions as well.

        # # CHANGED from:
        # # pi, logp = ac.pi(obs, action_tuple)
        # # Remember, action_tuple is not just a single datapoint. It is an array of datapoints containing action tuples. 
        # # Modify this to compute log probabilities over the batch.

        # pi, b_pi, z_pi, logp = ac.evaluate_batch_logprob(obs, action_tuple)

        # CHANGED TO sampling just z. 
        # Since we don't need a separate function for evaluating batch_logprob, just use ac.pi.
        obs, z_act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        pi, logp = ac.pi(obs, z_act)

        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()

        # CHANGE: NOTE: This is entropy of the low level policy distribution. Since this is only used for logging and not training, this is fine. 
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    #######################################################
    # Critic function loss
    #######################################################
    
    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    #######################################################
    # Set up optimizers for policy and value function
    #######################################################

    if True:
        # CHANGED: Use all parameters.     
        # Can't just use the ac.parameters(), because we need separate optimizers for the latent and low-level policy optimizers. 
        pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)


        # Now that we're moving to the MLP Actor Critic version, without actual Hierarchical Actor Critic, just implement as standard optimizer list. 

        # parameter_list = list(ac.pi.parameters())+list(ac.latent_b_policy.parameters())+list(ac.latent_z_policy.parameters())
        # parameter_list = list(ac.pi.parameters())+list(ac.latent_b_policy.parameters())+list(ac.latent_z_policy.parameters())
        # pi_optimizer = Adam(parameter_list, lr=pi_lr)
        vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

        # Set up model saving
        logger.setup_pytorch_saver(ac)

    #######################################################
    # Update function
    #######################################################
    
    # We never particularly changed the update function - we changed the compute_loss functions instead. 

    def update():

        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    #######################################################
    # Remember, new rollout procedure. 
    #######################################################

    if True:

        # 1) Initialize. 
        # 2) While we haven't exceeded timelimit and are still non-terminal:
        #   # 3) Sample z from z policy. 
        #   # 4) While we haven't exceeded skill timelimit and are still non terminal, and haven't exceeded overall timelimit. 
        #       # 5) Sample low-level action a from low-level policy. 
        #       # 6) Step in environment. 
        #       # 7) Increment counters, reset states, log cummulative rewards, etc. 
        #   # 8) Reset episode if necessary.           
    
        # Prepare for interaction with environment
        start_time = time.time()
        o, ep_ret, ep_len = env.reset(), 0, 0
        
        # Set global skill time limit.
        skill_time_limit = 14

        ##########################################
        # 1) Initialize / reset. 
        ##########################################

        for epoch in range(epochs):

            ##########################################
            # 1) Initialize / reset. (State is actually already reset here.)
            ##########################################

            t = 0 
            # reset hidden state for incremental policy forward.
            hidden = None

            ##########################################
            # 2) While we haven't exceeded timelimit and are still non-terminal:
            ##########################################

            while t<local_steps_per_epoch:

                ##########################################
                # 3) Sample z from z policy. 
                ##########################################

                # Now no longer need a hierarchical actor critic? 
                # Probably can implement as usual - just assemble appropriate inputs and query high level policy. 
                # And then query low level policy....? Hmmmmmm, how does update work? 
                # Probably need to feed high-level policy log probabilities to PPO to get it to work. 

                # Revert to regular forward of Z AC (not receiving tuples).
                # action_tuple, v, logp_tuple = ac.step(torch.as_tensor(o, dtype=torch.float32))
                z_action, v, z_logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

                # First reset skill timer. 
                t_skill = 0

                ##########################################
                # 4) While we haven't exceeded skill timelimit and are still non terminal, and haven't exceeded overall timelimit. 
                ##########################################

                while t_skill<skill_time_limit and not(terminal) and t<local_steps_per_epoch:
                    
                    ##########################################
                    # 5) Sample low-level action a from low-level policy. 
                    ##########################################

                    # 5a) Get joint state from observation.
                    pure_joint_state = o['joint_pos']
                    gripper_state = o['gripper_qpos'][0]-o['gripper_qpos'][1]
                    joint_state = np.concatenate([pure_joint_state, gripper_state])

                    # 5b) Assemble input. 
                    if t==0:
                        low_level_action = np.zeros_like(joint_state)
                    assembled_input = torch.tensor(np.concatenate([joint_state,low_level_action])).to(device).float()

                    # 5c) Now actually retrieve action.
                    low_level_action, hidden = lowlevel_policy.incremental_reparam_get_actions(assembled_input, greedy=True, hidden=hidden)

                    # 5d) Normalize action for benefit of environment. 
                    # normalized_low_level_action = 

                    ##########################################
                    # 6) Step in environment. 
                    ##########################################

                    # Set low level action.
                    
                    next_o, r, d, _ = env.step(normalized_low_level_action)

                    ##########################################
                    # 7) Increment counters, reset states, log cummulative rewards, etc. 
                    ##########################################

                    ep_ret += r
                    ep_len += 1

                    # Also adding to the skill time. # Treating overall time the same.
                    t_skill += 1

                    # save and log
                    # CHANGED: Saving the action tuple in the buffer instead of just the action..
                    # buf.store(o, action_tuple, r, v, logp_tuple)
                    # CHANGING TO STORING Z ACTION AND Z LOGP.
                    buf.store(o, z_action, r, v, z_logp)
                    logger.store(VVals=v)
                    
                    # Update obs (critical!)
                    o = next_o

                    timeout = ep_len == max_ep_len
                    terminal = d or timeout
                    epoch_ended = t==local_steps_per_epoch-1

                    if terminal or epoch_ended:
                        if epoch_ended and not(terminal):
                            print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                        # if trajectory didn't reach terminal state, bootstrap value target
                        if timeout or epoch_ended:
                            _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                        else:
                            v = 0
                        buf.finish_path(v)
                        if terminal:
                            # only save EpRet / EpLen if trajectory finished
                            logger.store(EpRet=ep_ret, EpLen=ep_len)
                        o, ep_ret, ep_len = env.reset(), 0, 0
            
            ##########################################
            # 8) Save, update, and log. 
            ##########################################

            # Save model

            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)

            # Perform PPO update!
            update()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('DeltaLossPi', average_only=True)
            logger.log_tabular('DeltaLossV', average_only=True)
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('ClipFrac', average_only=True)
            logger.log_tabular('StopIter', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()



        # #############################################################
        # # Old form of rollout in training loop. 
        # #############################################################

        # # Main loop: collect experience in env and update/log each epoch
        # for epoch in range(epochs):
        #     for t in range(local_steps_per_epoch):

        #         # CHANGED: Firstly, make sure the policy class implements actions as a tuple of a, z, b, and joint log probability of these. 
        #         action_tuple, v, logp_tuple = ac.step(torch.as_tensor(o, dtype=torch.float32))
        #         # CHANGED: Now remember, the action is a tuple of a, z, b. So take a step in the low level environment with the low-level action. 
        #         next_o, r, d, _ = env.step(action_tuple[0])

        #         ep_ret += r
        #         ep_len += 1

        #         # save and log
        #         # CHANGED: Saving the action tuple in the buffer instead of just the action..
        #         buf.store(o, action_tuple, r, v, logp_tuple)
        #         logger.store(VVals=v)
                
        #         # Update obs (critical!)
        #         o = next_o

        #         timeout = ep_len == max_ep_len
        #         terminal = d or timeout
        #         epoch_ended = t==local_steps_per_epoch-1

        #         if terminal or epoch_ended:
        #             if epoch_ended and not(terminal):
        #                 print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
        #             # if trajectory didn't reach terminal state, bootstrap value target
        #             if timeout or epoch_ended:
        #                 _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
        #             else:
        #                 v = 0
        #             buf.finish_path(v)
        #             if terminal:
        #                 # only save EpRet / EpLen if trajectory finished
        #                 logger.store(EpRet=ep_ret, EpLen=ep_len)
        #             o, ep_ret, ep_len = env.reset(), 0, 0


        #     # Save model
        #     if (epoch % save_freq == 0) or (epoch == epochs-1):
        #         logger.save_state({'env': env}, None)

        #     # Perform PPO update!
        #     update()

        #     # Log info about epoch
        #     logger.log_tabular('Epoch', epoch)
        #     logger.log_tabular('EpRet', with_min_and_max=True)
        #     logger.log_tabular('EpLen', average_only=True)
        #     logger.log_tabular('VVals', with_min_and_max=True)
        #     logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        #     logger.log_tabular('LossPi', average_only=True)
        #     logger.log_tabular('LossV', average_only=True)
        #     logger.log_tabular('DeltaLossPi', average_only=True)
        #     logger.log_tabular('DeltaLossV', average_only=True)
        #     logger.log_tabular('Entropy', average_only=True)
        #     logger.log_tabular('KL', average_only=True)
        #     logger.log_tabular('ClipFrac', average_only=True)
        #     logger.log_tabular('StopIter', average_only=True)
        #     logger.log_tabular('Time', time.time()-start_time)
        #     logger.dump_tabular()

        #     #############################################################
        #     # End 
        #     #############################################################

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # ppo(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
    #     ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
    #     seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
    #     logger_kwargs=logger_kwargs)

    # CHANGED: 
    hierarchical_ppo(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
    seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
    logger_kwargs=logger_kwargs)
    #######################################################