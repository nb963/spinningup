import time
import joblib
import os
import os.path as osp
import tensorflow as tf
import torch
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
import numpy as np
from IPython import embed
import robosuite
from PolicyNetworks import ContinuousPolicyNetwork

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def load_policy_and_env(fpath, itr='last', deterministic=False, return_policy=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the 
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a 
    PyTorch save.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    if backend == 'tf1':
        get_action = load_tf_policy(fpath, itr, deterministic)
    else:
        if return_policy:
            get_action, policy = load_pytorch_policy(fpath, itr, deterministic, return_policy)
        else:
            get_action = load_pytorch_policy(fpath, itr, deterministic, return_policy)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    if return_policy:
        return env, get_action, policy
    else:
        return env, get_action

def load_tf_policy(fpath, itr, deterministic=False):
    """ Load a tensorflow policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'tf1_save'+itr)
    print('\n\nLoading from %s.\n\n'%fname)

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, fname)

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    return get_action

def load_pytorch_policy(fpath, itr, deterministic=False, return_policy=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    if return_policy is True:
        return get_action, model
    else:
        return get_action

def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True, hierarchical=False):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0

    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)  

        if hierarchical: 
            a = a[0] 
        
        o, r, d, _ = env.step(a)

        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()

def render_episode(env, get_action, max_ep_len=None, hierarchical=False):

    # Function to render the episode using the simulation renderer.

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    o, r, done, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0

    # Create an image list. 
    image_list = []
    image = np.flipud(env.sim.render(600,600, camera_name='frontview'))
    image_list.append(image)

    while not(done):
        a = get_action(o)
        
        if hierarchical: 
            a = a[0] 

        o, r, done, _ = env.step(a)

        image = np.flipud(env.sim.render(600,600, camera_name='frontview'))
        image_list.append(image)

        ep_ret += r
        ep_len += 1

    return image_list

def hierarchical_run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True, args=None):
    
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0

    # here, we *always* rollout a skill for .. skill time limit * downsample freq # timesteps? 
    downsample_freq = 20
    orig_skill_time_limit = 14
    skill_time_limit = orig_skill_time_limit*downsample_freq
    eval_time_limit = 1000
    basedir = args.basedir

    if args.data in ['MIME']:
        state_size = 16
        lower_joint_limits = np.load(os.path.join(basedir,"MIME/MIME_Orig_Min.npy"))
        upper_joint_limits = np.load(os.path.join(basedir,"MIME/MIME_Orig_Max.npy"))
    elif args.data in ['Roboturk','FullRoboturk']:
        state_size = 8
        lower_joint_limits = np.load(os.path.join(basedir,"Roboturk/Roboturk_Min.npy"))
        upper_joint_limits = np.load(os.path.join(basedir,"Roboturk/Roboturk_Max.npy"))

    joint_limit_range = upper_joint_limits - lower_joint_limits
    input_size = 2*state_size
    output_size = state_size

    hidden_size = 48
    number_layers = 4
    args.batch_size = 1
    args.z_dimensions = 16
    latent_z_dimension = 16 
    args.dropout = 0.
    args.mean_nonlinearity = 0
    terminal = False
    max_ep_len = 1000

    # Instantiate policy based on parameters.    
    lowlevel_policy = ContinuousPolicyNetwork(input_size, hidden_size, output_size, args, number_layers).to(device)

    # If model file for lowlevel policy, load it. 
    if args.lowlevel_policy_model is not None:
        load_object = torch.load(args.lowlevel_policy_model)
        lowlevel_policy.load_state_dict(load_object['Policy_Network'])

    # Now collect num_episodes rather than just 1...
    while n < num_episodes:
        t = 0
        hidden = None

        while t<eval_time_limit and not(terminal):

            # z_action, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
            # Get z 
            z_action = get_action(o)            
            t_skill = 0

            while t_skill<skill_time_limit and not(terminal):

                ##########################################
                # 5) Sample low-level action a from low-level policy. 
                ##########################################

                # 5a) Get joint state from observation.
                
                obs_spec = env.observation_spec()
                max_gripper_state = 0.042

                if float(robosuite.__version__[:3])>1.:
                    pure_joint_state = env.sim.get_state()[1][:7]						
                    
                    if args.env_name=='Wipe':
                        # here, no gripper, so set dummy joint pos
                        gripper_state = np.array([max_gripper_state/2])
                    else:
                        gripper_state = np.array([obs_spec['robot0_gripper_qpos'][0]-obs_spec['robot0_gripper_qpos'][1]])
                else:
                    
                    pure_joint_state = obs_spec['joint_pos']					
                    gripper_state = np.array([obs_spec['gripper_qpos'][0]-obs_spec['gripper_qpos'][1]])
                
                # Norm gripper state from 0 to 1
                gripper_state = gripper_state/max_gripper_state
                # Norm gripper state from -1 to 1. 
                gripper_state = 2*gripper_state-1
                joint_state = np.concatenate([pure_joint_state, gripper_state])
                
                # Normalize joint state according to joint limits (minmax normaization).
                normalized_joint_state = (joint_state - lower_joint_limits)/joint_limit_range

                # 5b) Assemble input. 
                if t==0:
                    low_level_action_numpy = np.zeros_like(normalized_joint_state)                    
                assembled_states = np.concatenate([normalized_joint_state,low_level_action_numpy])
                assembled_input = np.concatenate([assembled_states, z_action])
                torch_assembled_input = torch.tensor(assembled_input).to(device).float().view(-1,1,input_size+latent_z_dimension)

                # 5c) Now actually retrieve action.
                low_level_action, hidden = lowlevel_policy.incremental_reparam_get_actions(torch_assembled_input, greedy=True, hidden=hidden)
                low_level_action_numpy = low_level_action.detach().squeeze().squeeze().cpu().numpy()
                unnormalized_low_level_action_numpy = args.action_scaling * low_level_action_numpy

                # 5d) Normalize action for benefit of environment. 
                # Output of policy is minmax normalized, which is 0-1 range. Change to -1 to 1 range. 
                normalized_low_level_action = unnormalized_low_level_action_numpy

                ##########################################
                # 6) Step in environment. 
                ##########################################

                # Set low level action.

                if args.env_name=='Wipe':
                    next_o, r, d, _ = env.step(normalized_low_level_action[:-1])
                else:
                    next_o, r, d, _ = env.step(normalized_low_level_action)

                ep_ret += r
                ep_len += 1
                t_skill += 1                    
                t+=1
                timeout = ep_len == max_ep_len
                terminal = d or timeout

                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
                    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                    n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()

def hierarchical_render_episode(env, get_action, max_ep_len=None, args=None):

    # Function to render the episode using the simulation renderer.

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    o, r, done, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0

    # Create an image list. 
    image_list = []
    image = np.flipud(env.sim.render(600,600, camera_name='frontview'))
    image_list.append(image)

    # here, we *always* rollout a skill for .. skill time limit * downsample freq # timesteps? 
    downsample_freq = 20
    orig_skill_time_limit = 14
    skill_time_limit = orig_skill_time_limit*downsample_freq
    eval_time_limit = 1000
    t = 0

    hidden_size = 48
    number_layers = 4
    latent_z_dimension = 16 
    basedir = args.basedir

    if args.data in ['MIME']:
        state_size = 16
        lower_joint_limits = np.load(os.path.join(basedir,"MIME/MIME_Orig_Min.npy"))
        upper_joint_limits = np.load(os.path.join(basedir,"MIME/MIME_Orig_Max.npy"))
    elif args.data in ['Roboturk','FullRoboturk']:
        state_size = 8
        lower_joint_limits = np.load(os.path.join(basedir,"Roboturk/Roboturk_Min.npy"))
        upper_joint_limits = np.load(os.path.join(basedir,"Roboturk/Roboturk_Max.npy"))
    
    hidden = None

    while t<eval_time_limit and not(d):

        # z_action, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
        # Get z 
        z_action = get_action(o)
        t_skill = 0

        while t_skill<skill_time_limit:

            ##########################################
            # 5) Sample low-level action a from low-level policy. 
            ##########################################

            # 5a) Get joint state from observation.
            
            obs_spec = env.observation_spec()
            max_gripper_state = 0.042

            if float(robosuite.__version__[:3])>1.:
                pure_joint_state = env.sim.get_state()[1][:7]						
                
                if args.env_name=='Wipe':
                    # here, no gripper, so set dummy joint pos
                    gripper_state = np.array([max_gripper_state/2])
                else:
                    gripper_state = np.array([obs_spec['robot0_gripper_qpos'][0]-obs_spec['robot0_gripper_qpos'][1]])
            else:
                
                pure_joint_state = obs_spec['joint_pos']					
                gripper_state = np.array([obs_spec['gripper_qpos'][0]-obs_spec['gripper_qpos'][1]])
            
            # Norm gripper state from 0 to 1
            gripper_state = gripper_state/max_gripper_state
            # Norm gripper state from -1 to 1. 
            gripper_state = 2*gripper_state-1
            joint_state = np.concatenate([pure_joint_state, gripper_state])
            
            # Normalize joint state according to joint limits (minmax normaization).
            normalized_joint_state = (joint_state - lower_joint_limits)/joint_limit_range

            # 5b) Assemble input. 
            if t==0:
                low_level_action_numpy = np.zeros_like(normalized_joint_state)                    
            assembled_states = np.concatenate([normalized_joint_state,low_level_action_numpy])
            assembled_input = np.concatenate([assembled_states, z_action])
            torch_assembled_input = torch.tensor(assembled_input).to(device).float().view(-1,1,input_size+latent_z_dimension)

            # 5c) Now actually retrieve action.
            low_level_action, hidden = lowlevel_policy.incremental_reparam_get_actions(torch_assembled_input, greedy=True, hidden=hidden)
            low_level_action_numpy = low_level_action.detach().squeeze().squeeze().cpu().numpy()
            unnormalized_low_level_action_numpy = args.action_scaling * low_level_action_numpy

            # 5d) Normalize action for benefit of environment. 
            # Output of policy is minmax normalized, which is 0-1 range. Change to -1 to 1 range. 
            normalized_low_level_action = unnormalized_low_level_action_numpy

            ##########################################
            # 6) Step in environment. 
            ##########################################

            # Set low level action.

            if args.env_name=='Wipe':
                next_o, r, d, _ = env.step(normalized_low_level_action[:-1])
            else:
                next_o, r, d, _ = env.step(normalized_low_level_action)

            if t%10==0:
                image_list.append(np.flipud(env.sim.render(600,600,camera_name='vizview1')))

            ep_ret += r
            ep_len += 1
            t+=1 
            t_skill+=1

    return image_list


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy_and_env(args.fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))