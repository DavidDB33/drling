import drling
import gym
import os.path
import numpy as np
import tensorflow as tf

DIRNAME = os.path.join("tests", "res")

def _get_Box(n):
    return gym.spaces.Box(low=np.zeros(n, dtype=np.float32), high=np.ones(n, dtype=np.float32), dtype=np.float32)

def _get_spaces(o, a):
    obs_space = _get_Box(o)
    act_space = gym.spaces.MultiDiscrete(a)
    return obs_space, act_space

def _get_Agent2(spaces):
    obs_space, act_space = spaces
    config = drling.get_config(path=os.path.join(DIRNAME, "config.yaml"))
    model = drling.get_model("DQNv2", obs_space, act_space, config)
    agent = drling.get_agent(label="Agentv2", model=model, memory="Memoryv1")
    return agent

def _get_Agent(spaces):
    obs_space, act_space = spaces
    config = drling.get_config(path=os.path.join(DIRNAME, "config.yaml"))
    model = drling.get_model("DQNv1", obs_space, act_space, config)
    agent = drling.get_agent(label="Agentv1", model=model, memory="Memoryv1")
    return agent

def test__initialize():
    spaces = obs_space, act_space = _get_spaces(5, (3,3))
    agent = _get_Agent(spaces)

def test_agent_qvalue():
    ### Expected values
    qvalues_E = np.array([[[0.7037938237190247,  0.0970928966999054,  -0.3128317892551422,
                            0.0990651547908783, -0.36335521936416626, -0.45463865995407104,
                            0.9741138219833374, -0.4892215132713318,   1.066206932067871]]])
    qvalue1_E = np.array(0.0970928966999054)
    qvalue5_E = np.array(-0.45463865995407104)
    qvalue6_E = np.array(0.9741138219833374)
    ### Setup
    tf.random.set_seed(1)
    spaces = obs_space, act_space = _get_spaces(5, (3,3))
    agent = _get_Agent(spaces)
    o = tf.constant([[[0, 0.2, 0.4, 0.6, 0.8]]])
    ### Real values
    qvalues_R = agent(o).numpy()
    qvalue1_R = agent.qvalue(o, tf.constant(1))
    qvalue5_R = agent.qvalue(o, tf.constant(5))
    qvalue6_R = agent.qvalue(o, tf.constant(6))
    ### Assert
    np.testing.assert_array_almost_equal_nulp(qvalues_R, qvalues_E)
    np.testing.assert_array_almost_equal_nulp(qvalue1_R, qvalue1_E)
    np.testing.assert_array_almost_equal_nulp(qvalue5_R, qvalue5_E)
    np.testing.assert_array_almost_equal_nulp(qvalue6_R, qvalue6_E)

def test_agent_qvalue_max():
    ### Expected values
    qvalues_E = np.array([[[0.7037938237190247,  0.0970928966999054,  -0.3128317892551422,
                            0.0990651547908783, -0.36335521936416626, -0.45463865995407104,
                            0.9741138219833374, -0.4892215132713318,   1.066206932067871]]])
    qvalue_max_E = np.array(1.066206932067871)
    ### Setup
    tf.random.set_seed(1)
    spaces = obs_space, act_space = _get_spaces(5, (3,3))
    agent = _get_Agent(spaces)
    o = tf.constant([[[0, 0.2, 0.4, 0.6, 0.8]]])
    ### Real values
    qvalues_R = agent(o).numpy()
    qvalue_max_R = agent.qvalue_max(o)
    ### Assert
    np.testing.assert_array_almost_equal_nulp(qvalues_R, qvalues_E)
    np.testing.assert_array_almost_equal_nulp(qvalue_max_R, qvalue_max_E)

def test_agent_act():
    ### Expected values
    qvalues_E = np.array([[[0.7037938237190247,  0.0970928966999054,  -0.3128317892551422,
                            0.0990651547908783, -0.36335521936416626, -0.45463865995407104,
                            0.9741138219833374, -0.4892215132713318,   1.066206932067871]]])
    act_E = np.array([2,2]) # 8
    ### Setup
    tf.random.set_seed(1)
    spaces = obs_space, act_space = _get_spaces(5, (3,3))
    agent = _get_Agent(spaces)
    o = tf.constant([[[0, 0.2, 0.4, 0.6, 0.8]]])
    ### Real values
    qvalues_R = agent(o).numpy()
    act_R = agent.act(o)
    ### Assert
    np.testing.assert_array_almost_equal_nulp(qvalues_R, qvalues_E)
    np.testing.assert_array_almost_equal_nulp(act_R, act_E)
    
def test_agent_guess():
    ### Setup params
    HISTORY_LENGTH = 6
    N_OBS = 5
    ### Expected values
    h0_E = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0]])
    h2_E = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.8073904514312744,   0.514500081539154,   0.19111815094947815, 0.18687477707862854, 0.7725216150283813],  # 1st obs
                     [0.051827892661094666, 0.06952621042728424, 0.5151751041412354,  0.6885436177253723,  0.4596281051635742]]) # 2nd obs
    h6_E = np.array([[0.8073904514312744,   0.514500081539154,   0.19111815094947815, 0.18687477707862854, 0.7725216150283813],  # 1st obs
                     [0.051827892661094666, 0.06952621042728424, 0.5151751041412354,  0.6885436177253723,  0.4596281051635742],  # 2nd obs
                     [0.9020112156867981,   0.7752674221992493,  0.17072395980358124, 0.02876790426671505, 0.8895143866539001],  # 3th obs
                     [0.41380903124809265,  0.4714929759502411,  0.508656919002533,   0.8238011002540588,  0.8263044357299805],  # 4th obs
                     [0.5830936431884766,   0.04657283052802086, 0.8071987628936768,  0.8190988302230835,  0.06596405059099197], # 5th obs
                     [0.6259687542915344,   0.20261245965957642, 0.7555670738220215,  0.7613345980644226,  0.6785264015197754]]) # 6th obs
    ### Setup
    spaces = obs_space, act_space = _get_spaces(N_OBS, (3,3))
    obs_space.seed(1)
    o_list = [obs_space.sample() for _ in range(HISTORY_LENGTH)]
    tf.random.set_seed(1)
    agent = _get_Agent2(spaces)
    ### Real values
    h_list = [agent.guess()]
    for o in o_list:
        h_list.append(agent.guess(o, h_list[-1]))
    h0_R = h_list[0]
    h2_R = h_list[2]
    h6_R = h_list[6]
    ### Assert
    np.testing.assert_array_almost_equal_nulp(h0_R, h0_E)
    np.testing.assert_array_almost_equal_nulp(h2_R, h2_E)
    np.testing.assert_array_almost_equal_nulp(h6_R, h6_E)

if __name__ == "__main__":
    spaces = obs_space, act_space = _get_spaces(5, (3,3))
    config = drling.get_config(path=os.path.join(DIRNAME, "config.yaml"))
    tf.random.set_seed(1)
    m = drling.get_model("DQNv1", obs_space, act_space, config)
    a = drling.get_agent(label="Agentv1", model=m, memory="Memoryv1")
    tf.random.set_seed(1)
    m2 = drling.get_model("DQNv2", obs_space, act_space, config)
    a2 = drling.get_agent(label="Agentv2", model=m2, memory="Memoryv1")
    obs_space.seed(1);
    o_list = [obs_space.sample() for _ in range(6)]
    h_list = [a2.guess()]
    for o in o_list:
        h_list.append(a2.guess(o, h_list[-1]))
