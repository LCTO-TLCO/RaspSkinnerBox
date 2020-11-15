import numpy as np
from lerning_rule_func import *
NUM_ACTION = 5

### 方策
def try_except_beta_softmax_p(q_value, beta):
    try:
        p = np.exp(q_value*beta) / np.sum(np.exp(q_value*beta))
    except RuntimeWarning as e:
        print("Policy softmax RuntimeWarning !")
        p = np.zeros_like(q_value)
        p[np.argmax(q_value)] = 1
    return p

def softmax_p(q_value, params_dict, other_dict):
    try:
        beta = params_dict["beta"]
    except:
        beta = 1

    #p = try_except_beta_softmax_p(q_value, beta)
    x = q_value - np.max(q_value)
    p = np.exp(x*beta) / np.sum(np.exp(x*beta))
    return p

def greedy_p(q_value, params_dict, other_dict):
    p = np.zeros_like(q_value, dtype="float")
    p[np.argmax(q_value)] = 1
    return p


def e_greedy_p(q_value, params_dict, other_dict):
    epsilon = params_dict["epsilon"]
    p = np.zeros_like(q_value, dtype="float")
    p[np.argmax(q_value)] += 1 - epsilon
    p += epsilon / NUM_ACTION
    return p


def no_count_rs_p(q_value, params_dict, other_dict):
    aleph = params_dict["aleph"]
    max_q = max(q_value)
    if max_q < aleph:
        return 1 / (np.sum(1 / (aleph - q_value)) * (aleph - q_value))
    else:
        return greedy_p(q_value, params_dict, other_dict)


def use_count_rs_p(q_value, params_dict, other_dict):
    aleph = params_dict["aleph"]
    try:
        alpha_3 = params_dict["alpha_3"]
    except:
        alpha_3 = 0
    n = other_dict["n"]
    step = 100  # どれだけ先のステップまでカウントして確率を出すか

    max_q = max(q_value)
    if max_q < aleph:
        tmp_n = np.copy(n)
        new_n = np.zeros(NUM_ACTION)
        for i in range(step):
            rs_value = tmp_n * (q_value - aleph)
            action = np.argmax(rs_value)
            tmp_n[action] = (tmp_n[action] + 1) / (1 - alpha_3)
            tmp_n = (1 - alpha_3) * tmp_n
            new_n[action] += 1

        return new_n / np.sum(new_n)
    else:
        return greedy_p(q_value, params_dict, other_dict)

### simple_asp_e_greedy_p

def simple_asp_use_wei_e_greedy_p(q_value, params_dict, other_dict):
    aleph = params_dict["aleph"]
    epsilon = params_dict["epsilon"]  # ここで使うわけじゃないけど必要だよ
    weight = other_dict["weight"][other_dict["i"]]

    if weight < aleph:
        return e_greedy_p(q_value, params_dict, other_dict)
    else:
        return greedy_p(q_value, params_dict, other_dict)

def simple_asp_use_foo_e_greedy_p(q_value, params_dict, other_dict):
    aleph = params_dict["aleph"]
    epsilon = params_dict["epsilon"]  # ここで使うわけじゃないけど必要だよ
    num_fooding = other_dict["num_fooding"][other_dict["i"]]

    if num_fooding < aleph:
        return e_greedy_p(q_value, params_dict, other_dict)
    else:
        return greedy_p(q_value, params_dict, other_dict)

def simple_asp_use_maxQ_e_greedy_p(q_value, params_dict, other_dict):
    aleph = params_dict["aleph"]
    epsilon = params_dict["epsilon"]  # ここで使うわけじゃないけど必要だよ
    max_q = max(q_value)

    if max_q < aleph:
        return e_greedy_p(q_value, params_dict, other_dict)
    else:
        return greedy_p(q_value, params_dict, other_dict)

### simple_asp_softmax_p

def simple_asp_use_wei_softmax_p(q_value, params_dict, other_dict):
    aleph = params_dict["aleph"]
    weight = other_dict["weight"][other_dict["i"]]
    if weight < aleph:
        return softmax_p(q_value, params_dict, other_dict)
    else:
        return greedy_p(q_value, params_dict, other_dict)

def simple_asp_use_foo_softmax_p(q_value, params_dict, other_dict):
    aleph = params_dict["aleph"]
    num_fooding = other_dict["num_fooding"][other_dict["i"]]
    if num_fooding < aleph:
        return softmax_p(q_value, params_dict, other_dict)
    else:
        return greedy_p(q_value, params_dict, other_dict)

def simple_asp_use_maxQ_softmax_p(q_value, params_dict, other_dict):
    aleph = params_dict["aleph"]
    max_q = max(q_value)
    if max_q < aleph:
        return softmax_p(q_value, params_dict, other_dict)
    else:
        return greedy_p(q_value, params_dict, other_dict)

### asp_lower_gradation_softmax_p

def asp_lower_gradation_use_wei_softmax_p(q_value, params_dict, other_dict):
    aleph = params_dict["aleph"]
    b = params_dict["b"]
    weight = other_dict["weight"][other_dict["i"]]
    ro = min(weight / (aleph + 1e-3), 1) ** b
    return ro * greedy_p(q_value, params_dict, other_dict) + (1 - ro) * softmax_p(q_value, params_dict, other_dict)

def asp_lower_gradation_use_foo_softmax_p(q_value, params_dict, other_dict):
    aleph = params_dict["aleph"]
    b = params_dict["b"]
    num_fooding = other_dict["num_fooding"][other_dict["i"]]
    ro = min(num_fooding / (aleph + 1e-3), 1) ** b
    return ro * greedy_p(q_value, params_dict, other_dict) + (1 - ro) * softmax_p(q_value, params_dict, other_dict)

def asp_lower_gradation_use_maxQ_softmax_p(q_value, params_dict, other_dict):
    aleph = params_dict["aleph"]
    b = params_dict["b"]
    max_q = max(q_value)
    ro = min(max(max_q, 1e-3) / (aleph + 1e-3), 1) ** b
    return ro * greedy_p(q_value, params_dict, other_dict) + (1 - ro) * softmax_p(q_value, params_dict, other_dict)

### asp_upper_gradation_softmax_p

def asp_upper_gradation_softmax_p(q_value, params_dict, other_dict):
    aleph = params_dict["aleph"]
    b = params_dict["b"]
    max_q = max(q_value)
    ro = (1 - min(aleph / max(max_q, 1e-3), 1)) ** b
    return ro * greedy_p(q_value, params_dict, other_dict) + (1 - ro) * softmax_p(q_value, params_dict, other_dict)

### scaling_type

def beta_scaling_base_softmax_p(q_value, params_dict, scaling_func, y):
    scale = scaling_func(y, params_dict)
    scaled_beta = 1 * scale
    p = try_except_beta_softmax_p(q_value, scaled_beta)
    return p

### beta_scaling_y_softmax_p

def beta_scaling_y_use_wei_softmax_p(q_value, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = scaling_y
    p = beta_scaling_base_softmax_p(q_value, params_dict, scaling_func, y)
    return p

def beta_scaling_y_use_foo_softmax_p(q_value, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = scaling_y
    p = beta_scaling_base_softmax_p(q_value, params_dict, scaling_func, y)
    return p

def beta_scaling_y_use_maxQ_softmax_p(q_value, params_dict, other_dict):
    y = max(q_value)
    scaling_func = scaling_y
    p = beta_scaling_base_softmax_p(q_value, params_dict, scaling_func, y)
    return p

### beta_scaling_y_minus_a_softmax_p

def beta_scaling_y_minus_a_use_wei_softmax_p(q_value, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = scaling_y_minus_a
    p = beta_scaling_base_softmax_p(q_value, params_dict, scaling_func, y)
    return p

def beta_scaling_y_minus_a_use_foo_softmax_p(q_value, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = scaling_y_minus_a
    p = beta_scaling_base_softmax_p(q_value, params_dict, scaling_func, y)
    return p

def beta_scaling_y_minus_a_use_maxQ_softmax_p(q_value, params_dict, other_dict):
    y = max(q_value)
    scaling_func = scaling_y_minus_a
    p = beta_scaling_base_softmax_p(q_value, params_dict, scaling_func, y)
    return p

###　beta_scaling_a_minus_y_softmax_p

def beta_scaling_a_minus_y_use_wei_softmax_p(q_value, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = scaling_a_minus_y
    p = beta_scaling_base_softmax_p(q_value, params_dict, scaling_func, y)
    return p

def beta_scaling_a_minus_y_use_foo_softmax_p(q_value, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = scaling_a_minus_y
    p = beta_scaling_base_softmax_p(q_value, params_dict, scaling_func, y)
    return p

def beta_scaling_a_minus_y_use_maxQ_softmax_p(q_value, params_dict, other_dict):
    y = max(q_value)
    scaling_func = scaling_a_minus_y
    p = beta_scaling_base_softmax_p(q_value, params_dict, scaling_func, y)
    return p

### beta_scaling_y_slash_a_softmax_p

def beta_scaling_y_slash_a_use_wei_softmax_p(q_value, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = scaling_y_slash_a
    p = beta_scaling_base_softmax_p(q_value, params_dict, scaling_func, y)
    return p

def beta_scaling_y_slash_a_use_foo_softmax_p(q_value, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = scaling_y_slash_a
    p = beta_scaling_base_softmax_p(q_value, params_dict, scaling_func, y)
    return p

def beta_scaling_y_slash_a_use_maxQ_softmax_p(q_value, params_dict, other_dict):
    y = max(q_value)
    scaling_func = scaling_y_slash_a
    p = beta_scaling_base_softmax_p(q_value, params_dict, scaling_func, y)
    return p

def parallel_softmax_p(q_values, params_dict, other_dict):
    try:
        beta = params_dict["beta"]
    except:
        beta = 1

    ps = np.exp(q_values * beta) / np.sum(np.exp(q_values * beta), axis=1)[:, None]
    return ps