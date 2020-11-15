import numpy as np
NUM_ACTION = 5

### 価値関数
def Q_update(q_value, action, reward, params_dict, other_dict):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = 0
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = 0

    for i in range(NUM_ACTION):
        if i == action:
            if reward == 1:
                q_value[i] = (1 - alpha_1) * q_value[i] + alpha_1 * kappa_1
            else:
                q_value[i] = (1 - alpha_1) * q_value[i] - alpha_1 * kappa_2
        else:
            q_value[i] = (1 - alpha_2) * q_value[i]
    return q_value


def FQ_update(q_value, action, reward, params_dict, other_dict):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = alpha_1
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]

    for i in range(NUM_ACTION):
        if i == action:
            if reward == 1:
                q_value[i] = (1 - alpha_1) * q_value[i] + alpha_1 * kappa_1
            else:
                q_value[i] = (1 - alpha_1) * q_value[i] - alpha_1 * kappa_2
        else:
            q_value[i] = (1 - alpha_2) * q_value[i]
    return q_value


def DFQ_update(q_value, action, reward, params_dict, other_dict):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = params_dict["alpha_2"]
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]

    for i in range(NUM_ACTION):
        if i == action:
            if reward == 1:
                q_value[i] = (1 - alpha_1) * q_value[i] + alpha_1 * kappa_1
            else:
                q_value[i] = (1 - alpha_1) * q_value[i] - alpha_1 * kappa_2
        else:
            q_value[i] = (1 - alpha_2) * q_value[i]
    return q_value


def MBIEEB_FQ_update(q_value, action, reward, params_dict, other_dict):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = alpha_1
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]
    b = params_dict["b"]
    n = other_dict["n"]

    for i in range(NUM_ACTION):
        if i == action:
            add_rew = b / np.sqrt(n[action])
            if reward == 1:
                q_value[i] = (1 - alpha_1) * q_value[i] + alpha_1 * (kappa_1 + add_rew)
            else:
                q_value[i] = (1 - alpha_1) * q_value[i] - alpha_1 * (-kappa_2 + add_rew)
        else:
            q_value[i] = (1 - alpha_2) * q_value[i]
    return q_value

### scaling_type

def scaling_y(y, params_dict):
    b = params_dict["b"]
    return min(max(y, 1e-3), 1e6) ** b

def scaling_y_minus_a(y, params_dict):
    a = params_dict["a"]
    b = params_dict["b"]
    return min(max(y - a, 1e-3), 1e6) ** b

def scaling_a_minus_y(y, params_dict):
    a = params_dict["a"]
    b = params_dict["b"]
    return min(max(a - y, 1e-3), 1e6) ** b

def scaling_y_slash_a(y, params_dict):
    a = params_dict["a"]
    b = params_dict["b"]
    return min(max(y / (a + 1e-3), 1e-3), 1e6) ** b

def limit_scaling_y(y, params_dict):
    b = params_dict["b"]
    return min(max(y, 1e-3) ** b, 1)

def limit_scaling_y_minus_a(y, params_dict):
    a = params_dict["a"]
    b = params_dict["b"]
    return min(max(y - a, 1e-3) ** b, 1)

def limit_scaling_a_minus_y(y, params_dict):
    a = params_dict["a"]
    b = params_dict["b"]
    return min(max(a - y, 1e-3) ** b, 1)

def limit_scaling_y_slash_a(y, params_dict):
    a = params_dict["a"]
    b = params_dict["b"]
    return min(max(y / (a + 1e-3), 1e-3) ** b, 1)

def kappa_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = alpha_1
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]

    scale = scaling_func(y, params_dict)
    scaled_kappa_1 = kappa_1 * scale
    scaled_kappa_2 = kappa_2 * scale
    for i in range(NUM_ACTION):
        if i == action:
            if reward == 1:
                q_value[i] = (1 - alpha_1) * q_value[i] + alpha_1 * scaled_kappa_1
            else:
                q_value[i] = (1 - alpha_1) * q_value[i] - alpha_1 * scaled_kappa_2
        else:
            q_value[i] = (1 - alpha_2) * q_value[i]
    return q_value

def kappa_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = alpha_1
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]

    for i in range(NUM_ACTION):
        if i == action:
            scale = scaling_func(q_value[i], params_dict)
            scaled_kappa_1 = kappa_1 * scale
            scaled_kappa_2 = kappa_2 * scale
            if reward == 1:
                q_value[i] = (1 - alpha_1) * q_value[i] + alpha_1 * scaled_kappa_1
            else:
                q_value[i] = (1 - alpha_1) * q_value[i] - alpha_1 * scaled_kappa_2
        else:
            q_value[i] = (1 - alpha_2) * q_value[i]
    return q_value

def alpha_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = alpha_1
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]

    scale = scaling_func(y, params_dict)
    scaled_alpha_1 = min(alpha_1 * scale, 1)
    for i in range(NUM_ACTION):
        if i == action:
            if reward == 1:
                q_value[i] = (1 - scaled_alpha_1) * q_value[i] + scaled_alpha_1 * kappa_1
            else:
                q_value[i] = (1 - scaled_alpha_1) * q_value[i] - scaled_alpha_1 * kappa_2
        else:
            q_value[i] = (1 - alpha_2) * q_value[i]
    return q_value

def alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = alpha_1
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]

    for i in range(NUM_ACTION):
        if i == action:
            scale = scaling_func(q_value[i], params_dict)
            scaled_alpha_1 = min(alpha_1 * scale, 1)
            if reward == 1:
                q_value[i] = (1 - scaled_alpha_1) * q_value[i] + scaled_alpha_1 * kappa_1
            else:
                q_value[i] = (1 - scaled_alpha_1) * q_value[i] - scaled_alpha_1 * kappa_2
        else:
            q_value[i] = (1 - alpha_2) * q_value[i]
    return q_value

### kappa_scaling_y_FQ_update

def kappa_scaling_y_use_wei_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = scaling_y
    q_value = kappa_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def kappa_scaling_y_use_foo_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = scaling_y
    q_value = kappa_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def kappa_scaling_y_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = scaling_y
    q_value = kappa_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

def limit_kappa_scaling_y_use_wei_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = limit_scaling_y
    q_value = kappa_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def limit_kappa_scaling_y_use_foo_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = limit_scaling_y
    q_value = kappa_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def limit_kappa_scaling_y_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = limit_scaling_y
    q_value = kappa_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

### kappa_scaling_y_minus_a_FQ_update

def kappa_scaling_y_minus_a_use_wei_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = scaling_y_minus_a
    q_value = kappa_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def kappa_scaling_y_minus_a_use_foo_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = scaling_y_minus_a
    q_value = kappa_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def kappa_scaling_y_minus_a_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = scaling_y_minus_a
    q_value = kappa_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

def limit_kappa_scaling_y_minus_a_use_wei_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = limit_scaling_y_minus_a
    q_value = kappa_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def limit_kappa_scaling_y_minus_a_use_foo_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = limit_scaling_y_minus_a
    q_value = kappa_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def limit_kappa_scaling_y_minus_a_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = limit_scaling_y_minus_a
    q_value = kappa_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

### kappa_scaling_a_minus_y_FQ_update

def kappa_scaling_a_minus_y_use_wei_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = scaling_a_minus_y
    q_value = kappa_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def kappa_scaling_a_minus_y_use_foo_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = scaling_a_minus_y
    q_value = kappa_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def kappa_scaling_a_minus_y_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = scaling_a_minus_y
    q_value = kappa_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

def limit_kappa_scaling_a_minus_y_use_wei_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = limit_scaling_a_minus_y
    q_value = kappa_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def limit_kappa_scaling_a_minus_y_use_foo_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = limit_scaling_a_minus_y
    q_value = kappa_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def limit_kappa_scaling_a_minus_y_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = limit_scaling_a_minus_y
    q_value = kappa_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

### kappa_scaling_y_slash_a_FQ_update

def kappa_scaling_y_slash_a_use_wei_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = scaling_y_slash_a
    q_value = kappa_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def kappa_scaling_y_slash_a_use_foo_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = scaling_y_slash_a
    q_value = kappa_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def kappa_scaling_y_slash_a_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = scaling_y_slash_a
    q_value = kappa_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

def limit_kappa_scaling_y_slash_a_use_wei_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = limit_scaling_y_slash_a
    q_value = kappa_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value


def limit_kappa_scaling_y_slash_a_use_foo_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = limit_scaling_y_slash_a
    q_value = kappa_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def limit_kappa_scaling_y_slash_a_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = limit_scaling_y_slash_a
    q_value = kappa_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

### alpha_scaling_y_FQ_update

def alpha_scaling_y_use_wei_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = scaling_y
    q_value = alpha_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def alpha_scaling_y_use_foo_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = scaling_y
    q_value = alpha_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def alpha_scaling_y_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = scaling_y
    q_value = alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

def limit_alpha_scaling_y_use_wei_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = limit_scaling_y
    q_value = alpha_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def limit_alpha_scaling_y_use_foo_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = limit_scaling_y
    q_value = alpha_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def limit_alpha_scaling_y_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = limit_scaling_y
    q_value = alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

### alpha_scaling_y_minus_a_FQ_update

def alpha_scaling_y_minus_a_use_wei_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = scaling_y_minus_a
    q_value = alpha_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def alpha_scaling_y_minus_a_use_foo_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = scaling_y_minus_a
    q_value = alpha_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def alpha_scaling_y_minus_a_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = scaling_y_minus_a
    q_value = alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

def limit_alpha_scaling_y_minus_a_use_wei_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = limit_scaling_y_minus_a
    q_value = alpha_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def limit_alpha_scaling_y_minus_a_use_foo_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = limit_scaling_y_minus_a
    q_value = alpha_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def limit_alpha_scaling_y_minus_a_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = limit_scaling_y_minus_a
    q_value = alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

### alpha_scaling_a_minus_y_FQ_update

def alpha_scaling_a_minus_y_use_wei_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = scaling_a_minus_y
    q_value = alpha_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def alpha_scaling_a_minus_y_use_foo_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = scaling_a_minus_y
    q_value = alpha_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def alpha_scaling_a_minus_y_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = scaling_a_minus_y
    q_value = alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

def limit_alpha_scaling_a_minus_y_use_wei_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = limit_scaling_a_minus_y
    q_value = alpha_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def limit_alpha_scaling_a_minus_y_use_foo_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = limit_scaling_a_minus_y
    q_value = alpha_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def limit_alpha_scaling_a_minus_y_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = limit_scaling_a_minus_y
    q_value = alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

### alpha_scaling_y_slash_a_FQ_update

def alpha_scaling_y_slash_a_use_wei_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = scaling_y_slash_a
    q_value = alpha_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def alpha_scaling_y_slash_a_use_foo_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = scaling_y_slash_a
    q_value = alpha_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def alpha_scaling_y_slash_a_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = scaling_y_slash_a
    q_value = alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

def limit_alpha_scaling_y_slash_a_use_wei_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["weight"][other_dict["i"]]
    scaling_func = limit_scaling_y_slash_a
    q_value = alpha_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def limit_alpha_scaling_y_slash_a_use_foo_FQ_update(q_value, action, reward, params_dict, other_dict):
    y = other_dict["num_fooding"][other_dict["i"]]
    scaling_func = limit_scaling_y_slash_a
    q_value = alpha_scaling_base_FQ_update(q_value, action, reward, params_dict, scaling_func, y)
    return q_value

def limit_alpha_scaling_y_slash_a_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = limit_scaling_y_slash_a
    q_value = alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value


################################### 上方向と下方向へのスケーリングアップデート実験

def posi_update_alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = alpha_1
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]

    for i in range(NUM_ACTION):
        if i == action:
            scale = scaling_func(q_value[i], params_dict)
            scaled_alpha_1 = min(alpha_1 * scale, 1)

            if reward == 1:
                td_error = kappa_1 - q_value[i]
            else:
                td_error = - kappa_2 - q_value[i]

            if td_error >= 0:
                q_value[i] = q_value[i] + scaled_alpha_1 * td_error
            else:
                q_value[i] = q_value[i] + alpha_1 * td_error
        else:
            q_value[i] = (1 - alpha_2) * q_value[i]
    return q_value

def nega_update_alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = alpha_1
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]

    for i in range(NUM_ACTION):
        if i == action:
            scale = scaling_func(q_value[i], params_dict)
            scaled_alpha_1 = min(alpha_1 * scale, 1)

            if reward == 1:
                td_error = kappa_1 - q_value[i]
            else:
                td_error = - kappa_2 - q_value[i]

            if td_error >= 0:
                q_value[i] = q_value[i] + alpha_1 * td_error
            else:
                q_value[i] = q_value[i] + scaled_alpha_1 * td_error
        else:
            q_value[i] = (1 - alpha_2) * q_value[i]
    return q_value

def posi_update_alpha_scaling_y_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = scaling_y
    q_value = posi_update_alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

def nega_update_alpha_scaling_y_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = scaling_y
    q_value = nega_update_alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

def posi_update_alpha_scaling_y_minus_a_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = scaling_y_minus_a
    q_value = posi_update_alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

def nega_update_alpha_scaling_y_minus_a_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = scaling_y_minus_a
    q_value = nega_update_alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

def posi_update_alpha_scaling_a_minus_y_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = scaling_a_minus_y
    q_value = posi_update_alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

def nega_update_alpha_scaling_a_minus_y_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = scaling_a_minus_y
    q_value = nega_update_alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

def posi_update_alpha_scaling_y_slash_a_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = scaling_y_slash_a
    q_value = posi_update_alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

def nega_update_alpha_scaling_y_slash_a_use_Qi_FQ_update(q_value, action, reward, params_dict, other_dict):
    scaling_func = scaling_y_slash_a
    q_value = nega_update_alpha_scaling_base_use_Qi_FQ_update(q_value, action, reward, params_dict, scaling_func)
    return q_value

# TD error を強める

def power_TD_error_FQ_update(q_value, action, reward, params_dict, other_dict):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = alpha_1
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]
    b = params_dict["b"]

    for i in range(NUM_ACTION):
        if i == action:
            if reward == 1:
                td_error = kappa_1 - q_value[i]
            else:
                td_error = -kappa_2 - q_value[i]
            if td_error >= 0:
                td_error = min(max(td_error, 1e-3), 1e6)**b
            else:
                td_error = -min(max(-td_error, 1e-3), 1e6)**b
            q_value[i] = q_value[i] + alpha_1 * td_error
        else:
            q_value[i] = (1 - alpha_2) * q_value[i]
    return q_value

def posi_update_power_TD_error_FQ_update(q_value, action, reward, params_dict, other_dict):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = alpha_1
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]
    b = params_dict["b"]

    for i in range(NUM_ACTION):
        if i == action:
            if reward == 1:
                td_error = kappa_1 - q_value[i]
            else:
                td_error = -kappa_2 - q_value[i]
            if td_error >= 0:
                td_error = min(max(td_error, 1e-3), 1e6)
                td_error = td_error**b
            else:
                td_error = td_error
            q_value[i] = q_value[i] + alpha_1 * td_error
        else:
            q_value[i] = (1 - alpha_2) * q_value[i]
    return q_value

def nega_update_power_TD_error_FQ_update(q_value, action, reward, params_dict, other_dict):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = alpha_1
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]
    b = params_dict["b"]

    for i in range(NUM_ACTION):
        if i == action:
            if reward == 1:
                td_error = kappa_1 - q_value[i]
            else:
                td_error = -kappa_2 - q_value[i]
            if td_error >= 0:
                td_error = td_error
            else:
                td_error = -min(max(-td_error, 1e-3), 1e6)**b
            q_value[i] = q_value[i] + alpha_1 * td_error
        else:
            q_value[i] = (1 - alpha_2) * q_value[i]
    return q_value

def posi_nega_independence_update_power_TD_error_FQ_update(q_value, action, reward, params_dict, other_dict):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = alpha_1
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]
    a = params_dict["a"]
    b = params_dict["b"]

    for i in range(NUM_ACTION):
        if i == action:
            if reward == 1:
                td_error = kappa_1 - q_value[i]
            else:
                td_error = -kappa_2 - q_value[i]
            if td_error >= 0:
                td_error = min(max(td_error, 1e-3), 1e6)**a
            else:
                td_error = -min(max(-td_error, 1e-3), 1e6)**b
            q_value[i] = q_value[i] + alpha_1 * td_error
        else:
            q_value[i] = (1 - alpha_2) * q_value[i]
    return q_value

def posi_update_power_TD_error_plus_one_FQ_update(q_value, action, reward, params_dict, other_dict):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = alpha_1
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]
    b = params_dict["b"]

    for i in range(NUM_ACTION):
        if i == action:
            if reward == 1:
                td_error = kappa_1 - q_value[i]
            else:
                td_error = -kappa_2 - q_value[i]
            if td_error >= 0:
                td_error = min(max(td_error, 1e-3), 1e6)
                td_error = (td_error + 1)**b - 1
            else:
                td_error = td_error
            q_value[i] = q_value[i] + alpha_1 * td_error
        else:
            q_value[i] = (1 - alpha_2) * q_value[i]
    return q_value

def nega_update_power_TD_error_plus_one_FQ_update(q_value, action, reward, params_dict, other_dict):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = alpha_1
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]
    b = params_dict["b"]

    for i in range(NUM_ACTION):
        if i == action:
            if reward == 1:
                td_error = kappa_1 - q_value[i]
            else:
                td_error = -kappa_2 - q_value[i]
            if td_error >= 0:
                td_error = min(max(td_error, 1e-3), 1e6)
                td_error = td_error
            else:
                td_error = -((-td_error + 1)**b - 1)
            q_value[i] = q_value[i] + alpha_1 * td_error
        else:
            q_value[i] = (1 - alpha_2) * q_value[i]
    return q_value

def posi_update_multi_TD_error_FQ_update(q_value, action, reward, params_dict, other_dict):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = alpha_1
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]
    b = params_dict["b"]

    for i in range(NUM_ACTION):
        if i == action:
            if reward == 1:
                td_error = kappa_1 - q_value[i]
            else:
                td_error = -kappa_2 - q_value[i]
            if td_error > 0:
                td_error = min(max(td_error, 1e-3), 1e6)
                td_error = td_error * b
            else:
                td_error = td_error
            q_value[i] = q_value[i] + alpha_1 * td_error
        else:
            q_value[i] = (1 - alpha_2) * q_value[i]
    return q_value

def nega_update_multi_TD_error_FQ_update(q_value, action, reward, params_dict, other_dict):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = alpha_1
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]
    b = params_dict["b"]

    for i in range(NUM_ACTION):
        if i == action:
            if reward == 1:
                td_error = kappa_1 - q_value[i]
            else:
                td_error = -kappa_2 - q_value[i]
            if td_error > 0:
                td_error = min(max(td_error, 1e-3), 1e6)
                td_error = td_error
            else:
                td_error = td_error * b
            q_value[i] = q_value[i] + alpha_1 * td_error
        else:
            q_value[i] = (1 - alpha_2) * q_value[i]
    return q_value

def alpha_nega_posi_TD_error_DFQ_update(q_value, action, reward, params_dict, other_dict):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = params_dict["alpha_2"]
    alpha_3 = params_dict["alpha_3"]
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]

    for i in range(NUM_ACTION):
        if i == action:
            if reward == 1:
                td_error = kappa_1 - q_value[i]
            else:
                td_error = -kappa_2 - q_value[i]
            if td_error >= 0:
                q_value[i] = q_value[i] + alpha_1 * td_error
            else:
                q_value[i] = q_value[i] + alpha_2 * td_error

        else:
            q_value[i] = (1 - alpha_3) * q_value[i]
    return q_value

def alpha_nega_posi_TD_error_FQ_update(q_value, action, reward, params_dict, other_dict):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = params_dict["alpha_2"]
    alpha_3 = alpha_2
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]

    for i in range(NUM_ACTION):
        if i == action:
            if reward == 1:
                td_error = kappa_1 - q_value[i]
            else:
                td_error = - kappa_2 - q_value[i]
            if td_error >= 0:
                q_value[i] = q_value[i] + alpha_1 * td_error
            else:
                q_value[i] = q_value[i] + alpha_2 * td_error

        else:
            q_value[i] = (1 - alpha_3) * q_value[i]
    return q_value


### parallel

def parallel_Q_update(q_values, actions, rewards, params_dict, other_dict):
    alpha_1 = params_dict["alpha_1"]
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]
    select_onehot = np.eye(NUM_ACTION)[actions]
    reward_onehot = rewards[:, None]
    q_values = q_values + select_onehot * reward_onehot * alpha_1 * (kappa_1 - q_values) \
                    + select_onehot * (1 - reward_onehot) * alpha_1 * (-kappa_2 - q_values)
    return q_values

def parallel_FQ_update(q_values, actions, rewards, params_dict, other_dict):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = alpha_1
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]

    select_onehot = np.eye(NUM_ACTION)[actions]
    reward_onehot = rewards[:, None]

    q_values = q_values + select_onehot * reward_onehot * alpha_1 * (kappa_1 - q_values) \
                    + select_onehot * (1 - reward_onehot) * alpha_1 * (-kappa_2 - q_values) \
                    + (1 - select_onehot) * alpha_2 * (0 - q_values)

    return q_values

def parallel_DFQ_update(q_values, actions, rewards, params_dict, other_dict):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = params_dict["alpha_2"]
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]

    select_onehot = np.eye(NUM_ACTION)[actions]
    reward_onehot = rewards[:, None]

    q_values = q_values + select_onehot * reward_onehot * alpha_1 * (kappa_1 - q_values) \
                    + select_onehot * (1 - reward_onehot) * alpha_1 * (-kappa_2 - q_values) \
                    + (1 - select_onehot) * alpha_2 * (0 - q_values)

    return q_values

def parallel_alpha_nega_posi_TD_error_DFQ_update(q_values, actions, rewards, params_dict, other_dict):
    alpha_1 = params_dict["alpha_1"]
    alpha_2 = params_dict["alpha_2"]
    alpha_3 = params_dict["alpha_3"]
    kappa_1 = params_dict["kappa_1"]
    kappa_2 = params_dict["kappa_2"]

    select_onehot = np.eye(NUM_ACTION)[actions]
    reward_onehot = rewards[:, None]

    select_delta_q = select_onehot * reward_onehot * (kappa_1 - q_values) \
                     + select_onehot * (1 - reward_onehot) * (-kappa_2 - q_values)
    select_delta_q[select_delta_q >= 0] *= alpha_1
    select_delta_q[select_delta_q < 0] *= alpha_2
    q_values = q_values + select_delta_q + (1 - select_onehot) * alpha_3 * (0 - q_values)

    # q_values = q_values + select_onehot * reward_onehot * alpha_1 * (kappa_1 - q_values) \
    #                 + select_onehot * (1 - reward_onehot) * alpha_2 * (-kappa_2 - q_values) \
    #                 + (1 - select_onehot) * alpha_3 * (0 - q_values)

    return q_values
