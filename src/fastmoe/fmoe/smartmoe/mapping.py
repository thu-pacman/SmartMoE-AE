import torch

def is_valid_mapping(mapping, n):
    tmp = sorted(mapping)
    for i in range(n):
        if tmp[i] != i:
            return False

    return True

def dfs(avai, lst, cnt, num_expert, tot_expert, ori_S, now_S, rnk, tot_cost, f, C, pre_S):
    if(cnt == num_expert):
        if f[now_S][rnk] > tot_cost:
            f[now_S][rnk] =  tot_cost
            pre_S[now_S][rnk] = ori_S
        f[now_S][rnk] = min(f[now_S][rnk], tot_cost)
        return
    for idx in range(lst + 1, tot_expert - (num_expert - cnt - 1)):
        dfs(avai, idx, cnt + 1, num_expert, tot_expert, ori_S, now_S | (1 << idx), rnk, tot_cost + C[idx], f, C, pre_S)

def DP(num_expert, world_size, C):
    tot_expert = num_expert * world_size
    f = np.zeros((2**tot_expert, world_size + 1), dtype = float)
    pre_S = np.zeros((2**tot_expert, world_size + 1), dtype = int)
    for i in range(world_size + 1):
        for S in range(2**tot_expert):
            f[S][i] = 2.0**30

    ful = (1 << tot_expert) - 1
    f[0][0] = 0
    for i in range(world_size):
        for S in range(2**tot_expert):
            if f[S][i] != 2.0**30:
                avai = []
                for p in range(world_size):
                    if (S>>p)&1 == 0:
                        avai.append(p)
                dfs(avai, -1, 0, num_expert, tot_expert, S, S, i+1, f[S][i], f, C, pre_S)

    assert(f[ful][world_size] != 2.0**30)
    ret_mapping = [0 for _ in range(tot_expert)]

    now_S = ful
    for i in range(world_size, 0, -1):
        lst_S = pre_S[now_S][i]
        print(f"{now_S} -> {lst_S}")
        cur = now_S ^ lst_S
        cur_list = []
        for p in range(tot_expert):
            if (cur >> p) & 1 != 0:
                cur_list.append(p)
        assert(len(cur_list) == num_expert)
        cnt = 0
        for idx in cur_list:
            ret_mapping[idx] = (i - 1) * num_expert + cnt
            cnt += 1
        now_S = lst_S
    return ret_mapping

# 32 -> 8
def greedy_DP(num_expert, world_size, C):
    pre_res = greedy(4, 8, C)
    new_C = [0 for _ in range(8)]
    tot_expert = num_expert * world_size
    for i in range(tot_expert):
        dev = pre_res[i] // 4
        new_C[dev] += C[i]
    nxt_res = DP(8, world_size, C)
    ret = [0 for _ in range(tot_expert)]
    expert_cnt = [0 for _ in range(world_size)]
    for i in range(tot_expert):
        dev = pre_res[i] // 4
        dev_2 = nxt_res[dev]
        ret[i] = dev_2 * num_expert + expert_cnt[dev_2]
        expert_cnt[dev_2] += 1
    return ret

def greedy_gen_mapping_from_history(num_expert, world_size, cur_mapping, history_gate):
    tot_expert = num_expert * world_size

    cur_expert_tokens = [0 for _ in range(world_size)]
    for e in range(tot_expert):
        cur_expert_tokens[cur_mapping[e]//num_expert] += history_gate[e]
    
    cur_max = 0
    for i in range(world_size):
        cur_max = max(cur_max, cur_expert_tokens[i])

    expert_idx = list(history_gate.argsort())
    expert_idx.reverse()

    new_mapping = [-1 for _ in range(tot_expert)]

    new_max = 0
    expert_cnt = [0 for _ in range(world_size)]
    expert_tokens = [0 for _ in range(world_size)]
    for idx in expert_idx:
        old_pos = cur_mapping[idx] // num_expert

        cur_min = 0
        cur_pos = -1
        for i in range(0, world_size):
            if expert_cnt[i] < num_expert and (cur_pos == -1 or expert_tokens[i] < cur_min):
                cur_min = expert_tokens[i]
                cur_pos = i

        if expert_cnt[old_pos] < num_expert and expert_tokens[old_pos] == cur_min:
            cur_pos = old_pos
        
        new_location = cur_pos * num_expert + expert_cnt[cur_pos]
        expert_cnt[cur_pos] += 1
        expert_tokens[cur_pos] += history_gate[idx]
        new_mapping[idx] = new_location
        new_max = max(new_max, expert_tokens[cur_pos])

    if cur_max > new_max * 1.0:
        return new_mapping
    else:
        return cur_mapping


def generate_mapping_from_history(num_expert, world_size, cur_mapping, history_gate, method='Greedy'):
    # history_gate's index is generated from original gate, without mapping
    tot_expert = num_expert * world_size
    assert history_gate.shape[0] == tot_expert
    if method is None:
        return cur_mapping
    
    if method == 'Greedy':
        new_mapping = greedy_gen_mapping_from_history(num_expert, world_size, cur_mapping, history_gate)
    else:
        assert False, f"mapping generation method {method} not found."
    
    assert is_valid_mapping(new_mapping, tot_expert)
    return new_mapping 