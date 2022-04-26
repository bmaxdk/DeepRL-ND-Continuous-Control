import numpy as np

# projection function based on:
# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter17/06_train_d4pg.py
def distr_projection(next_distr_v, rewards_v, dones_mask_t, gamma, Vmin, Vmax, Natoms):
    Delta_Z = (Vmax - Vmin) / (Natoms - 1)
    next_distr = next_distr_v.data.cpu().numpy()
    rewards = rewards_v.data.cpu().numpy()
    rewards = np.squeeze(rewards)
    dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
    # dones_mask = np.squeeze(dones_mask)
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, Natoms), dtype=np.float32)
    
    for atom in range(Natoms):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * Delta_Z) * gamma))
        b_j = (tz_j - Vmin) / Delta_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        # eq_mask = np.squeeze(eq_mask)
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        # ne_mask = np.squeeze(ne_mask)
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
        
    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones_mask]))
        b_j = (tz_j - Vmin) / Delta_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        # eq_mask = np.squeeze(eq_mask)
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        # ne_mask = np.squeeze(ne_mask)
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr