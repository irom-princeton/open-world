import math
import torch

@torch.no_grad()
def create_targets_shortcut(latents, num_history, action_hidden, condition_latent,
                            added_time_ids,
                            model_predict_v, 
                            labels=None, bootstrap_cfg=False, 
                         bootstrap_bs=1, DENOISE_TIMESTEPS=128):
    """
    latents: (B, F, C, H, W) where first num_history frames are conditioning history
    DENOISE_TIMESTEPS: total denoising steps (e.g., 128
    """
    # breakpoint()
    device = latents.device
    B, F, C, H, W = latents.shape
    
    min_bootstrap_bs = 1 # ensure at least one bootstrap sample
    if min_bootstrap_bs > 0:
        bootstrap_bs = max(bootstrap_bs, min_bootstrap_bs)
    bootstrap_bs = min(bootstrap_bs, B)  # cap

    # schedule sections
    log2_sections = int(math.log2(DENOISE_TIMESTEPS))
    
    # common "section values": dt_base in {log2_sections-1, ..., 0}
    section_vals = (log2_sections - 1) - torch.arange(log2_sections, device=device)  # [.., 2,1,0]
    
    # --- 1) dt_base & dt for bootstrap portion (like your shortcut code)
    idx = torch.randint(0, log2_sections, (bootstrap_bs,), device=device)
    dt_base = section_vals[idx].to(torch.int64)           
    
    # dt_base = torch.repeat_interleave(
    #     (log2_sections - 1) - torch.arange(log2_sections, device=device),
    #     bootstrap_bs // log2_sections
    # )
    # dt_base = torch.cat([dt_base, torch.zeros(bootstrap_bs - dt_base.shape[0], device=device)])
    # dt_base = dt_base.to(torch.int64)

    dt = 1 / (2 ** dt_base.float())                # (bootstrap_bs,)
    dt_base_bootstrap = dt_base + 1
    dt_bootstrap = dt / 2                          # (bootstrap_bs,)

    # --- 2) sample t on a grid depending on dt_base (same trick)
    dt_sections = (2 ** dt_base).to(torch.float32)  # (bootstrap_bs,)
    t = torch.cat([torch.randint(0, int(v.item()), (1,), device=device).float() for v in dt_sections], dim=0)
    t = t / dt_sections                              # (bootstrap_bs,)
    t_full = t[:, None, None, None, None]            # (bootstrap_bs,1,1,1,1)

    if not bootstrap_cfg:
        # --- 3) build x_t on future frames only (history frames are copied from real latents)
        x1_future = latents[:bootstrap_bs, num_history:]             # (bs, Ff, C,H,W)
        x0_future = torch.randn_like(x1_future)                      # (bs, Ff, C,H,W)

        x_t_future = (1 - (1 - 1e-5) * t_full) * x0_future + t_full * x1_future

        # full input tensor to model: keep history frames as-is (or you can feed noisy_history externally)
        x_t_full = latents[:bootstrap_bs].clone()
        x_t_full[:, num_history:] = x_t_future

        # --- 4) bootstrap target via two half-steps (v_b1,v_b2 averaged)
        
        v_b1 = model_predict_v(x_t_full, t, dt_base_bootstrap,
                                action_hidden=action_hidden[:bootstrap_bs],
                                condition_latent=condition_latent[:bootstrap_bs],
                                added_time_ids=added_time_ids[:bootstrap_bs],
                                num_history=num_history) 

        t2 = t + dt_bootstrap # in page, this is t + 2d
        # evolve x_t_future with v_b1 (only future frames)
        x_t2_full = x_t_full.clone()
        x_t2_full[:, num_history:] = torch.clamp(
            x_t_future + dt_bootstrap[:, None, None, None, None] * v_b1,
            -4, 4
        )

       
        v_b2 = model_predict_v(x_t2_full, t2, dt_base_bootstrap, 
                                action_hidden=action_hidden[:bootstrap_bs],
                                condition_latent=condition_latent[:bootstrap_bs],
                                added_time_ids=added_time_ids[:bootstrap_bs],
                                num_history=num_history)
    else:
        # TODO: implement classifier-free guidance version of bootstrap targets
        pass
    
    v_bootstrap = torch.clamp((v_b1 + v_b2) / 2, -4, 4)  # (bs, Ff, C,H,W)

    # --- 5) flow-matching targets for the rest of the batch (dt_base = log2(T))
    t_flow = torch.randint(0, DENOISE_TIMESTEPS, (B,), device=device).float() / DENOISE_TIMESTEPS
    t_flow_full = t_flow[:, None, None, None, None]

    x1_all_future = latents[:, num_history:]
    x0_all_future = torch.randn_like(x1_all_future)

    x_t_all_future = (1 - (1 - 1e-5) * t_flow_full) * x0_all_future + t_flow_full * x1_all_future
    v_flow_future = x1_all_future - (1 - 1e-5) * x0_all_future

    dt_flow = int(math.log2(DENOISE_TIMESTEPS))
    dt_base_flow = torch.full((B,), dt_flow, device=device, dtype=torch.int64)

    # --- 6) merge bootstrap + flow (same layout as your shortcut code)
    bst_size = bootstrap_bs
    data_size = B - bst_size

    x_t_merged = latents.clone()
    x_t_merged[:bst_size] = x_t_full
    x_t_merged[bst_size:, num_history:] = x_t_all_future[:data_size]

    t_merged = torch.cat([t, t_flow[:data_size]], dim=0)
    dt_base_merged = torch.cat([dt_base, dt_base_flow[:data_size]], dim=0)
    v_target_merged = torch.cat([v_bootstrap, v_flow_future[:data_size]], dim=0)

    return x_t_merged, v_target_merged, t_merged, dt_base_merged


@torch.no_grad()
def create_targets(latents, num_history, action_hidden, condition_latent,
                            added_time_ids,
                            model_predict_v, 
                            labels=None, bootstrap_cfg=False, 
                         bootstrap_bs=1, DENOISE_TIMESTEPS=128):
    """
    latents: (B, F, C, H, W) where first num_history frames are conditioning history
    DENOISE_TIMESTEPS: total denoising steps (e.g., 128
    """
    device = latents.device
    B, F, C, H, W = latents.shape
    Ff = F - num_history
    assert Ff > 0, "num_history must be < F"
    

    # Sample t on the discrete grid like your code (matches your scheduler-like convention)
    t = torch.randint(0, DENOISE_TIMESTEPS, (B,), device=device).float() / float(DENOISE_TIMESTEPS)
    t_full = t[:, None, None, None, None]  # (B,1,1,1,1)

    # Future frames: x1 is data, x0 is noise
    x1_future = latents[:, num_history:]                 # (B, Ff, C,H,W)
    x0_future = torch.randn_like(x1_future)              # (B, Ff, C,H,W)

    eps_t = 1e-5
    
    # Linear path (slightly modified with eps_t)
    x_t_future = (1 - (1 - eps_t) * t_full) * x0_future + t_full * x1_future

    # Constant velocity for this path
    v_target_future = x1_future - (1 - eps_t) * x0_future

    # Build merged x_t: keep history as-is, replace future with x_t
    x_t_merged = latents.clone()
    x_t_merged[:, num_history:] = x_t_future

    # Keep dt_base for API compatibility (if your model takes it)
    dt_base = int(math.log2(DENOISE_TIMESTEPS))
    dt_base_merged = torch.full((B,), dt_base, device=device, dtype=torch.int64)

    return x_t_merged, v_target_future, t, dt_base_merged


@torch.no_grad()
def create_targets_flow_matching(latents, num_history, return_flow_map_dt=False):
    """
    latents: (B, F, C, H, W) where first num_history frames are conditioning history
    """
    device = latents.device
    B, F, C, H, W = latents.shape
    Ff = F - num_history
    assert Ff > 0, "num_history must be < F"
    
    # Sample t from uniform [0,1]
    t = torch.rand(B, device=device)
    t_full = t[:, None, None, None, None]  # (B,1,1,1,1)

    # Future frames: x1 is data, x0 is noise
    x1_future = latents[:, num_history:]                 # (B, Ff, C,H,W)
    x0_future = torch.randn_like(x1_future)              # (B, Ff, C,H,W)

    # Linear path (slightly modified with eps_t)
    x_t_future = (1 -  t_full) * x0_future + t_full * x1_future

    # Constant velocity for this path
    v_target_future = x1_future -  x0_future

    # Build merged x_t: keep history as-is, replace future with x_t
    x_t_merged = latents.clone()
    x_t_merged[:, num_history:] = x_t_future
    
    if return_flow_map_dt:
        dt_base = torch.zeros_like(t)
    else:
        dt_base = None
    
    return x_t_merged, v_target_future, t, dt_base



def sample_s_t(B, device):
    """
    Sample s,t values where 0 <= s <= t <= 1
    """
    u = torch.rand(B, device=device)
    v = torch.rand(B, device=device)
    s = torch.minimum(u, v)
    t = torch.maximum(u, v)

    return s, t



def create_targets_lsd(latents, num_history, action_hidden, condition_latent,
                            added_time_ids,
                            model_partial_t, 
                            bootstrap_bs=1):
    device = latents.device
    B, F, C, H, W = latents.shape
    Ff = F - num_history
    assert Ff > 0, "num_history must be < F"
    
    min_bootstrap_bs = 1 # ensure at least one bootstrap sample
    if min_bootstrap_bs > 0:
        bootstrap_bs = max(bootstrap_bs, min_bootstrap_bs)
    bootstrap_bs = min(bootstrap_bs, B)  # cap
    
    bst_size = bootstrap_bs
    data_size = B - bst_size
    
    # --- 1) flow map targets for bootstrap portion
    # sample the s, t values, where 0 <= s <= t <= 1
    s, t = sample_s_t(bootstrap_bs, device)
    s_full = s[:, None, None, None, None]
    t_full = t[:, None, None, None, None]
    
    x1_future = latents[:bootstrap_bs, num_history:]  # (bs, Ff, C,H,W)
    x0_future = torch.randn_like(x1_future)      
    
    Is_future = s_full * x1_future + (1 - s_full) * x0_future
    
    Is_full = latents[:bootstrap_bs].clone()
    Is_full[:, num_history:] = Is_future
    
    # TODO: support cfg version if needed
    with torch.enable_grad():
        Xst_Is, dt_Xst = model_partial_t(s, t, Is_full, 
                                    action_hidden=action_hidden[:bootstrap_bs],
                                    condition_latent=condition_latent[:bootstrap_bs],
                                    added_time_ids=added_time_ids[:bootstrap_bs],
                                    num_history=num_history)
    
    # detach targets
    Xst_Is = Xst_Is.detach()
    dt_Xst = dt_Xst.detach()
    
    # --- 2) flow-matching targets for the rest of the batch (dt_base = log2(T))
    with torch.no_grad():
        if data_size > 0:
            t_flow = torch.rand(data_size, device=device)
            t_flow_full = t_flow[:, None, None, None, None]

            x1_all_future = latents[bst_size:, num_history:]
            x0_all_future = torch.randn_like(x1_all_future)

            x_t_all_future = (1 - t_flow_full) * x0_all_future + t_flow_full * x1_all_future
            v_flow_future = x1_all_future - x0_all_future
        else:
            t_flow = None
    
    # --- 3) merge bootstrap + flow (same layout as your shortcut code)
   
    x_t_merged = latents.clone()
    v_target_merged = torch.empty((B, Ff, C, H, W), device=device, dtype=latents.dtype)
    
    # bootstrap rows
    x_t_merged[:bst_size, num_history:] = Xst_Is
    v_target_merged[:bst_size] = dt_Xst
    
    # flow rows
    if data_size > 0:
        x_t_merged[bst_size:, num_history:] = x_t_all_future
        v_target_merged[bst_size:] = v_flow_future
        t_merged = torch.cat([t, t_flow], dim=0)
    else:
        t_merged = t
    
    dt_merged = torch.zeros_like(t_merged)
    
    return x_t_merged, v_target_merged, t_merged, dt_merged


def debug_check(latents, v_target_merged, name):
    print(f"[{name}] grad_enabled:", torch.is_grad_enabled())
    print(f"[{name}] latents.requires_grad:", latents.requires_grad)
    print(f"[{name}] v_target_merged.requires_grad:", v_target_merged.requires_grad)


@torch.no_grad()
def create_targets_psd(latents, num_history, action_hidden, condition_latent,
                            added_time_ids,
                            model_predict_v, 
                            bootstrap_bs=1, psd_sample_mode="uniform",
                            bias_prob=-1,
                            ):
    
    device = latents.device
    B, F, C, H, W = latents.shape
    Ff = F - num_history
    assert Ff > 0, "num_history must be < F"
    
    min_bootstrap_bs = 1 # ensure at least one bootstrap sample
    if min_bootstrap_bs > 0:
        bootstrap_bs = max(bootstrap_bs, min_bootstrap_bs)
    bootstrap_bs = min(bootstrap_bs, B)  # cap
    
    bst_size = bootstrap_bs
    data_size = B - bst_size
    
    # sample s, u, t
    if bias_prob >= 0:
        do_bias_sample = (torch.rand(1).item() < bias_prob)
    else:
        do_bias_sample = False

    if not do_bias_sample:
        s, t = sample_s_t(bootstrap_bs, device)
    else:
        # let s to be as close to the 0, controlled using sigma_max and t be 1
        s = torch.ones(bootstrap_bs, device=device, dtype=latents.dtype) * (1/(1+700)) # starting from sigma_max
        t = torch.ones(bootstrap_bs, device=device, dtype=latents.dtype)
        
    if psd_sample_mode == "uniform":
        gamma = torch.rand(bootstrap_bs, device=device)
    elif psd_sample_mode == "midpoint":
        gamma = torch.full((bootstrap_bs,), 0.5, device=device)
    u = gamma * s + (1 - gamma) * t
    
    s_full = s[:, None, None, None, None]
    u_full = u[:, None, None, None, None]
    t_full = t[:, None, None, None, None]
    
    # from s to u
    
    x1_future = latents[:bootstrap_bs, num_history:]  # (bs, Ff, C,H,W)
    x0_future = torch.randn_like(x1_future)      
    
    Is_future = s_full * x1_future + (1 - s_full) * x0_future
    
    Is_full = latents[:bootstrap_bs].clone()
    Is_full[:, num_history:] = Is_future
    
    dt_s_u = u - s
    phi_su = model_predict_v(Is_full, s, dt_s_u,
                            action_hidden=action_hidden[:bootstrap_bs],
                            condition_latent=condition_latent[:bootstrap_bs],
                            added_time_ids=added_time_ids[:bootstrap_bs],
                            num_history=num_history)
    # evolve Is to Iu
    Iu_future = Is_full[:, num_history:] + dt_s_u[:, None, None, None, None] * phi_su
    Iu_full = Is_full.clone()
    Iu_full[:, num_history:] = Iu_future
    
    # from u to t
    dt_u_t = t - u
    phi_ut = model_predict_v(Iu_full, u, dt_u_t,
                            action_hidden=action_hidden[:bootstrap_bs],
                            condition_latent=condition_latent[:bootstrap_bs],
                            added_time_ids=added_time_ids[:bootstrap_bs],
                            num_history=num_history)
    # evolve Iu to It
    # It_future = Iu_full[:, num_history:] + dt_u_t[:, None, None, None, None] * phi_ut
    # It_full = Iu_full.clone()
    # It_full[:, num_history:] = It_future
    
    if psd_sample_mode == "uniform":
        v_psd = gamma[:, None, None, None, None] * phi_su + (1 - gamma)[:, None, None, None, None] * phi_ut
    elif psd_sample_mode == "midpoint":
        v_psd = 0.5 * (phi_su + phi_ut)
    
    # --- 2) flow-matching targets for the rest of the batch
    if data_size > 0:
        t_flow = torch.rand(data_size, device=device)
        t_flow_full = t_flow[:, None, None, None, None]

        x1_all_future = latents[bst_size:, num_history:]
        x0_all_future = torch.randn_like(x1_all_future)

        x_t_all_future = (1 - t_flow_full) * x0_all_future + t_flow_full * x1_all_future
        v_flow_future = x1_all_future - x0_all_future
    else:
        t_flow = None
    
    # --- 3) merge bootstrap + flow (same layout as your shortcut code)
    
    x_t_merged = latents.clone()
    v_target_merged = torch.empty((B, Ff, C, H, W), device=device, dtype=latents.dtype)
    
    # bootstrap rows
    x_t_merged[:bst_size, num_history:] = Is_future
    v_target_merged[:bst_size] = v_psd
    
    if data_size > 0:
        # flow rows
        x_t_merged[bst_size:, num_history:] = x_t_all_future
        v_target_merged[bst_size:] = v_flow_future
        t_merged = torch.cat([s, t_flow], dim=0)
        dt_merged = torch.cat([t-s, torch.zeros_like(t_flow)], dim=0)
    else:
        t_merged = s
        dt_merged = t - s
    
    # debug_check(latents, v_target_merged, "PSD")    
    return x_t_merged, v_target_merged, t_merged, dt_merged


@torch.no_grad()
def create_targets_one_step(latents, num_history, action_hidden, condition_latent,
                            added_time_ids,
                            model_predict_v, 
                            bootstrap_bs=1, psd_sample_mode="uniform"):
    
    device = latents.device
    B, F, C, H, W = latents.shape
    Ff = F - num_history
    assert Ff > 0, "num_history must be < F"
    
    min_bootstrap_bs = 1 # ensure at least one bootstrap sample
    if min_bootstrap_bs > 0:
        bootstrap_bs = max(bootstrap_bs, min_bootstrap_bs)
    bootstrap_bs = min(bootstrap_bs, B)  # cap
    
    bst_size = bootstrap_bs
    data_size = B - bst_size
    
    # get staring s and ending t
    s = torch.ones(bootstrap_bs, device=device, dtype=latents.dtype) * (1/(1+700)) # starting from sigma_max
    t = torch.ones(bootstrap_bs, device=device, dtype=latents.dtype)
    
    # s_full = s[:, None, None, None, None]
    # t_full = t[:, None, None, None, None]
    
    x1_future = latents[:bootstrap_bs, num_history:]  # (bs, Ff, C,H,W)
    x0_future = torch.randn_like(x1_future)
    
    Is_future = x0_future  # initialize
    
    v_one_step = (x1_future - x0_future) / (t - s)[:, None, None, None, None]
    
    # --- 2) flow-matching targets for the rest of the batch
    if data_size > 0:
        t_flow = torch.rand(data_size, device=device)
        t_flow_full = t_flow[:, None, None, None, None]

        x1_all_future = latents[bst_size:, num_history:]
        x0_all_future = torch.randn_like(x1_all_future)

        x_t_all_future = (1 - t_flow_full) * x0_all_future + t_flow_full * x1_all_future
        v_flow_future = x1_all_future - x0_all_future
    else:
        t_flow = None
    
    # --- 3) merge bootstrap + flow (same layout as your shortcut code)
    
    x_t_merged = latents.clone()
    v_target_merged = torch.empty((B, Ff, C, H, W), device=device, dtype=latents.dtype)
    
    # bootstrap rows
    x_t_merged[:bst_size, num_history:] = Is_future
    v_target_merged[:bst_size] = v_one_step
    
    if data_size > 0:
        # flow rows
        x_t_merged[bst_size:, num_history:] = x_t_all_future
        v_target_merged[bst_size:] = v_flow_future
        t_merged = torch.cat([s, t_flow], dim=0)
        dt_merged = torch.cat([t-s, torch.zeros_like(t_flow)], dim=0)
    else:
        t_merged = s
        dt_merged = t - s
    
    # debug_check(latents, v_target_merged, "ONE_STEP")
    return x_t_merged, v_target_merged, t_merged, dt_merged