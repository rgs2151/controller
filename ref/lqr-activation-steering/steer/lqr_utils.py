import torch as th
import matplotlib.pyplot as plt
import os
import gc

def print_curr_mem(msg):
    print("======================================================================\n======================================================================")
    print(msg)
    gc.collect()
    if th.cuda.is_available():
        device_id = th.cuda.current_device()

        # Print allocated memory (currently used by tensors)
        print(f"th.cuda.memory_allocated: {th.cuda.memory_allocated(device_id)/1024**3:.3f}GB")
        
        # Print reserved memory (allocated by PyTorch's internal memory manager, including cached free blocks)
        print(f"th.cuda.memory_reserved: {th.cuda.memory_reserved(device_id)/1024**3:.3f}GB")
        
        # Print reserved memory (allocated by PyTorch's internal memory manager, including cached free blocks)
        print(f"th.cuda.max_memory_allocated: {th.cuda.max_memory_allocated(device_id)/1024**3:.3f}GB")

        # Print peak memory usage during the current process lifetime
        print(f"th.cuda.max_memory_reserved: {th.cuda.max_memory_reserved(device_id)/1024**3:.3f}GB")

        # Optional: Clear the memory cache (can make `nvidia-smi` report lower usage, but doesn't affect PyTorch's ability to allocate new tensors)
        th.cuda.empty_cache() 
    else:
        print("CUDA not available")
    print("======================================================================\n======================================================================")


def linearize(tfs, T, m, X_nom):
    """
    Linearize nonlinear dynamics f around nominal trajectory (X_nom, U_nom=0).

    Args:
        f: dynamics function f(x,u) -> x_next
        X_nom: nominal states (T+1, k, n)
        U_nom: nominal controls (T, m)

    Returns:
        A: linearized A matrices (T, n, n)
        B: linearized B matrices (T, n, m)
    """
    U_nom = th.zeros([T, m], device=X_nom.device)
    n = X_nom.shape[-1]
    # print(X_nom.shape)

    A = th.zeros((T, n, n), dtype=X_nom.dtype, device=X_nom.device)
    # B = th.zeros((T, n, m), dtype=X_nom.dtype, device=X_nom.device)

    for t in range(T):
        x = X_nom[t].detach().requires_grad_(True)
        u = U_nom[t].detach().requires_grad_(True)

        def f_last(x, u):
            return tfs[t](x,u)[..., -1, :]

        # Compute Jacobians:
        Jx = th.autograd.functional.jacobian(lambda x_: f_last(x_, u), x, create_graph=False, vectorize=True)   # shape: [n, *x.shape]
        # Ju = th.autograd.functional.jacobian(lambda u_: f_last(x, u_), u, create_graph=False, vectorize=True)   # shape: [n, *u.shape]
        A[t] = Jx[...,-1,:]
        # B[t] = Ju

    return A

def linearize_jvp_streamed_gpu(tfs, T, m, X_nom, A_out):
    """
    Streamed forward-mode Jacobian computation (column-by-column),
    writing directly into a preallocated GPU tensor.

    Args:
        tfs: list of dynamics functions
        T: time horizon
        m: control dim
        X_nom: (T+1, ..., n)
        A_out: (T, n, n) PREALLOCATED on GPU

    Returns:
        None (writes into A_out in-place)
    """
    device = X_nom.device
    dtype = X_nom.dtype
    n = X_nom.shape[-1]

    U_nom = th.zeros((T, m), device=device, dtype=dtype)


    print_curr_mem("before internal loop")


    for t in range(T):
        x = X_nom[t].detach()
        u = U_nom[t].detach()

        def f_last(x_):
            return tfs[t](x_, u)[..., -1, :]  # (..., n)

        for j in range(n):
            v = th.zeros_like(x)
            v[..., j] = 1.0

            # JVP: returns (f(x), J @ v)
            _, jvp_out = th.autograd.functional.jvp(
                f_last,
                (x,),
                (v,),
                create_graph=False,
                strict=False,
            )

            A_out[t, :, j] = jvp_out

            del v, jvp_out



        del x, u

        print_curr_mem("in internal loop")



def time_varying_lqr_noB(A, Q, R, S_T):
    """
    Solve the time-varying LQR problem given linearized dynamics.

    Args:
        A: (T, n, n)
        B: (T, n, m)
        Q: (T, n, n)
        R: (T, m, m)
        Qf: (n, n)

    Returns:
        K: (T, m, n) feedback gains
    """
    T, n, m = A.shape

    S = th.zeros((T+1, n, n), dtype=A.dtype, device=A.device)
    K = th.zeros((T, m, n), dtype=A.dtype, device=A.device)


    S[T] = S_T

    for t in reversed(range(T)):
        At = A[t]
        Qt = Q[t]
        Rt = R[t]

        P = (S[t+1] + Rt).to(A.device) # = BkT Sk+1 Bk + Rk
        F = (S[t+1] @ At).to(A.device) # = BkT Sk+1 Ak  
        G = (Qt + At.transpose(-2, -1) @ S[t+1] @ At).to(A.device) # = Ak^T Sk+1 Ak + Qk

        P_inv = th.linalg.inv(P)
        K[t] = P_inv @ F

        S[t] = G - F.transpose(-2, -1) @ P_inv @ F

    return K

def time_varying_lqr_noB_mem_efficient(A, Q, R, S_T):
    """
    Solve the time-varying LQR problem given linearized dynamics.

    Args:
        A: (T, n, n)
        B: (T, n, m)
        Q: (T, n, n)
        R: (T, m, m)
        Qf: (n, n)

    Returns:
        K: (T, m, n) feedback gains
    """
    T, n, m = A.shape

    S = th.zeros((T+1, n, n), dtype=A.dtype)
    K = th.zeros((T, m, n), dtype=A.dtype, device=A.device)


    S[T] = S_T

    for t in reversed(range(T)):
        At = A[t]
        Qt = Q[t]
        Rt = R[t]
        Stp1 = S[t+1].to(A.device)

        P = (Stp1 + Rt).to(A.device) # = BkT Sk+1 Bk + Rk
        F = (Stp1 @ At).to(A.device) # = BkT Sk+1 Ak  
        G = (Qt + At.transpose(-2, -1) @ Stp1 @ At).to(A.device) # = Ak^T Sk+1 Ak + Qk

        K[t] = th.linalg.solve(P, F)

        S[t] = (G - F.transpose(-2, -1) @ th.linalg.solve(P, F)).to('cpu')

    return K


def time_varying_lqr(A, B, Q, R, S_T):
    """
    Solve the time-varying LQR problem given linearized dynamics.

    Args:
        A: (T, n, n)
        B: (T, n, m)
        Q: (T, n, n)
        R: (T, m, m)
        Qf: (n, n)

    Returns:
        K: (T, m, n) feedback gains
    """
    T, n, m = B.shape

    S = th.zeros((T+1, n, n), dtype=A.dtype, device=A.device)
    K = th.zeros((T, m, n), dtype=A.dtype, device=A.device)


    S[T] = S_T

    for t in reversed(range(T)):
        At = A[t]
        Bt = B[t]
        Qt = Q[t]
        Rt = R[t]

        # Sk = Ak^T [Sk+1 − Sk+1 Bk (BkT Sk+1 Bk + Rk)^-1 BkT Sk+1]Ak + Qk
        P = (Bt.transpose(-2, -1) @ S[t+1] @ Bt + Rt).to(A.device) # = BkT Sk+1 Bk + Rk
        F = (Bt.transpose(-2, -1) @ S[t+1] @ At).to(A.device) # = BkT Sk+1 Ak  
        G = (Qt + At.transpose(-2, -1) @ S[t+1] @ At).to(A.device) # = Ak^T Sk+1 Ak + Qk

        P_inv = th.linalg.inv(P)
        K[t] = P_inv @ F

        S[t] = G - F.transpose(-2, -1) @ P_inv @ F

    return K

def transformerBlockControl(tf, x, u):
    # print(f"if ousdfdsdfd: {x.shape}")
    x_next = tf(x)
    # x_next[:,-1,:] = x_next[:,-1,:] + u # 4.40.2
    x_next[...,-1,:] = x_next[...,-1,:] + u
    return x_next


def find_random_target(model, x0):
    x = x0
    for block in model.blocks:
        x = block(x)
    return x


def tf_block_wrapper(block, attention_mask, position_ids, position_embeddings, x): # 4.57
    x = x.unsqueeze(0)
    return block(x, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]