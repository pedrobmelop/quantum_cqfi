import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings
import matplotlib.ticker as mticker
warnings.filterwarnings('ignore')

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)


class DrivenThermalQubit:
    """
    Driven two-level system in a thermal bath.
    
    H = ε(σ₊ + σ₋) in the rotating frame
    
    Jump operators:
        L₋ = √(Γ₀(n̄+1)) σ₋  (emission)
        L₊ = √(Γ₀n̄) σ₊      (absorption)
    """
    
    def __init__(self, epsilon: float = 0.15, Gamma0: float = 0.4, 
                 omega: float = 1.0, T: float = 0.8):
        self.epsilon = epsilon
        self.Gamma0 = Gamma0
        self.omega = omega
        self.T = T
        
        self.nbar = 1.0 / (np.exp(omega / T) - 1) if T > 0 else 0.0
        self.Gamma_m = Gamma0 * (self.nbar + 1)  # emission
        self.Gamma_p = Gamma0 * self.nbar         # absorption
        
        # Jump operators
        self.L_m = np.sqrt(self.Gamma_m) * sigma_minus
        self.L_p = np.sqrt(self.Gamma_p) * sigma_plus
        
        # Hamiltonian (RWA)
        self.H = epsilon * (sigma_plus + sigma_minus)
        
        # Effective non-Hermitian Hamiltonian
        self.H_eff = self.H - 0.5j * (self.L_m.conj().T @ self.L_m + 
                                       self.L_p.conj().T @ self.L_p)
    
    def dissipator(self, rho: np.ndarray) -> np.ndarray:
        """Lindblad dissipator D[ρ]"""
        D = np.zeros_like(rho)
        for L in [self.L_m, self.L_p]:
            Ld = L.conj().T
            D += L @ rho @ Ld - 0.5 * (Ld @ L @ rho + rho @ Ld @ L)
        return D
    
    def drho_dt(self, rho: np.ndarray) -> np.ndarray:
        """Master equation RHS: dρ/dt = -i[H,ρ] + D[ρ]"""
        return -1j * (self.H @ rho - rho @ self.H) + self.dissipator(rho)
    
    def evolve_rho(self, rho0: np.ndarray, times: np.ndarray) -> List[np.ndarray]:
        """Integrate master equation with RK4."""
        rho = rho0.copy()
        rho_list = [rho.copy()]
        
        for i in range(len(times) - 1):
            dt = times[i+1] - times[i]
            k1 = self.drho_dt(rho)
            k2 = self.drho_dt(rho + 0.5*dt*k1)
            k3 = self.drho_dt(rho + 0.5*dt*k2)
            k4 = self.drho_dt(rho + dt*k3)
            rho = rho + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            rho = 0.5 * (rho + rho.conj().T)
            rho = rho / np.trace(rho)
            rho_list.append(rho.copy())
        
        return rho_list
    
    def quantum_trajectory(self, psi0: np.ndarray, times: np.ndarray, 
                           seed: int = None) -> List[np.ndarray]:
        """Quantum jump unraveling."""
        if seed is not None:
            np.random.seed(seed)
        
        psi = psi0.copy() / np.linalg.norm(psi0)
        psi_list = [psi.copy()]
        
        for i in range(len(times) - 1):
            dt = times[i+1] - times[i]
            
            # Jump probabilities
            Lm_psi = self.L_m @ psi
            Lp_psi = self.L_p @ psi
            p_m = dt * np.real(np.vdot(Lm_psi, Lm_psi))
            p_p = dt * np.real(np.vdot(Lp_psi, Lp_psi))
            
            r = np.random.random()
            
            if r < p_m:
                psi = Lm_psi
            elif r < p_m + p_p:
                psi = Lp_psi
            else:
                psi = psi - 1j * dt * self.H_eff @ psi
            
            psi = psi / np.linalg.norm(psi)
            psi_list.append(psi.copy())
        
        return psi_list


def compute_sld(rho: np.ndarray, drho: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 1e-14)
    
    L = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            vi, vj = eigvecs[:, i], eigvecs[:, j]
            elem = np.vdot(vi, drho @ vj)
            denom = eigvals[i] + eigvals[j]
            if denom > 1e-13:
                L += (2 * elem / denom) * np.outer(vi, vj.conj())
    return L


def ensemble_qfi(rho: np.ndarray, L: np.ndarray) -> float:
    return np.real(np.trace(rho @ L @ L))


def ensemble_qfi_decomposition(rho: np.ndarray, drho: np.ndarray) -> Tuple[float, float]:
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 1e-14)
    
    dp = np.array([np.real(np.vdot(eigvecs[:, i], drho @ eigvecs[:, i])) for i in range(2)])
    F_IC = sum(dp[i]**2 / eigvals[i] for i in range(2) if eigvals[i] > 1e-13)
    
    F_C = 0.0
    for i in range(2):
        for j in range(2):
            if i != j and eigvals[i] + eigvals[j] > 1e-13:
                sigma = (eigvals[i] - eigvals[j])**2 / (eigvals[i] + eigvals[j])
                off = np.vdot(eigvecs[:, j], drho @ eigvecs[:, i])
                if abs(eigvals[i] - eigvals[j]) > 1e-13:
                    F_C += 2 * sigma * np.abs(off)**2 / (eigvals[i] - eigvals[j])**2
    return F_IC, F_C


def cqfi_direct(psi: np.ndarray, rho: np.ndarray, L: np.ndarray) -> float:
    psi = psi / np.linalg.norm(psi)
    p = np.real(np.vdot(psi, rho @ psi))
    if p < 1e-14:
        return 0.0
    L2_rho = L @ L @ rho
    return np.real(np.vdot(psi, L2_rho @ psi)) / p

def alt_cqfi_direct(psi: np.ndarray, rho: np.ndarray, L: np.ndarray) -> float:
    psi = psi / np.linalg.norm(psi)
    p = np.real(np.vdot(psi, rho @ psi))
    if p < 1e-14:
        return 0.0
    L2 = L @ L
    return np.real(np.vdot(psi, L2 @ psi))


def cqfi_decomposed(psi: np.ndarray, dpsi: np.ndarray, 
                    rho: np.ndarray, drho: np.ndarray) -> Tuple[float, float, float]:
    psi = psi / np.linalg.norm(psi)
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 1e-14)
    drho_eig = eigvecs.conj().T @ drho @ eigvecs
    dim = len(eigvals)
    L_eig = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            denom = eigvals[i] + eigvals[j]
            if denom > 1e-13:
                L_eig[i, j] = 2.0 * drho_eig[i, j] / denom
    D_eig = np.diag(np.diag(L_eig))
    O_eig = L_eig - D_eig
    c = eigvecs.conj().T @ psi
    v_D = D_eig @ c
    f_inc = np.real(np.vdot(v_D, v_D))
    v_O = O_eig @ c
    f_coh = np.real(np.vdot(v_O, v_O))
    f_cross = 2.0 * np.real(np.vdot(v_D, v_O))
    return f_inc, f_coh, f_cross

def numerical_derivative(arr: List[np.ndarray], t: np.ndarray, i: int) -> np.ndarray:
    if i == 0:
        return (arr[1] - arr[0]) / (t[1] - t[0])
    elif i == len(t) - 1:
        return (arr[-1] - arr[-2]) / (t[-1] - t[-2])
    return (arr[i+1] - arr[i-1]) / (t[i+1] - t[i-1])


def main():
    plt.rcParams.update({
        "font.family": "serif", 
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "mathtext.fontset": "stix",
        "mathtext.rm": "serif",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.5,
        "legend.frameon": True,
        "legend.framealpha":0.4,
        "figure.dpi": 300,
        "savefig.bbox": "tight"
    })
    
    # System
    system = DrivenThermalQubit(epsilon=0.15, Gamma0=0.4, omega=1.0, T=0.8)
    
    # Time evolution
    N_times = 1000
    tau = 10.0
    times = np.linspace(0, tau, N_times)
    dt = times[1] - times[0]

    # Initial state
    psi0 = np.array([1, 0], dtype=complex)
    rho0 = np.outer(psi0, psi0.conj())
    
    # Ensemble evolution
    rho_t = system.evolve_rho(rho0, times)
    
    # Ensemble QFI
    drho_t = [numerical_derivative(rho_t, times, i) for i in range(N_times)]
    L_t = [compute_sld(rho_t[i], drho_t[i]) for i in range(N_times)]
    
    F_Q = np.array([ensemble_qfi(rho_t[i], L_t[i]) for i in range(N_times)])
    F_IC_ens = np.zeros(N_times)
    F_C_ens = np.zeros(N_times)
    Length_F_Q = np.cumsum(np.sqrt(F_Q)) * dt
    Action_F_Q = np.cumsum(F_Q) * dt
    for i in range(N_times):
        F_IC_ens[i], F_C_ens[i] = ensemble_qfi_decomposition(rho_t[i], drho_t[i])

    # Trajectories
    # CHANGED: Reduced number of trajectories for VM performance
    N_traj = 50000 # User allowed less trajectories. Original was 50000.
    print(f"[3/3] Simulating {N_traj} quantum trajectories...")

    f_Q_all = np.zeros((N_traj, N_times))
    alt_f_Q_all = np.zeros((N_traj, N_times))
    f_IC_all = np.zeros((N_traj, N_times))
    f_C_all = np.zeros((N_traj, N_times))
    f_Cross_all = np.zeros((N_traj, N_times)) 
    stoch_length_f_Q = np.zeros((N_traj, N_times))
    stoch_action_f_Q = np.zeros((N_traj, N_times))

    for n in range(N_traj):
        if (n + 1) % 50 == 0:
            print(f"      {n+1}/{N_traj}")

        psi_t = system.quantum_trajectory(psi0, times, seed=n)
        dpsi_t = [numerical_derivative(psi_t, times, i) for i in range(N_times)]
        
        for i in range(N_times):
            f_Q_all[n, i] = cqfi_direct(psi_t[i], rho_t[i], L_t[i])
            alt_f_Q_all[n, i] = alt_cqfi_direct(psi_t[i], rho_t[i], L_t[i])
            f_IC_all[n, i], f_C_all[n, i], f_Cross_all[n,i] = cqfi_decomposed(
                psi_t[i], dpsi_t[i], rho_t[i], drho_t[i])
        stoch_length_f_Q[n,:] = np.cumsum(np.sqrt(f_Q_all[n,:])) * dt
        stoch_action_f_Q[n,:] = np.cumsum(f_Q_all[n,:]) * dt
    
    # Statistics
    f_Q_mean = np.mean(f_Q_all, axis=0)
    alt_f_Q_mean = np.mean(alt_f_Q_all, axis=0)
    f_Q_std = np.std(f_Q_all, axis=0)
    f_IC_mean = np.mean(f_IC_all, axis=0)
    f_C_mean = np.mean(f_C_all, axis=0)
    f_Cross_mean = np.mean(f_Cross_all, axis=0)
    stoch_length_mean = np.mean(stoch_length_f_Q,axis=0)
    stoch_action_mean = np.mean(stoch_action_f_Q,axis=0)
    
    # ========== PLOTTING FIGURE 2 ==========
    
    fig, axes = plt.subplots(2, 2, figsize=(6, 5), constrained_layout=True)
    fig.patch.set_facecolor('white')
    
    # Style
    c1, c2, c3, c4 = '#E63946', '#1D3557', '#457B9D', '#F4A261'
    
    # (a) Total CQFI vs Ensemble QFI
    ax = axes[0, 0]
    ax.plot(times, stoch_action_mean, color=c2, lw=2.5, label=r'$\langle j \rangle$')
    ax.plot(times, Action_F_Q, '--', color=c4, lw=2.5, label=r'$\mathcal{J}$')
    ax.set_xlabel(r'$t$', fontsize=13)
    ax.set_ylabel(r'$\mathcal{J}(t)$', fontsize=13)
    ax.text(0.8, 0.05, r'(a)', transform=ax.transAxes, fontsize=13) # Text inside
    # Legend above:
    ax.legend(fontsize=11, loc='center left', frameon=True)
    ax.set_xlim(-1, tau)
    ax.set_yscale('log')
    
    # (b) CQFI decomposition
    ax = axes[0, 1]
    ax.plot(times, Length_F_Q**2 / Action_F_Q, color=c1, lw=2.5, label=r'$\mathcal{L}^2 / \mathcal{J}$')
    ax.plot(times, times, color=c3, lw=2.5, label=r'$t$')
    ax.plot(times, stoch_length_mean**2 / stoch_action_mean, '--', color=c2, lw=2, label=r'$\langle l \rangle^2 / \mathcal{J}$ ')
    ax.set_xlabel(r'$t$', fontsize=13)
    ax.set_ylabel(r'$\mathcal{L}^2 / \mathcal{J}$', fontsize=13)
    ax.text(0.05, 0.8, r'(b)', transform=ax.transAxes, fontsize=13) # Text inside
    # Legend above:
    ax.legend(fontsize=11, loc='center left', frameon=True)
    ax.set_xlim(-1, tau)
    ax.set_yscale('log')
    
    # (c) Modified Plot: Square of f_Cross_mean vs number of trajectories
    ax = axes[1, 0]
    
    # Computing the running mean of f_Cross for each time step
    # shape: (N_traj, N_times)
    # cumsum along axis 0 (trajectories)
    f_Cross_cumsum = np.cumsum((f_Cross_all), axis=0)
    traj_counts = np.arange(1, N_traj + 1).reshape(-1, 1)
    f_Cross_running_mean = f_Cross_cumsum / traj_counts
    
    # Square of the mean
    f_Cross_sq_mean = np.abs(f_Cross_running_mean)
    
    # Select 5 equally sparse times
    # We ignore t=0 usually as it might be singular or trivial, but let's take linspace
    # times is length 1000. 
    # indices: 0, 249, 499, 749, 999 roughly
    selected_indices = np.linspace(N_times/4, N_times - 1, 4, dtype=int)
    
    # Use a colormap for the lines
    colors = plt.cm.plasma(np.linspace(0, 0.8, 5))
    
    for idx, t_idx in enumerate(selected_indices):
        t_val = times[t_idx]
        ax.plot(traj_counts, f_Cross_sq_mean[:, t_idx], 
                color=colors[idx], lw=1.5, label=f't={t_val:.1f}')
    
    ax.set_xlabel(r'$N_{{\rm trajs}}$', fontsize=13)
    ax.set_ylabel(r'$\langle f_{Q}^{X} \rangle^2$', fontsize=13)
    ax.text(0.05, 0.1, r'(c)', transform=ax.transAxes, fontsize=13) # Text inside
    # Legend above:
    ax.legend(fontsize=11, loc='lower center', frameon=True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # (d) Sample trajectories
    ax = axes[1, 1]
    for n in range(min(10, N_traj)):
        ax.plot(times, stoch_length_f_Q[n]**2 / stoch_action_f_Q[n], alpha=0.4, lw=0.8)
    ax.plot(times, times, 'k-', lw=1, label=r'$t$')
    ax.plot(times, Length_F_Q**2 / Action_F_Q, '--', color=c4, lw=1, label=r'$\mathcal{L}^2 / \mathcal{J}$')
    ax.set_xlabel(r'$t$', fontsize=13)
    ax.set_ylabel(r'$\ell^2 / j$', fontsize=13)
    ax.text(0.05, 0.8, r'(d)', transform=ax.transAxes, fontsize=13) # Text inside
    # Legend above:
    ax.legend(fontsize=11, loc='lower right', frameon=True)
    ax.set_xlim(-0.001, tau)
    ax.set_yscale('log')
    
    plt.savefig('info_geom_driven_tls_modified.png', dpi=200, 
                bbox_inches='tight', facecolor='white')
    plt.savefig('info_geom_driven_tls_modified.pdf', bbox_inches='tight')

    print("Modified figure saved.")

if __name__ == "__main__":
    main()