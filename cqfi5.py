import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings
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
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "legend.fontsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
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
    
    # RHS is Length_F_Q (which is \int sqrt(F_Q) dt)
    Length_F_Q = np.cumsum(np.sqrt(F_Q)) * dt
    qsl_rhs_ens = Length_F_Q 

    # ==========================================
    # IMPLEMENTATION: Ensemble Speed Limit (Eq. 37)
    # Observable O = H
    # d<H>/dt = Tr(H * (-i[H, rho] + D[rho]))
    # Since [H, H] = 0, d<H>/dt = Tr(H * D[rho])
    # ==========================================
    O_obs = system.H
    O_sq = O_obs @ O_obs
    
    qsl_integrand_ens = np.zeros(N_times)
    
    for i, rho in enumerate(rho_t):
        # 1. Variance Delta_rho H
        tr_O2 = np.real(np.trace(rho @ O_sq))
        tr_O = np.real(np.trace(rho @ O_obs))
        var = tr_O2 - tr_O**2
        obs_std = np.sqrt(max(var, 1e-16))
        
        # 2. Rate of change (Unitary + Dissipative)
        # Unitary part -i[H,H] = 0
        # Dissipative part Tr(H D[rho])
        dissipator_rho = system.dissipator(rho)
        rate_val = np.abs(np.real(np.trace(O_obs @ dissipator_rho)))
        
        qsl_integrand_ens[i] = rate_val / (obs_std + 1e-16)

    qsl_lhs_ens = np.cumsum(qsl_integrand_ens) * dt

    # ==========================================
    
    # Trajectories
    N_traj = 50000  
    print(f"Simulating {N_traj} quantum trajectories...")

    f_Q_all = np.zeros((N_traj, N_times))
    stoch_length_f_Q = np.zeros((N_traj, N_times))
    qsl_lhs_traj_all = np.zeros((N_traj, N_times))

    for n in range(N_traj):
        if (n + 1) % 50 == 0:
            print(f"      {n+1}/{N_traj}")
        psi_t = system.quantum_trajectory(psi0, times, seed=n)
        dpsi_t = [numerical_derivative(psi_t, times, i) for i in range(N_times)]
        
        integrand_traj = np.zeros(N_times)
        
        for i in range(N_times):
            # QFI Calculation
            f_Q_all[n, i] = cqfi_direct(psi_t[i], rho_t[i], L_t[i])
            
            # ==========================================
            # IMPLEMENTATION: Trajectory Speed Limit (Eq. 38/39)
            # Observable O = H, State = |psi><psi|
            # Using Unitary+Dissipative contributions from Master Eq
            # ==========================================
            psi = psi_t[i]
            
            # Variance Delta_psi H
            val_O = np.real(np.vdot(psi, O_obs @ psi))
            val_O2 = np.real(np.vdot(psi, O_sq @ psi))
            var_traj = val_O2 - val_O**2
            obs_std_traj = np.sqrt(max(var_traj, 1e-16))
            
            # Rate of change (Unitary + Dissipative applied to pure state)
            # rho_pure = |psi><psi|
            rho_pure = np.outer(psi, psi.conj())
            
            # Since [H, H] = 0, unitary part is 0.
            # Dissipative part: Tr(H D[rho_pure])
            dissipator_psi = system.dissipator(rho_pure)
            rate_traj = np.abs(np.real(np.trace(O_obs @ dissipator_psi)))
            
            integrand_traj[i] = rate_traj / (obs_std_traj + 1e-16)

        stoch_length_f_Q[n,:] = np.cumsum(np.sqrt(f_Q_all[n,:])) * dt
        qsl_lhs_traj_all[n, :] = np.cumsum(integrand_traj) * dt

    # ==========================================
    # PLOTTING FIGURE 3: SPEED LIMITS
    # ==========================================
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    fig.patch.set_facecolor('white')
    
    # Colors
    c_lhs = '#E63946' # Reddish
    c_rhs = '#1D3557' # Dark Blue
    
    # (a) Ensemble Speed Limit
    ax = axes[0]
    ax.plot(times, qsl_rhs_ens, color=c_rhs, lw=2.5, label=r'$\int \sqrt{\mathcal{F}_Q} dt$')
    ax.plot(times, qsl_lhs_ens, '--', color=c_lhs, lw=2.5, label=r'$\int{\rm d}t~\frac{|\langle\dot{H}(t)\rangle|}{\Delta H}$')
    
    ax.set_xlabel(r'$t$', fontsize=15)
    ax.set_title(r'(a)', fontsize=15)
    ax.legend(fontsize=15, loc='upper left')
    ax.set_xlim(0, tau)
    
    # (b) Trajectory Speed Limits
    ax = axes[1]
    
    for n in range(min(20, N_traj)):
        # Plot RHS (Solid thin)
        ax.plot(times, stoch_length_f_Q[n], color=c_rhs, alpha=0.3, lw=1.0)
        # Plot LHS (Dashed thin)
        ax.plot(times, qsl_lhs_traj_all[n], '--', color=c_lhs, alpha=0.3, lw=1.0)
        
    ax.plot([], [], color=c_rhs, lw=2, label=r'$\int \sqrt{f_Q} dt$')
    ax.plot([], [], '--', color=c_lhs, lw=2, label=r'$\int{\rm d}t~\frac{|\langle\dot{H}(t)\rangle|}{\Delta_\gamma H}$')
    
    ax.set_xlabel(r'$t$', fontsize=15)
    ax.set_title(r'(b)', fontsize=15)
    ax.legend(fontsize=15, loc='upper left')
    ax.set_xlim(0, tau)
    
    plt.tight_layout()
    plt.savefig('speed_limits_hamiltonian_contributions.png', dpi=200, 
                bbox_inches='tight', facecolor='white')
    plt.savefig('speed_limits_hamiltonian_contributions.pdf', bbox_inches='tight')

    print("Speed limits figure saved.")

if __name__ == "__main__":
    main()