import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import matplotlib.ticker as mticker
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
    """
    Symmetric Logarithmic Derivative: dρ/dt = (Lρ + ρL)/2
    
    In eigenbasis: L_xy = 2⟨x|dρ/dt|y⟩/(p_x + p_y)0
    """
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
    """F_Q = Tr(ρL²)"""
    return np.real(np.trace(rho @ L @ L))


def ensemble_qfi_decomposition(rho: np.ndarray, drho: np.ndarray) -> Tuple[float, float]:
    """
    F_Q^IC = Σ_x (dp_x/dt)² / p_x
    F_Q^C  = 2Σ_{x≠y} σ_xy |⟨y|∂_t x⟩|²  where σ_xy = (p_x-p_y)²/(p_x+p_y)
    """
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 1e-14)
    
    # dp/dt in eigenbasis
    dp = np.array([np.real(np.vdot(eigvecs[:, i], drho @ eigvecs[:, i])) for i in range(2)])
    
    # Incoherent
    F_IC = sum(dp[i]**2 / eigvals[i] for i in range(2) if eigvals[i] > 1e-13)
    
    # Coherent (from off-diagonal terms)
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
    """
    CQFI via Eq. 11: f_Q = Tr(Π L² ρ) / Tr(Π ρ)
    where Π = |ψ⟩⟨ψ|
    """
    psi = psi / np.linalg.norm(psi)
    p = np.real(np.vdot(psi, rho @ psi))
    if p < 1e-14:
        return 0.0
    
    L2_rho = L @ L @ rho
    return np.real(np.vdot(psi, L2_rho @ psi)) / p

def alt_cqfi_direct(psi: np.ndarray, rho: np.ndarray, L: np.ndarray) -> float:
    """
    CQFI via Eq. 11: f_Q = Tr(Π L²)
    where Π = |ψ⟩⟨ψ|
    """
    psi = psi / np.linalg.norm(psi)
    p = np.real(np.vdot(psi, rho @ psi))
    if p < 1e-14:
        return 0.0
    
    L2 = L @ L
    return np.real(np.vdot(psi, L2 @ psi))


def cqfi_decomposed(psi: np.ndarray, dpsi: np.ndarray, 
                    rho: np.ndarray, drho: np.ndarray) -> Tuple[float, float, float]:
    """
    CQFI decomposition via SLD operator partition L = D + O.
    
    f_Q = <ψ| L² |ψ> = <ψ| (D+O)² |ψ> 
        = <ψ| D² |ψ> + <ψ| O² |ψ> + <ψ| {D,O} |ψ>
        
    Where:
      - D (Diagonal): Incoherent contribution (eigenvalue changes)
      - O (Off-diagonal): Coherent contribution (basis rotation)
      - {D,O} (Cross): Interference term (vanishes in ensemble average)
    
    Returns:
        f_inc, f_coh, f_cross
    """
    psi = psi / np.linalg.norm(psi)
    
    # 1. Diagonalize density matrix
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 1e-14)
    
    # 2. Compute SLD matrix elements in the eigenbasis
    # L_mn = 2 <m|drho|n> / (p_m + p_n)
    
    # Transform drho to eigenbasis
    drho_eig = eigvecs.conj().T @ drho @ eigvecs
    
    dim = len(eigvals)
    L_eig = np.zeros((dim, dim), dtype=complex)
    
    for i in range(dim):
        for j in range(dim):
            denom = eigvals[i] + eigvals[j]
            if denom > 1e-13:
                L_eig[i, j] = 2.0 * drho_eig[i, j] / denom
    
    # 3. Partition L into Diagonal (D) and Off-Diagonal (O) parts
    D_eig = np.diag(np.diag(L_eig))
    O_eig = L_eig - D_eig
    
    # 4. Express state |ψ> in eigenbasis: c = U†|ψ>
    c = eigvecs.conj().T @ psi
    
    # 5. Compute contributions: <c| Op |c>
    
    # Incoherent: <ψ| D² |ψ> = <c| D_eig @ D_eig |c>
    # D is diagonal and real, so D^2 is just element-wise square
    # Using vectors for clarity: v_D = D_eig @ c
    v_D = D_eig @ c
    f_inc = np.real(np.vdot(v_D, v_D))
    
    # Coherent: <ψ| O² |ψ> = <c| O_eig @ O_eig |c>
    # Since O is Hermitian, <c|O^2|c> = |O|c>|^2
    v_O = O_eig @ c
    f_coh = np.real(np.vdot(v_O, v_O))
    
    # Interference: <ψ| {D,O} |ψ> = <ψ|DO|ψ> + <ψ|OD|ψ>
    # = <ψ|DO|ψ> + <ψ|DO|ψ>* = 2 Re <ψ|DO|ψ>
    # = 2 Re <c| D_eig @ O_eig |c> = 2 Re <D† c | O c>
    # Since D is Hermitian: 2 Re <v_D | v_O>
    f_cross = 2.0 * np.real(np.vdot(v_D, v_O))
    
    # Note: These are "local" Fisher information values <ψ|L²|ψ>.
    # If standard CQFI is normalized by p_psi = <ψ|ρ|ψ>, these values are implicitly 
    # weighted by the probability of the trajectory occurring naturally, 
    # but the operator expectation <L²> is usually the quantity of interest.
    # The previous code calculated f_Q directly as <ψ|L²|ψ> / p_psi? 
    # Wait, the previous `cqfi_direct` calculated Tr(Pi L^2 rho) / Tr(Pi rho).
    # Since we are in a quantum trajectory, rho_trajectory = |ψ><ψ|.
    # The "state" for the SLD calculation is the global rho.
    # The expectation value we want is <ψ| L_rho^2 |ψ>.
    # This corresponds exactly to the decomposition above.
    
    return f_inc, f_coh, f_cross

def numerical_derivative(arr: List[np.ndarray], t: np.ndarray, i: int) -> np.ndarray:
    """Central difference."""
    if i == 0:
        return (arr[1] - arr[0]) / (t[1] - t[0])
    elif i == len(t) - 1:
        return (arr[-1] - arr[-2]) / (t[-1] - t[-2])
    return (arr[i+1] - arr[i-1]) / (t[i+1] - t[i-1])


def main():
    plt.rcParams.update({
        "font.family": "serif",  # Usar fonte com serifa
        "font.serif": ["Times New Roman", "DejaVu Serif"], # Preferir Times
        "font.size": 13,         # Tamanho da fonte base
        "axes.labelsize": 13,    # Tamanho da fonte dos eixos
        "axes.titlesize": 13,    # Tamanho da fonte dos títulos
        "legend.fontsize": 13,    # Tamanho da fonte da legenda
        "xtick.labelsize": 13,    # Tamanho da fonte dos ticks x
        "ytick.labelsize": 13,    # Tamanho da fonte dos ticks y
        "mathtext.fontset": "stix", # Usar fontes STIX para LaTeX
        "mathtext.rm": "serif",
        "xtick.direction": "in", # Ticks para dentro
        "ytick.direction": "in",
        "xtick.top": True,       # Habilitar ticks em cima
        "ytick.right": True,     # Habilitar ticks à direita
        "axes.linewidth": 1.0,   # Largura da linha dos eixos
        "lines.linewidth": 1.5,  # Largura da linha do plot
        "legend.frameon": True, # Legenda sem borda
        "legend.framealpha":0.4,
        "figure.dpi": 300,       # DPI para salvar
        "savefig.bbox": "tight"  # Salvar com bounding box justo
    })
    print("=" * 70)
    print("CONDITIONAL QUANTUM FISHER INFORMATION")
    print("Driven-Dissipative Two-Level System")
    print("=" * 70)
    
    # System
    system = DrivenThermalQubit(epsilon=0.15, Gamma0=0.4, omega=1.0, T=0.8)
    
    print(f"\nParameters:")
    print(f"  Driving: ε = {system.epsilon}")
    print(f"  Decay: Γ₀ = {system.Gamma0}")
    print(f"  Temperature: T = {system.T} → n̄ = {system.nbar:.3f}")
    print(f"  Emission rate: Γ₋ = {system.Gamma_m:.4f}")
    print(f"  Absorption rate: Γ₊ = {system.Gamma_p:.4f}")
    
    # Time evolution
    N_times = 1000
    tau = 10.0
    times = np.linspace(0, tau, N_times)
    dt = times[1] - times[0]

    # Initial state: excited |e⟩
    psi0 = np.array([1, 0], dtype=complex)
    rho0 = np.outer(psi0, psi0.conj())
    
    # Ensemble evolution
    print("\n[1/3] Computing ensemble dynamics...")
    rho_t = system.evolve_rho(rho0, times)
    
    # Ensemble QFI
    print("[2/3] Computing ensemble QFI...")
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
    N_traj = 5000
    print(f"[3/3] Simulating {N_traj} quantum trajectories...")
    
    f_Q_all = np.zeros((N_traj, N_times))
    alt_f_Q_all = np.zeros((N_traj, N_times))
    f_IC_all = np.zeros((N_traj, N_times))
    f_C_all = np.zeros((N_traj, N_times))
    f_Cross_all = np.zeros((N_traj, N_times)) # Added for cross term
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
    
    # ========== PLOTTING ==========
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(6, 5), constrained_layout=True)
    fig.patch.set_facecolor('white')
    
    # Style
    c1, c2, c3, c4 = '#E63946', '#1D3557', '#457B9D', '#F4A261'
    c_cross = '#2A9D8F' # Color for cross term
    
    # (a) Total CQFI vs Ensemble QFI
    ax = axes[0, 0]
    ax.plot(times, f_Q_mean, color=c2, lw=2.5, label=r'$\langle f_Q \rangle$')
    ax.plot(times, alt_f_Q_mean, color=c3, lw=2.5, label=r'$\langle f_Q \rangle_{s.\rho}$')    
    ax.plot(times, F_Q, '--', color=c4, lw=2.5, label=r'$\mathcal{F}_Q$')
    ax.set_xlabel(r'$t$', fontsize=13)
    ax.set_ylabel(r'$\mathcal{F}_Q$', fontsize=13)
    ax.text(0.05, 0.9, r'(a)', transform=ax.transAxes, fontsize=13) # Text inside
    ax.legend(fontsize=11, loc='upper right', frameon=True)
    ax.set_xlim(0, tau)
    ax.set_yscale('log')
    
    # (b) CQFI decomposition
    ax = axes[0, 1]
    ax.plot(times, f_IC_mean, color=c1, lw=2.5, label=r'$\langle f_Q^{\mathrm{IC}} \rangle$')
    ax.plot(times, f_C_mean, color=c3, lw=2.5, label=r'$\langle f_Q^{\mathrm{C}} \rangle$')
    ax.plot(times, f_Cross_mean, color=c_cross, lw=2.5, label=r'$\langle f_Q^{\mathrm{X}} \rangle$')
    ax.plot(times, f_Q_mean, '--', color=c2, lw=2, label=r'$\langle f_Q \rangle$ ')
    ax.set_xlabel(r'$t$', fontsize=13)
    ax.set_ylabel(r'$\langle f_Q\rangle$', fontsize=13)
    ax.text(0.05, 0.9, r'(b)', transform=ax.transAxes, fontsize=13) # Text inside
    ax.legend(fontsize=11, loc='lower right', frameon=True)
    ax.set_xlim(0, tau)
    # --- Ticks at every 10^2 ---
    ax.set_yscale('symlog', linthresh=1e-4)
    ax.yaxis.set_major_locator(mticker.SymmetricalLogLocator(linthresh=1e-3, base=1000))
    
    # (c) Ensemble QFI decomposition
    ax = axes[1, 0]
    ax.plot(times, F_IC_ens, color=c1, lw=2.5, label=r'$\mathcal{F}_Q^{\mathrm{IC}}$')
    ax.plot(times, F_C_ens, color=c3, lw=2.5, label=r'$\mathcal{F}_Q^{\mathrm{C}}$')
    ax.plot(times, F_Q, '--', color=c2, lw=2, label=r'$\mathcal{F}_Q$')
    ax.set_xlabel(r'$t$', fontsize=13)
    ax.set_ylabel(r'$\mathcal{F}_Q$', fontsize=13)
    ax.text(0.05, 0.9, r'(c)', transform=ax.transAxes, fontsize=13) # Text inside
    ax.legend(fontsize=11, loc='lower left', frameon=True)
    ax.set_xlim(0, tau)
    ax.set_yscale('log')
    
    # (d) Sample trajectories
    ax = axes[1, 1]
    for n in range(min(10, N_traj)):
        ax.plot(times, alt_f_Q_all[n], alpha=0.4, lw=0.8)
    ax.plot(times, f_Q_mean, 'k-', lw=2.5, label=r'$\langle f_Q \rangle$')
    ax.plot(times, F_Q, '--', color=c4, lw=2.5, label=r'$\mathcal{F}_Q$')
    ax.set_xlabel(r'$t$', fontsize=13)
    ax.set_ylabel(r'$\langle f_Q\rangle$', fontsize=13)
    ax.text(0.05, 0.9, r'(d)', transform=ax.transAxes, fontsize=13) # Text inside
    ax.legend(fontsize=11, loc='upper right', frameon=True)
    ax.set_xlim(0, tau)
    ax.set_yscale('log')
    
    plt.savefig('cqfi_driven_tls.png', dpi=200, 
                bbox_inches='tight', facecolor='white')
    plt.savefig('cqfi_driven_tls.pdf', bbox_inches='tight')
    

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    idx_mid = N_times // 2
    print(f"\nAt t = {times[idx_mid]:.2f}:")
    print(f"  Ensemble F_Q        = {F_Q[idx_mid]:.6f}")
    print(f"    F_Q^IC            = {F_IC_ens[idx_mid]:.6f}")
    print(f"    F_Q^C             = {F_C_ens[idx_mid]:.6f}")
    print(f"  Mean CQFI <f_Q>     = {f_Q_mean[idx_mid]:.6f}")
    
    print(f"\nAt t = {tau}:")
    print(f"  Ensemble F_Q        = {F_Q[-1]:.6f}")
    print(f"  Mean CQFI <f_Q>     = {f_Q_mean[-1]:.6f}")
    
    # Verify averaging property
    rel_err = np.mean(np.abs(f_Q_mean[10:] - F_Q[10:]) / (F_Q[10:] + 1e-10)) * 100
    print(f"\nAveraging relation ⟨f_Q⟩ ≈ F_Q:")
    print(f"  Mean relative error = {rel_err:.2f}%")
    
    return times, F_Q, f_Q_mean


if __name__ == "__main__":
    times, F_Q, f_Q_mean = main()