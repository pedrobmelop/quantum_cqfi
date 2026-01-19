import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

# ==========================================
# 1. GENERATE "CARTOON" DATA
# ==========================================
def get_cartoon_data():
    t = np.linspace(0, 10, 200)
    
    # Incoherent (Population) - Decays
    ic_term = 4.0 * np.exp(-0.15 * t) + 0.5
    
    # Coherent (Rotation) - Rises then settles
    c_term = 3.0 * (1 - np.exp(-0.5 * t)) + 0.2
    
    # Cross Term (Ensemble) - Effectively zero noise
    np.random.seed(42)
    x_ens = np.random.normal(0, 0.05, len(t))
    
    # Cross Term (Single Trajectory) - Big negative dip!
    x_traj = x_ens + 1.5 * np.sin(t) * np.cos(3*t)
    x_traj[40:80] -= 3.5 # The "Whoops" moment
    
    return t, ic_term, c_term, x_ens, x_traj

# ==========================================
# 2. DRAWING HELPERS (Horizontal Cartoon)
# ==========================================
def draw_schematic_elements(ax):
    """Draws the physical system cartoon in a wide format."""
    ax.set_xlim(0, 14)
    ax.set_ylim(-2, 6)
    ax.axis('off')
    ax.set_title("(a) Setup", fontsize=16, y=1.05)

    # --- 1. The Qubit System (Left) ---
    # Energy Levels
    ax.plot([1, 3.5], [4.5, 4.5], 'k-', lw=3) # Excited
    ax.text(0.8, 4.5, '|e>', fontsize=14, ha='right', va='center')
    
    ax.plot([1, 3.5], [2.0, 2.0], 'k-', lw=3) # Ground
    ax.text(0.8, 2.0, '|g>', fontsize=14, ha='right', va='center')
    
    # Driving Field (Double Arrow)
    ax.annotate('', xy=(2.25, 4.4), xytext=(2.25, 2.1),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax.text(2.4, 3.2, 'Drive', color='purple', fontsize=12)

    # --- 2. The Thermal Bath (Blob) ---
    # Random blob shape
    theta = np.linspace(0, 2*np.pi, 50)
    r = 1.0 + 0.1 * np.random.rand(50)
    bx = 1.5 + r * np.cos(theta)
    by = 3.5 + r * np.sin(theta)
    
    # Draw bath (behind lines ideally, but order is tricky in simple plot)
    ax.plot(bx, by, 'r-', lw=2, alpha=0.5)
    ax.text(0.5, 5.0, 'Hot Bath', color='red', fontsize=10)
    
    # Interaction Wiggles
    x_wig = np.linspace(1.2, 1.8, 15)
    y_wig = 3.5 + 0.2 * np.sin(30 * x_wig)
    ax.plot(x_wig, y_wig, 'r-', lw=1.5)

    # --- 3. The Observation (Center) ---
    eye_x, eye_y = 6.0, 3.0
    
    # Dashed line from system to eye
    ax.annotate('', xy=(eye_x - 0.5, eye_y), xytext=(3.6, 3.2),
                arrowprops=dict(arrowstyle='->', linestyle='dashed', color='black', lw=2))
    
    # Eyeball
    ax.plot(eye_x, eye_y, 'ko', markersize=8) # Pupil
    t = np.linspace(0, np.pi, 20)
    ax.plot(eye_x + 0.7*np.cos(t), eye_y + 0.4*np.sin(t), 'k-', lw=2.5) # Top lid
    ax.plot(eye_x + 0.7*np.cos(t), eye_y - 0.4*np.sin(t), 'k-', lw=2.5) # Bottom lid
    ax.text(eye_x, eye_y - 1.2, 'Monitoring', ha='center', fontsize=12)

    # --- 4. THE TRAJECTORIES (Right) ---
    start_x = eye_x + 1.0
    start_y = eye_y
    
    # Ghost paths (Ensemble)
    np.random.seed(99)
    for i in range(7):
        tr_x = np.linspace(start_x, 13.5, 40)
        noise = np.cumsum(np.random.normal(0, 0.15, 40))
        # Spread them out
        spread = (i - 3) * 0.4
        tr_y = start_y + noise + spread
        ax.plot(tr_x, tr_y, color='gray', alpha=0.3, lw=1.5)

    # Highlight Path (The "Green" one)
    tr_x_main = np.linspace(start_x, 13.5, 50)
    noise_main = np.cumsum(np.random.normal(0, 0.2, 50))
    noise_main[25:] -= 1.5 # The Dip
    tr_y_main = start_y + noise_main
    
    ax.plot(tr_x_main, tr_y_main, 'xkcd:green', lw=3)
    
    ax.text(11.5, start_y + 2.5, 'Stochastic\nOutcomes', ha='center', fontsize=12, color='gray')
    ax.annotate('This run!', xy=(13, tr_y_main[-1]), xytext=(11, 1.0),
                arrowprops=dict(arrowstyle='->', color='green'), color='green', fontsize=12)


# ==========================================
# 3. MAIN PLOT FUNCTION
# ==========================================
def draw_full_cartoon_figure():
    t, ic_term, c_term, x_ens, x_traj = get_cartoon_data()
    
    # Activate XKCD style!
    with plt.xkcd():
        # Taller figure to accommodate top row
        fig = plt.figure(figsize=(8, 6))
        
        # GridSpec: Top row (schematic) gets less height ratio than plots
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.2)
        
        # --- TOP ROW: SCHEMATIC ---
        ax_schematic = fig.add_subplot(gs[0, :]) # Spans all columns
        draw_schematic_elements(ax_schematic)
        
        # --- BOTTOM LEFT: SINGLE TRAJECTORY ---
        ax_single = fig.add_subplot(gs[1, 0])
        ax_single.set_title("(b)", fontsize=14)
        ax_single.plot(t, ic_term, 'xkcd:sky blue', lw=3, label='Populations')
        ax_single.plot(t, c_term, 'xkcd:tomato', lw=3, label='Coherences')
        ax_single.plot(t, x_traj, 'xkcd:green', lw=4, label='Cross-Term')
        
        # Zero line
        ax_single.axhline(0, color='gray', linestyle='--', lw=2)
        
        # Annotation for the dip
        ax_single.annotate('DESTRUCTIVE INTERFERENCE!', xy=(3, -2.5), xytext=(0.5, -4),
                           arrowprops=dict(arrowstyle='->', color='red', lw=2),
                           color='red', fontsize=12, fontweight='bold')
        
        ax_single.set_ylabel("Trajectory QFI")
        ax_single.set_xlabel("time")
        # Legend inside
        ax_single.legend(loc='upper right', frameon=True, framealpha=0.4, fontsize=10)
        
        # --- BOTTOM RIGHT: ENSEMBLE ---
        ax_ens = fig.add_subplot(gs[1, 1])
        ax_ens.set_title("(c) ", fontsize=14)
        ax_ens.plot(t, ic_term, 'xkcd:sky blue', lw=3, alpha=0.5)
        ax_ens.plot(t, c_term, 'xkcd:tomato', lw=3, alpha=0.5)
        ax_ens.plot(t, x_ens, 'xkcd:green', lw=4, label='Avg Cross-Term ~ 0')
        
        ax_ens.axhline(0, color='gray', linestyle='--', lw=2)
        
        ax_ens.annotate('Basically Zero', xy=(5, 0), xytext=(3, 1),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        ax_ens.set_xlabel("time")
        ax_ens.set_ylabel("Average QFI")
        ax_ens.legend(loc='upper right', frameon=True, framealpha=0.5, fontsize=10)

        plt.tight_layout()
        plt.savefig('cartoon_top_layout.pdf', dpi=200)
        plt.savefig('cartoon_top_layout.png', dpi=150)
        plt.show()

if __name__ == "__main__":
    draw_full_cartoon_figure()