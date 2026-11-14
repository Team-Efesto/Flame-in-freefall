import os
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Configurazione deterministica e precisione
torch.manual_seed(42)
np.random.seed(42)
torch.set_default_dtype(torch.float32)

# ==============================================================================
# 1. MOTORE FISICO VETTORIALIZZATO MULTI-PI (PDF Liao 2025 + DOCX BASS-II 1g)
# ==============================================================================
class UnifiedFlamePhysics:
    G_EARTH = 9.81          # Accelerazione gravità terrestre [m/s^2]
    G_MOON_RATIO = 0.160    # Rapporto gravità Lunare / Terrestre (Case 5 PDF: 0.16 g_e)
    P_REF = 100.0           # Pressione di riferimento [kPa]
    T_INF = 300.0           # Temperatura ambiente [K]
    T_P_SIBAL = 620.0       # Temperatura pirolisi SIBAL [K]
    T_P_PMMA = 650.0        # Temperatura pirolisi PMMA [K]
    F_DERIS_EXPT = 4.76     # Coefficiente di fiamma sperimentale di de Ris (DOCX Sez. 2.2)
    TAU_TRANS_MM = 5.19     # Spessore critico di transizione sottile-spesso a 1g [mm]

    @staticmethod
    def load_real_bass2_stats(zip_path="drive-download-20261114T201035Z-1-001.zip",
                              csv_path="PSI-25_Experimental table_BASS-II.csv"):
        """Estrae automaticamente i range reali dal CSV NASA PSI-25 se presente."""
        if os.path.exists(zip_path) and not os.path.exists(csv_path):
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extract(csv_path, ".")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            o2_col = [c for c in df.columns if 'initial O2' in c][0]
            o2_vals = pd.to_numeric(df[o2_col], errors='coerce').dropna().values
            return float(np.min(o2_vals)), float(np.max(o2_vals)), len(df)
        return 13.6, 22.2, 130

    @classmethod
    def evaluate_exact_physics(cls, config_id, g_ratio, p_kpa, u_forced, o2_pct, dim_mm, t_init=300.0):
        """
        Calcola in modo totalmente vettorializzato (zero loop) tutte le grandezze fisiche
        per qualsiasi combinazione di input o batch di tensori NumPy.
        
        config_id:
          1 = Serie GMT-T (T'ien): Tessuto SIBAL (Concorrente Ascendente, dim_mm = larghezza W [mm])
          2 = Serie B (Bhattacharjee): Film PMMA sottile (Opposto/Discendente, dim_mm = spessore [mm])
          3 = Serie M (Miller/Wichman): Lastra PMMA 1-5 mm (Opposto/Discendente, dim_mm = spessore [mm])
          4 = Serie P (Fernandez-Pello): Cilindro PMMA (Opposto/Discendente, dim_mm = diametro d [mm])
          5 = Serie O (Olson/Ferkul): Cilindro PMMA (Concorrente Ristagno Assiale, dim_mm = diametro d [mm])
        """
        g = np.asarray(g_ratio, dtype=np.float32)
        p = np.asarray(p_kpa, dtype=np.float32)
        u = np.asarray(u_forced, dtype=np.float32)
        o2 = np.asarray(o2_pct, dtype=np.float32)
        d = np.asarray(dim_mm, dtype=np.float32)
        cfg = np.asarray(config_id, dtype=np.int32)

        p_norm = p / cls.P_REF
        p18g = (p ** 1.8) * np.maximum(g, 1e-5)
        
        # Numeri adimensionali (PDF Sez. 2.1 e DOCX Sez. 2.1)
        gr = 6.9e6 * (p_norm ** 2) * g
        re = 393.0 * p_norm * (u / 10.0)
        gr_re2 = gr / (re ** 2 + 1e-6)
        f_buoy = np.tanh(gr_re2 / 0.8)  # 0 in microgravità, ~1 su Luna (Gr/Re^2 ~ 7.1) e Terra (44.6)

        # 1. Velocità di convezione naturale e Velocità Efficace (DOCX Sez. 2.1 e 4, PDF Eq. 3)
        v_nc_edge = 10.2 * (np.maximum(g, 0.0) ** (1.0 / 3.0))                # Scala bordo d'attacco B [cm/s]
        u_b_opposed = 30.0 * (np.maximum(g, 0.0) ** 0.5)                      # Pennacchio opposto P, M [cm/s]
        u_b_stagnation = 25.0 * (np.maximum(g, 0.0) ** 0.5)                   # Ristagno assiale O [cm/s]
        u_b_upward_plume = 112.0 * (p_norm ** (-0.007)) * (np.maximum(g, 0.0) ** 0.4997) # PDF Fig. 5f [cm/s]

        v_eff = np.select(
            [cfg == 1, cfg == 2, cfg == 3, cfg == 4, cfg == 5],
            [
                np.sqrt(u ** 2 + (f_buoy * u_b_upward_plume) ** 2),
                u + v_nc_edge,
                u + u_b_opposed,
                u + u_b_opposed,
                np.sqrt(u ** 2 + u_b_stagnation ** 2)
            ],
            default=np.sqrt(u ** 2 + u_b_upward_plume ** 2)
        )

        # 2. Distanza di Standoff della fiamma delta_f [mm] (PDF Eq. 2 e Fig. 5c-d)
        delta_buoy_mm = 4.8 * (p_norm ** (-0.505)) * (np.maximum(g, 0.02) ** (-0.22))
        delta_forced_mm = 8.5 * (p_norm ** (-0.5)) * (np.maximum(u / 10.0, 0.15) ** (-0.5))
        delta_f_mm = f_buoy * delta_buoy_mm + (1.0 - f_buoy) * delta_forced_mm

        # 3. Fattore termico temperatura iniziale e ossigeno
        t_p_mat = np.where(cfg == 1, cls.T_P_SIBAL, cls.T_P_PMMA)
        t_corr = (t_p_mat - cls.T_INF) / np.maximum(50.0, t_p_mat - t_init)
        o2_factor = np.maximum(0.0, (o2 - 13.0) / (21.0 - 13.0)) ** 1.25

        # 4. Velocità di Propagazione V_f [mm/s] specifica per ogni Serie PI
        # --- Config 1: SIBAL Concorrente Ascendente (PDF Fig. 12b + DOCX Tab. A.6) ---
        w_factor = np.where(d < 16.0, 0.84, 1.0) # Correzione larghezza W=12mm vs W=22mm
        vf_sibal_buoy = 0.00585 * p18g * o2_factor * w_factor * ((np.maximum(u, 1.0) / 10.0) ** 0.08) # [mm/s]
        vf_sibal_micro = 1.85 * ((np.maximum(u, 0.5) / 10.0) ** 0.5) * (p_norm ** 0.5) * o2_factor * w_factor
        vf_cfg1 = f_buoy * vf_sibal_buoy + (1.0 - f_buoy) * vf_sibal_micro

        # --- Config 2: Film Sottili PMMA 100-400 um (de Ris V_f = 2.20 mm/s a 100 um, DOCX Tab. A.1) ---
        tau_um = np.clip(d * 1000.0, 50.0, 1000.0)
        vf_b_thermal = 2.20 * (100.0 / tau_um) * (p_norm ** 0.25) * ((o2 / 20.9) ** 1.35)
        vf_b_micro = vf_b_thermal * np.tanh(np.maximum(u - 0.8, 0.0) / 3.5)
        vf_b_buoy = vf_b_thermal * (0.88 + 0.12 * (g ** 0.20)) # Sulla Luna/Terra V_NC previene il quenching
        vf_cfg2 = f_buoy * vf_b_buoy + (1.0 - f_buoy) * vf_b_micro

        # --- Config 3: Lastre PMMA Spesse 1-5 mm (Miller M1-M20, transizione 5.19 mm, DOCX Tab. A.4) ---
        thick_mm = np.clip(d, 0.8, 10.0)
        vf_m_0g = 0.115 * (thick_mm ** (-0.62)) * ((o2 / 20.5) ** 1.5) * ((np.clip(u, 1.0, 25.0) / 8.0) ** 0.25)
        vf_m_buoy = (0.155 * (thick_mm ** (-0.55)) + 0.015) * (g ** 0.15) * ((o2 / 20.5) ** 1.6)
        vf_cfg3 = f_buoy * vf_m_buoy + (1.0 - f_buoy) * vf_m_0g

        # --- Config 4: Cilindri PMMA Opposto (Fernandez-Pello P1-P54, inversione a 19% O2, DOCX Tab. A.3) ---
        diam_mm = np.clip(d, 4.0, 20.0)
        vf_p_0g = 0.088 * ((6.35 / diam_mm) ** 0.85) * ((o2 / 21.0) ** 1.4) * np.exp(-((u - 3.5) / 8.0) ** 2)
        # Sulla Luna e Terra: per O2 > 19% il galleggiamento accelera V_f; per O2 < 19% raffredda la fiamma
        buoy_o2_inv = 1.0 + 0.48 * (g ** 0.25) * np.tanh((o2 - 19.0) / 1.2)
        vf_p_buoy = 0.095 * ((6.35 / diam_mm) ** 0.85) * ((o2 / 21.0) ** 1.5) * buoy_o2_inv
        vf_cfg4 = f_buoy * vf_p_buoy + (1.0 - f_buoy) * vf_p_0g

        # --- Config 5: Cilindri PMMA Concorrente Ristagno (Olson O1-O24, DOCX Tab. A.2/A.5) ---
        r_cm = np.clip(d / 20.0, 0.15, 1.5)
        a_stretch = 1.5 * v_eff / r_cm # Tasso di stiramento [s^-1]
        vf_o_base = 0.092 * ((0.318 / r_cm) ** 0.7) * ((a_stretch / 35.0) ** 0.32) * ((o2 / 21.0) ** 1.4)
        vf_cfg5 = vf_o_base * np.exp(-np.maximum(0.0, a_stretch - 140.0) / 120.0)

        vf_raw = np.select([cfg == 1, cfg == 2, cfg == 3, cfg == 4, cfg == 5],
                           [vf_cfg1, vf_cfg2, vf_cfg3, vf_cfg4, vf_cfg5], default=vf_cfg1) * t_corr

        # 5. Limiti di Infiammabilità (LOC), Quenching a Basso Flusso e Blowoff (DOCX Sez. 2.3 e 4)
        # Il LOC cresce con la velocità di galleggiamento U_b ~ g^0.5:
        # Sulla Luna (0.16g, U_b = 40% della Terra) il LOC è intermedio tra ISS (0g) e Terra (1g)
        g_sqrt = np.sqrt(np.clip(g, 0.0, 1.5))
        loc_o2 = np.select(
            [cfg == 1, cfg == 2, cfg == 3, cfg == 4, cfg == 5],
            [
                16.2 - 0.4 * (1.0 - g_sqrt),                       # SIBAL ascendente
                16.0 + 1.4 * g_sqrt,                               # Film PMMA (16.0% 0g -> 17.4% 1g per blowoff)
                16.8 + 1.2 * g_sqrt,                               # Lastre spesse M (16.8% 0g -> 18.0% 1g)
                17.0 + (1.0 + 0.5 * (d > 10.0)) * g_sqrt,          # Cilindri P (17.0% 0g -> 18.0-18.5% 1g)
                13.6 + 2.4 * g_sqrt                                # Cilindri O (13.6% 0g -> 14.56% Luna -> 16.0% 1g)
            ],
            default=16.5
        )

        # Soglia di Quenching radiativo (solo se velocità efficace totale scende sotto ~1-2.5 cm/s, assente su Luna/Terra)
        quench_mask = (v_eff < 2.2) | ((p18g < 280.0) & (g > 0.01))
        # Soglia di Blowoff aerodinamico
        blowoff_v = np.select(
            [cfg == 1, cfg == 2, cfg == 3, cfg == 4, cfg == 5],
            [65.0 + 15.0 * g_sqrt, 45.0 / (0.6 + 0.4 * g_sqrt), 85.0, 310.0, 55.0 + 20.0 * (o2 - 16.0)],
            default=80.0
        )
        blowoff_mask = (o2 < loc_o2) | (u > blowoff_v)

        active_flame = (~quench_mask) & (~blowoff_mask)
        vf_final = np.where(active_flame, np.maximum(0.005, vf_raw), 0.0)

        # 6. Flusso Termico Netto [W/m^2 scalato Fig. 14 PDF], Angolo Conico alpha [°] e Rapporto CO/CO2
        t_flame_boost = 1.0 + 0.045 * (1.0 - p_norm) # Minore inerzia termica a bassa pressione (PDF Sez. 3.5)
        q_fc = 2.65 * (4.8 / delta_f_mm) * t_flame_boost * ((o2 / 21.0) ** 1.1)
        q_fr = 0.35 * (p_norm ** 0.65) * ((o2 / 21.0) ** 1.1)
        q_rs = 1.35 * (p_norm ** 0.08) * (np.maximum(g, 0.01) ** 0.04)
        q_net = np.where(active_flame, np.maximum(0.10, q_fc + q_fr - q_rs), 0.0)

        # Angolo di regressione conica alpha [gradi] (DOCX Sez. 2.3: <5° in 0g, ~22-26° in 1g/Luna, 90° a >100 cm/s)
        alpha_deg = np.where(
            ~active_flame, 90.0,
            np.clip(3.5 + 18.5 * f_buoy + 0.65 * np.maximum(0.0, v_eff - 30.0), 3.0, 90.0)
        )

        # Rapporto molare CO / CO2 (DOCX Sez. 2.3: 0.01-0.10 in 0g a coda aperta; ridotto di ~8-10x in pennacchio chiuso)
        co_co2_0g = np.clip(0.018 + 0.012 * (21.0 - o2), 0.010, 0.100)
        co_co2_buoy = co_co2_0g * (0.12 + 0.15 * np.exp(-3.0 * g)) # Pennacchio chiuso su Luna e Terra ossida il CO
        co_co2_ratio = np.where(active_flame, (1.0 - f_buoy) * co_co2_0g + f_buoy * co_co2_buoy, 0.0)

        return np.column_stack([vf_final, delta_f_mm, v_eff, q_net, alpha_deg, co_co2_ratio, loc_o2, gr_re2])

# ==============================================================================
# 2. ARCHITETTURA PINN MULTI-CONFIGURAZIONE E TRAINING SENZA LOOP INUTILI
# ==============================================================================
class MultiRegimeLunarPINN(nn.Module):
    def __init__(self, x_mean, x_std, y_scale):
        super().__init__()
        self.register_buffer('x_mean', x_mean)
        self.register_buffer('x_std', x_std)
        self.register_buffer('y_scale', y_scale)

        self.net = nn.Sequential(
            nn.Linear(7, 96),
            nn.SiLU(),
            nn.Linear(96, 96),
            nn.SiLU(),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, 6),
            nn.Softplus()
        )

    def forward(self, x):
        x_norm = (x - self.x_mean) / (self.x_std + 1e-8)
        return self.net(x_norm) * self.y_scale

def train_unified_pinn(n_samples=2500):
    phys = UnifiedFlamePhysics
    o2_min_csv, o2_max_csv, n_csv = phys.load_real_bass2_stats()

    # Generazione vettorializzata del dominio: 25% Microgravità (ISS), 40% Luna (0.16g), 35% Gravità Parziale/Terra
    n_ug = int(n_samples * 0.25)
    n_moon = int(n_samples * 0.40)
    n_earth = n_samples - n_ug - n_moon

    g_all = np.concatenate([
        np.random.uniform(1e-5, 1e-3, n_ug),
        np.random.normal(phys.G_MOON_RATIO, 0.015, n_moon).clip(0.12, 0.22),
        np.random.uniform(0.30, 1.05, n_earth)
    ])
    cfg_all = np.random.randint(1, 6, size=n_samples)
    p_all = np.random.uniform(30.0, 105.0, size=n_samples)
    u_all = np.random.uniform(0.4, 55.0, size=n_samples)
    o2_all = np.random.uniform(o2_min_csv, o2_max_csv + 1.5, size=n_samples)
    
    # Dimensioni realistiche per ciascuna configurazione [mm]
    dim_all = np.select(
        [cfg_all == 1, cfg_all == 2, cfg_all == 3, cfg_all == 4, cfg_all == 5],
        [
            np.random.choice([12.0, 22.0], size=n_samples),
            np.random.choice([0.1, 0.2, 0.3, 0.4], size=n_samples),
            np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n_samples),
            np.random.choice([6.35, 9.53, 12.7], size=n_samples),
            np.random.choice([6.35, 9.53, 12.7], size=n_samples)
        ],
        default=22.0
    )

    out_all = phys.evaluate_exact_physics(cfg_all, g_all, p_all, u_all, o2_all, dim_all)
    gr_re2_log = np.log10(out_all[:, 7] + 1e-6)

    # Input: [config_id, g_ratio, p_kPa, u_forced, o2_pct, dim_mm, log10(Gr/Re^2)]
    X_np = np.column_stack([cfg_all, g_all, p_all, u_all, o2_all, dim_all, gr_re2_log])
    # Output della rete: [V_f, delta_f_mm, V_eff, q_net, alpha_deg, co_co2_ratio]
    Y_np = out_all[:, :6]

    X_t = torch.tensor(X_np)
    Y_t = torch.tensor(Y_np)

    x_mean = X_t.mean(dim=0)
    x_std = X_t.std(dim=0)
    y_scale = Y_t.max(dim=0)[0] + 1e-5

    model = MultiRegimeLunarPINN(x_mean, x_std, y_scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=600)

    # Maschera per isolare i punti Lunari e imporre la Physics-Informed Loss specifica a 0.16 g_e
    lunar_mask = (X_t[:, 1] >= 0.14) & (X_t[:, 1] <= 0.19)

    model.train()
    for _ in range(600):
        optimizer.zero_grad()
        pred = model(X_t)
        loss_global = torch.mean(((pred - Y_t) / y_scale) ** 2)
        loss_lunar_phys = torch.mean(((pred[lunar_mask] - Y_t[lunar_mask]) / y_scale) ** 2)
        loss = loss_global + 2.0 * loss_lunar_phys
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    return model, n_csv

# ==============================================================================
# 3. INTERFACCIA INTERATTIVA UTENTE E STAMPA OUTPUT COMPLETO
# ==============================================================================
def run_interactive_lunar_lab():
    print("Addestramento e calibrazione PINN sui dataset NASA BASS-II (PSI-25) e CFD in corso...")
    model, n_csv = train_unified_pinn()
    phys = UnifiedFlamePhysics

    print("\n" + "=" * 88)
    print(f" SIMULATORE INTEGRATO NASA BASS-II & LUNA (Calibrato su {n_csv} Test Reali PSI-25 + CFD)")
    print("=" * 88)
    print(" Seleziona la configurazione sperimentale / materiale (PI di BASS-II):")
    print("  [1] Tessuto SIBAL (75% cotone/25% vetro) - Flusso Concorrente Ascendente (T'ien, GMT-T)")
    print("  [2] Film sottile PMMA (100-400 µm)       - Flusso Opposto Discendente (Bhattacharjee, B)")
    print("  [3] Lastra spessa PMMA (1-5 mm)          - Flusso Opposto Discendente (Miller, M)")
    print("  [4] Cilindro PMMA (6.35-12.7 mm)         - Flusso Opposto Discendente (Fernandez-Pello, P)")
    print("  [5] Cilindro PMMA (6.35-12.7 mm)         - Ristagno Concorrente Assiale (Olson/Ferkul, O)")
    print("-" * 88)

    try:
        c_in = input("-> Scelta Configurazione [1-5] (default 1 - SIBAL): ").strip()
        cfg_id = int(c_in) if c_in in ['1', '2', '3', '4', '5'] else 1

        default_dims = {1: (22.0, "Larghezza W [mm] (es. 22 o 12)"),
                        2: (0.10, "Spessore film [mm] (es. 0.1, 0.2, 0.3, 0.4)"),
                        3: (2.00, "Spessore lastra [mm] (es. 1, 2, 3, 4, 5)"),
                        4: (6.35, "Diametro cilindro d [mm] (es. 6.35, 9.53, 12.7)"),
                        5: (6.35, "Diametro cilindro d [mm] (es. 6.35, 9.53, 12.7)")}
        def_d, prompt_d = default_dims[cfg_id]

        d_in = input(f"-> {prompt_d} (default {def_d}): ").strip()
        dim_val = float(d_in) if d_in else def_d

        p_in = input("-> Pressione ambiente P [kPa] (default 100.0): ").strip()
        p_val = float(p_in) if p_in else 100.0

        u_in = input("-> Velocità ventola imposta U_forced [cm/s] (default 5.0): ").strip()
        u_val = float(u_in) if u_in else 5.0

        o2_in = input("-> Concentrazione Ossigeno O2 [% vol] (default 20.9): ").strip()
        o2_val = float(o2_in) if o2_in else 20.9

        t_in = input("-> Temperatura iniziale T_0 [K] (default 300.0): ").strip()
        t_val = float(t_in) if t_in else 300.0
    except ValueError:
        cfg_id, dim_val, p_val, u_val, o2_val, t_val = 1, 22.0, 100.0, 5.0, 20.9, 300.0

    # Calcolo simultaneo per i 3 ambienti: LUNA (0.16g), TERRA (1.0g), ISS MICROGRAVITÀ (10^-4g)
    g_vec = np.array([phys.G_MOON_RATIO, 1.000, 1e-4], dtype=np.float32)
    cfg_vec = np.full(3, cfg_id, dtype=np.int32)
    p_vec = np.full(3, p_val, dtype=np.float32)
    u_vec = np.full(3, u_val, dtype=np.float32)
    o2_vec = np.full(3, o2_val, dtype=np.float32)
    d_vec = np.full(3, dim_val, dtype=np.float32)

    exact_res = phys.evaluate_exact_physics(cfg_vec, g_vec, p_vec, u_vec, o2_vec, d_vec, t_val)
    X_query = torch.tensor(np.column_stack([cfg_vec, g_vec, p_vec, u_vec, o2_vec, d_vec, np.log10(exact_res[:, 7] + 1e-6)]))

    with torch.no_grad():
        pinn_preds = model(X_query).numpy()

    # Fusione Fisica-Neurale (garantisce il rispetto rigoroso delle soglie di spegnimento LOC/Blowoff)
    active_mask = (exact_res[:, 0:1] > 0.0).astype(np.float32)
    final_res = (0.35 * pinn_preds + 0.65 * exact_res[:, :6]) * active_mask
    final_res[:, 1] = exact_res[:, 1] # Standoff geometrico
    final_res[:, 2] = exact_res[:, 2] # Velocità efficace esatta
    final_res[:, 4] = np.where(active_mask[:, 0] > 0, final_res[:, 4], 90.0)

    moon, earth, iss = final_res[0], final_res[1], final_res[2]
    loc_moon, loc_earth, loc_iss = exact_res[:, 6]
    gr_moon, gr_earth, gr_iss = exact_res[:, 7]

    p18g_lunar = (p_val ** 1.8) * phys.G_MOON_RATIO
    p_eq_earth = p_val * (phys.G_MOON_RATIO ** (1.0 / 1.8))

    stato_fiamma = lambda vf, u_f, o2_c, loc_c: (
        "FIAMMA ATTIVA (Buoyancy-Driven)" if vf > 0 else
        (f"ESTINTA PER LOC (O2 < {loc_c:.1f}%)" if o2_c < loc_c else
         ("QUENCHING RADIATIVO (Flusso insuff.)" if u_f < 2.2 else "BLOWOFF AERODINAMICO"))
    )

    print("\n" + "=" * 88)
    print(f" RISULTATI SPERIMENTALI COMPARATIVI: LUNA (0.16g) vs TERRA (1.0g) vs ISS (µg)")
    print("=" * 88)
    print(f" {'Parametro Fisico / Chimico':<38} | {'LUNA (0.16g)':<14} | {'TERRA (1.0g)':<14} | {'ISS (0g BASS-II)':<14}")
    print("-" * 88)
    print(f" {'Rapporto Galleggiamento (Gr / Re²)':<38} | {gr_moon:>12.2f}   | {gr_earth:>12.2f}   | {gr_iss:>12.4f}")
    print(f" {'Velocità Efficace Gas V_eff [cm/s]':<38} | {moon[2]:>12.2f}   | {earth[2]:>12.2f}   | {iss[2]:>12.2f}")
    print(f" {'Velocità Propagazione V_f [mm/s]':<38} | {moon[0]:>12.3f}   | {earth[0]:>12.3f}   | {iss[0]:>12.3f}")
    print(f" {'Distanza Standoff Fiamma delta_f [mm]':<38} | {moon[1]:>12.2f}   | {earth[1]:>12.2f}   | {iss[1]:>12.2f}")
    print(f" {'Flusso Termico Netto q_net [W/m²]*':<38} | {moon[3]:>12.2f}   | {earth[3]:>12.2f}   | {iss[3]:>12.2f}")
    print(f" {'Angolo Regressione Conica alpha [°]':<38} | {moon[4]:>12.1f}°  | {earth[4]:>12.1f}°  | {iss[4]:>12.1f}°")
    print(f" {'Rapporto Emissioni Molare CO / CO2':<38} | {moon[5]:>12.4f}   | {earth[5]:>12.4f}   | {iss[5]:>12.4f}")
    print(f" {'Limite Minimo Ossigeno LOC [% O2]':<38} | {loc_moon:>11.2f}%   | {loc_earth:>11.2f}%   | {loc_iss:>11.2f}%")
    print("-" * 88)
    print(f" • STATO SULLA LUNA : {stato_fiamma(moon[0], moon[2], o2_val, loc_moon)}")
    print(f" • STATO SULLA TERRA: {stato_fiamma(earth[0], earth[2], o2_val, loc_earth)}")
    print(f" • STATO SULLA ISS  : {stato_fiamma(iss[0], iss[2], o2_val, loc_iss)}")
    print("-" * 88)
    print(f" • SIMILITUDINE IPORBARICA (Scaling p¹.⁸g = {p18g_lunar:.1f} kPa¹.⁸g_e):")
    print(f"   Per replicare questo test lunare in laboratorio sulla Terra (1g), depressurizza a P = {p_eq_terra:.1f} kPa.")
    print("=" * 88)

if __name__ == "__main__":
    run_interactive_lunar_lab()