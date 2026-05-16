"""
================================================================================
  SOLUCIONADOR DE TRANSFERÊNCIA DE CALOR E MASSA – PROVA USP
================================================================================
  Resolve os três itens da prova com base nos três últimos dígitos do Nº USP.
  Requer: pip install CoolProp scipy numpy
================================================================================
"""

import numpy as np
from scipy.optimize import brentq
try:
    from CoolProp.CoolProp import PropsSI
except ImportError:
    raise ImportError("CoolProp não encontrado. Execute: pip install CoolProp")

# ─────────────────────────────────────────────────────────────────
# 1. ENTRADA DOS DADOS
# ─────────────────────────────────────────────────────────────────

def get_digit(name: str, lo: int = 0, hi: int = 9) -> int:
    while True:
        try:
            val = int(input(f"  Digite {name} ({lo}–{hi}): "))
            if lo <= val <= hi:
                return val
        except ValueError:
            pass
        print(f"    ⚠  Valor inválido. Digite um inteiro entre {lo} e {hi}.")


print("=" * 60)
print("  SOLUCIONADOR DE TRANSFERÊNCIA DE CALOR – Nº USP")
print("=" * 60)
print("\nDigite os três últimos dígitos do seu Nº USP:")
D2 = get_digit("D₂ (centena)")
D1 = get_digit("D₁ (dezena)")
D0 = get_digit("D₀ (unidade)")

# ─────────────────────────────────────────────────────────────────
# 2. CONFIGURAÇÃO DO PROBLEMA (Tabela 1)
# ─────────────────────────────────────────────────────────────────

prod = (D1 + 1) * (D0 + 1)

if 12 <= prod <= 28:
    geometry = "Esfera"
    Lc_m = (6 + 2 * D1) / 100          # diâmetro em metros
elif 29 <= prod <= 100:
    geometry = "Placa Plana Vertical"
    Lc_m = (2 + D1) / 100              # comprimento em metros
else:
    geometry = "Cilindro Horizontal"
    Lc_m = (6 + 2 * D1) / 100          # diâmetro em metros

fluid_name = "Water" if D1 % 2 != 0 else "Air"
Ts  = 35 + 3 * D0          # °C
Tinf = 15 + D2              # °C
v_initial = 2 + 0.5 * D0   # m/s  (velocidade inicial para o item 1 e 3)
P = 1e5                     # Pa

Tf = (Ts + Tinf) / 2        # temperatura de filme (°C)
Ts_K   = Ts   + 273.15
Tinf_K = Tinf + 273.15
Tf_K   = Tf   + 273.15

Delta_T = Ts - Tinf         # ΔT positivo → superfície mais quente

print("\n" + "─" * 60)
print(f"  Produto (D₁+1)(D₀+1) = {prod}  →  Geometria: {geometry}")
print(f"  Dimensão característica: {Lc_m*100:.1f} cm  ({Lc_m:.4f} m)")
print(f"  Fluido: {fluid_name}")
print(f"  Ts = {Ts} °C  |  T∞ = {Tinf} °C  |  Tf = {Tf:.1f} °C")
print(f"  Velocidade forçada (itens 1 e 3): v = {v_initial} m/s")
print("─" * 60)

# ─────────────────────────────────────────────────────────────────
# 3. PROPRIEDADES DO FLUIDO (CoolProp na Tf)
# ─────────────────────────────────────────────────────────────────

def get_props(T_K: float, fluid: str = fluid_name, p: float = P) -> dict:
    """Retorna propriedades termodinâmicas e de transporte na temperatura T_K."""
    rho  = PropsSI("D",       "T", T_K, "P", p, fluid)   # kg/m³
    mu   = PropsSI("V",       "T", T_K, "P", p, fluid)   # Pa·s
    k    = PropsSI("L",       "T", T_K, "P", p, fluid)   # W/(m·K)
    cp   = PropsSI("C",       "T", T_K, "P", p, fluid)   # J/(kg·K)
    beta = PropsSI("isobaric_expansion_coefficient",
                              "T", T_K, "P", p, fluid)   # 1/K
    nu   = mu / rho                                        # m²/s
    alpha = k / (rho * cp)                                 # m²/s
    Pr   = nu / alpha                                      # adimensional
    return dict(rho=rho, mu=mu, k=k, cp=cp, beta=beta,
                nu=nu, alpha=alpha, Pr=Pr)

props = get_props(Tf_K)
print("\n📋 Propriedades do fluido em Tf = {:.1f} °C:".format(Tf))
for key, val in props.items():
    print(f"    {key:5s} = {val:.4e}")

# ─────────────────────────────────────────────────────────────────
# 4. CORRELAÇÕES DE NUSSELT
# ─────────────────────────────────────────────────────────────────

# ── Número de Rayleigh ──────────────────────────────────────────
def rayleigh(Lc: float, p: dict) -> float:
    return (9.80665*p["beta"] * abs(Delta_T) * Lc**3) / (p["nu"] * p["alpha"])

# ── Número de Reynolds ──────────────────────────────────────────
def reynolds(v: float, Lc: float, p: dict) -> float:
    return v * Lc / p["nu"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONVECÇÃO NATURAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def Nu_nat_placa(Ra: float, Pr: float) -> float:
    """Churchill & Chu – Placa plana vertical (toda faixa de Ra)."""
    psi = 1 + (0.492 / Pr) ** (9 / 16)
    Nu = (0.825 + (0.387 * Ra ** (1 / 6)) / psi ** (8 / 27)) ** 2
    return Nu


def Nu_nat_cilindro(Ra: float, Pr: float) -> float:
    """Churchill & Chu – Cilindro horizontal (toda faixa de Ra)."""
    psi = 1 + (0.559 / Pr) ** (9 / 16)
    Nu = (0.60 + (0.387 * Ra ** (1 / 6)) / psi ** (8 / 27)) ** 2
    return Nu


def Nu_nat_esfera(Ra: float, Pr: float) -> float:
    """Correlação clássica para esfera (Churchill, 1983)."""
    Nu = 2 + (0.589 * Ra ** (1 / 4)) / (1 + (0.469 / Pr) ** (9 / 16)) ** (4 / 9)
    return Nu


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONVECÇÃO FORÇADA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def Nu_forc_placa(Re: float, Pr: float) -> float:
    """Correlação laminar para placa plana (Re < 5×10⁵)."""
    return 0.664 * Re ** 0.5 * Pr ** (1 / 3)


def Nu_forc_cilindro_churchill(Re: float, Pr: float) -> float:
    """Churchill & Bernstein – Cilindro em escoamento cruzado (toda faixa)."""
    if Re * Pr < 0.2:
        return float("nan")
    term1 = 0.3
    term2_num   = 0.62 * Re ** 0.5 * Pr ** (1 / 3)
    term2_denom = (1 + (0.4 / Pr) ** (2 / 3)) ** 0.25
    term3 = (1 + (Re / 282_000) ** (5 / 8)) ** (4 / 5)
    return term1 + (term2_num / term2_denom) * term3


def Nu_forc_esfera(Re: float, Pr: float, mu_f: float, mu_s: float) -> float:
    """Whitaker – Esfera."""
    return 2 + (0.4 * Re ** 0.5 + 0.06 * Re ** (2 / 3)) * Pr ** 0.4 * (mu_f / mu_s) ** 0.25


# ─────────────────────────────────────────────────────────────────
# 5. CÁLCULO DOS Nu COM AS PROPRIEDADES DA TEMPERATURA DE FILME
# ─────────────────────────────────────────────────────────────────

Ra = rayleigh(Lc_m, props)
Re_init = reynolds(v_initial, Lc_m, props)
Pr = props["Pr"]

print(f"\n  Ra = {Ra:.4e}  |  Re (v={v_initial} m/s) = {Re_init:.4e}  |  Pr = {Pr:.4f}")

# μ na temperatura da superfície (apenas para Whitaker/esfera)
if geometry == "Esfera":
    props_s = get_props(Ts_K)
    mu_s = props_s["mu"]
else:
    mu_s = props["mu"]

# ── Nu natural ──────────────────────────────────────────────────
if geometry == "Placa Plana Vertical":
    Nu_nat = Nu_nat_placa(Ra, Pr)
elif geometry == "Cilindro Horizontal":
    Nu_nat = Nu_nat_cilindro(Ra, Pr)
else:
    Nu_nat = Nu_nat_esfera(Ra, Pr)

# ── Nu forçado (velocidade inicial) ─────────────────────────────
if geometry == "Placa Plana Vertical":
    Nu_forc = Nu_forc_placa(Re_init, Pr)
elif geometry == "Cilindro Horizontal":
    Nu_forc = Nu_forc_cilindro_churchill(Re_init, Pr)
else:
    Nu_forc = Nu_forc_esfera(Re_init, Pr, props["mu"], mu_s)

# ── Coeficientes de transferência de calor ──────────────────────
h_nat  = Nu_nat  * props["k"] / Lc_m
h_forc = Nu_forc * props["k"] / Lc_m

print(f"\n  Nu_nat  = {Nu_nat:.4f}   →  h_nat  = {h_nat:.4f} W/(m²·K)")
print(f"  Nu_forc = {Nu_forc:.4f}   →  h_forc = {h_forc:.4f} W/(m²·K)")

# ─────────────────────────────────────────────────────────────────
# ITEM 1 – Razão h_forç / h_nat
# ─────────────────────────────────────────────────────────────────

ratio_h = h_forc / h_nat

print("\n" + "=" * 60)
print(f"  ITEM 1  →  h_forç / h_nat = {ratio_h:.4f}  (adimensional)")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# ITEM 2 – Velocidade mínima para h_forç/h_comb > 0,99
# (Nu_comb)³ = (Nu_nat)³ + (Nu_forç)³  [mesmo sentido]
# ─────────────────────────────────────────────────────────────────

def Nu_forc_of_v(v: float) -> float:
    """Nu forçado como função de v."""
    Re_v = reynolds(v, Lc_m, props)
    if geometry == "Placa Plana Vertical":
        return Nu_forc_placa(Re_v, Pr)
    elif geometry == "Cilindro Horizontal":
        return Nu_forc_cilindro_churchill(Re_v, Pr)
    else:
        return Nu_forc_esfera(Re_v, Pr, props["mu"], mu_s)


def condition(v: float) -> float:
    """
    Retorna h_forc/h_comb − 0.99.
    Nu_comb³ = Nu_nat³ + Nu_forc³  (mesmo sentido de escoamento natural e forçado)
    """
    Nuf  = Nu_forc_of_v(v)
    Nuc  = (Nu_nat**3 + Nuf**3) ** (1 / 3)
    h_f  = Nuf * props["k"] / Lc_m
    h_c  = Nuc * props["k"] / Lc_m
    return h_f / h_c - 0.99


# Busca na faixa [0.001, 1000] m/s
v_lo, v_hi = 1e-3, 1000.0
try:
    v_min = brentq(condition, v_lo, v_hi, xtol=1e-6, maxiter=500)
    print(f"\n  ITEM 2  →  Velocidade mínima = {v_min:.4f} m/s")
    print(f"  (Condição: h_forç / h_comb > 0,99 → convecção natural desprezível)")
except ValueError:
    # Pode ocorrer se a condição nunca chega a 0,99 no intervalo
    f_lo = condition(v_lo)
    f_hi = condition(v_hi)
    print(f"\n  ITEM 2  →  Não foi possível encontrar raiz em [{v_lo}, {v_hi}] m/s")
    print(f"  f(v_lo)={f_lo:.4f}, f(v_hi)={f_hi:.4f} – verifique as correlações.")
    v_min = None

print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# ITEM 3 – Fluxo de calor médio na convecção forçada
# ─────────────────────────────────────────────────────────────────

q_medio = h_forc * (Ts - Tinf)   # W/m²

print(f"\n  ITEM 3  →  q̄'' = h_forç × ΔT = {h_forc:.4f} × {Ts - Tinf}")
print(f"            q̄'' = {q_medio:.2f} W/m²")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# RESUMO FINAL
# ─────────────────────────────────────────────────────────────────

print("\n╔══════════════════════════════════════════════════════════╗")
print("║                    RESUMO DE RESULTADOS                  ║")
print("╠══════════════════════════════════════════════════════════╣")
print(f"║  Geometria  : {geometry:<43}║")
print(f"║  Fluido     : {fluid_name:<43}║")
print(f"║  Ts / T∞    : {Ts} °C / {Tinf} °C{'':<32}║")
print(f"║  Lc         : {Lc_m*100:.2f} cm{'':<44}║")
print("╠══════════════════════════════════════════════════════════╣")
print(f"║  h_nat      : {h_nat:.4f} W/(m²·K){'':<34}║")
print(f"║  h_forç     : {h_forc:.4f} W/(m²·K)  (v = {v_initial} m/s){'':<19}║")
print("╠══════════════════════════════════════════════════════════╣")
print(f"║  Item 1 – h_forç/h_nat  : {ratio_h:.4f}{'':<30}║")
if v_min is not None:
    print(f"║  Item 2 – v_mín (m/s)   : {v_min:.4f}{'':<30}║")
else:
    print(f"║  Item 2 – v_mín (m/s)   : {'Não encontrada':<37}║")
print(f"║  Item 3 – q̄'' (W/m²)    : {q_medio:.2f}{'':<34}║")
print("╚══════════════════════════════════════════════════════════╝")
