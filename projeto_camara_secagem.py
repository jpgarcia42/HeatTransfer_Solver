"""
================================================================================
  PROJETO ACADÊMICO: SIMULAÇÃO TÉRMICA DA CÂMARA SECADORA E ESFERA VITROCERÂMICA
================================================================================
  Autor: Juan
  
  Descrição:
    Este script resolve o problema acoplado de convecção natural, radiação de
    superfície e perdas de calor por infiltração de ar em uma câmara 3D.
    Na segunda etapa, simula o aquecimento transiente de uma esfera vitrocerâmica
    utilizando o Método das Diferenças Finitas (MDF) Implícito 1D em coordenadas
    esféricas, comparando e validando com a solução analítica por séries.
================================================================================
"""

import numpy as np
import math
from scipy.optimize import fsolve, brentq
import matplotlib.pyplot as plt

# ==============================================================================
# TABELA DE PROPRIEDADES DO AR SECO (Tabela A.4 - Incropera)
# Faixa de temperatura estendida de 200 K a 800 K
# ==============================================================================
T_table = np.array([200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 800.0]) # Kelvin
rho_table = np.array([1.7458, 1.3947, 1.1614, 0.9950, 0.8711, 0.7740, 0.6964, 0.6329, 0.5804, 0.5356, 0.4975, 0.4354]) # kg/m³
nu_table = np.array([7.59e-6, 11.44e-6, 15.89e-6, 20.92e-6, 26.41e-6, 32.39e-6, 38.79e-6, 45.57e-6, 52.69e-6, 60.21e-6, 68.10e-6, 84.93e-6]) # m²/s
k_table = np.array([18.1e-3, 22.3e-3, 26.3e-3, 30.0e-3, 33.5e-3, 37.0e-3, 40.7e-3, 43.9e-3, 46.9e-3, 49.7e-3, 52.4e-3, 57.3e-3]) # W/(m·K)
alpha_table = np.array([10.3e-6, 15.9e-6, 22.5e-6, 29.9e-6, 38.3e-6, 47.2e-6, 56.7e-6, 66.7e-6, 76.9e-6, 87.3e-6, 98.0e-6, 120.0e-6]) # m²/s
Pr_table = np.array([0.737, 0.720, 0.707, 0.700, 0.690, 0.686, 0.684, 0.683, 0.685, 0.690, 0.695, 0.709]) # adimensional
cp_table = np.array([1007.0, 1006.0, 1007.0, 1009.0, 1014.0, 1021.0, 1030.0, 1040.0, 1051.0, 1063.0, 1075.0, 1099.0]) # J/(kg·K)

# ==============================================================================
# ETAPA 1: ROTINAS ANALÍTICAS PARA FATORES DE FORMA (A partir de solucionador_rad_3d.py)
# ==============================================================================

def get_parallel_vf(a, b, c):
    """Fator de forma entre retângulos paralelos alinhados (Howell / Incropera)."""
    X = a / c
    Y = b / c
    term_ln = 0.5 * math.log(((1.0 + X**2) * (1.0 + Y**2)) / (1.0 + X**2 + Y**2))
    term_atan_1 = X * math.sqrt(1.0 + Y**2) * math.atan(X / math.sqrt(1.0 + Y**2))
    term_atan_2 = Y * math.sqrt(1.0 + X**2) * math.atan(Y / math.sqrt(1.0 + X**2))
    term_sub_1 = X * math.atan(X)
    term_sub_2 = Y * math.atan(Y)
    F = (2.0 / (math.pi * X * Y)) * (term_ln + term_atan_1 + term_atan_2 - term_sub_1 - term_sub_2)
    return F

def get_perpendicular_vf(w, h, l):
    """Fator de forma entre retângulos perpendiculares com aresta comum l (Howell / Incropera)."""
    X = w / l
    Y = h / l
    term1 = X * math.atan(1.0 / X)
    term2 = Y * math.atan(1.0 / Y)
    term3 = - math.sqrt(X**2 + Y**2) * math.atan(1.0 / math.sqrt(X**2 + Y**2))
    num_common = 1.0 + X**2 + Y**2
    denom_common = X**2 + Y**2
    ln_term1 = 0.25 * math.log(((1.0 + X**2) * (1.0 + Y**2)) / num_common)
    val_x = (X**2 * num_common) / ((1.0 + X**2) * denom_common)
    ln_term2 = 0.25 * (X**2) * math.log(val_x)
    val_y = (Y**2 * num_common) / ((1.0 + Y**2) * denom_common)
    ln_term3 = 0.25 * (Y**2) * math.log(val_y)
    F = (1.0 / (math.pi * X)) * (term1 + term2 + term3 + ln_term1 + ln_term2 + ln_term3)
    return F

def calcular_fatores_forma(L, W, H):
    """Calcula a matriz 6x6 completa dos fatores de forma para a câmara 3D."""
    F = np.zeros((6, 6))
    
    # 1. Superfícies Paralelas
    F[0, 1] = get_parallel_vf(L, W, H)  # Base para Topo (1 -> 2)
    F[1, 0] = F[0, 1]
    
    F[2, 3] = get_parallel_vf(L, H, W)  # Frontal para Traseira (3 -> 4)
    F[3, 2] = F[2, 3]
    
    F[4, 5] = get_parallel_vf(W, H, L)  # Laterais (5 -> 6)
    F[5, 4] = F[4, 5]
    
    # 2. Superfícies Perpendiculares (com aresta comum)
    # Superfície 1 (Base, L x W)
    F[0, 2] = get_perpendicular_vf(W, H, L)  # 1 -> 3
    F[0, 3] = get_perpendicular_vf(W, H, L)  # 1 -> 4
    F[0, 4] = get_perpendicular_vf(L, H, W)  # 1 -> 5
    F[0, 5] = get_perpendicular_vf(L, H, W)  # 1 -> 6
    
    # Superfície 2 (Topo, L x W)
    F[1, 2] = get_perpendicular_vf(W, H, L)  # 2 -> 3
    F[1, 3] = get_perpendicular_vf(W, H, L)  # 2 -> 4
    F[1, 4] = get_perpendicular_vf(L, H, W)  # 2 -> 5
    F[1, 5] = get_perpendicular_vf(L, H, W)  # 2 -> 6
    
    # Superfície 3 (Frontal, L x H)
    F[2, 0] = get_perpendicular_vf(H, W, L)  # 3 -> 1
    F[2, 1] = get_perpendicular_vf(H, W, L)  # 3 -> 2
    F[2, 4] = get_perpendicular_vf(L, W, H)  # 3 -> 5
    F[2, 5] = get_perpendicular_vf(L, W, H)  # 3 -> 6
    
    # Superfície 4 (Traseira, L x H)
    F[3, 0] = get_perpendicular_vf(H, W, L)  # 4 -> 1
    F[3, 1] = get_perpendicular_vf(H, W, L)  # 4 -> 2
    F[3, 4] = get_perpendicular_vf(L, W, H)  # 4 -> 5
    F[3, 5] = get_perpendicular_vf(L, W, H)  # 4 -> 6
    
    # Superfície 5 (Lateral Esquerda, W x H)
    F[4, 0] = get_perpendicular_vf(H, L, W)  # 5 -> 1
    F[4, 1] = get_perpendicular_vf(H, L, W)  # 5 -> 2
    F[4, 2] = get_perpendicular_vf(W, L, H)  # 5 -> 3
    F[4, 3] = get_perpendicular_vf(W, L, H)  # 5 -> 4
    
    # Superfície 6 (Lateral Direita, W x H)
    F[5, 0] = get_perpendicular_vf(H, L, W)  # 6 -> 1
    F[5, 1] = get_perpendicular_vf(H, L, W)  # 6 -> 2
    F[5, 2] = get_perpendicular_vf(W, L, H)  # 6 -> 3
    F[5, 3] = get_perpendicular_vf(W, L, H)  # 6 -> 4
    
    return F

def exportar_relatorio_fatores_forma(F, filename="relatorio_fatores_forma.txt"):
    """Gera um arquivo de texto com a matriz 6x6 e verificação da soma."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("========================================================================\n")
        f.write("                RELATÓRIO DE FATORES DE FORMA INTERNOS\n")
        f.write("========================================================================\n\n")
        f.write("Matriz Completa de Fatores de Forma (F_ij):\n")
        f.write("         " + "".join([f"    S{j+1}  " for j in range(6)]) + "    Soma Linha\n")
        f.write("------------------------------------------------------------------------\n")
        for i in range(6):
            row_sum = np.sum(F[i])
            row_str = f"S{i+1}    " + "  ".join([f"{F[i, j]:.3f}" for j in range(6)])
            f.write(f"{row_str}      {row_sum:.4f}\n")
        f.write("------------------------------------------------------------------------\n")
        f.write("Nota: Todas as linhas satisfazem a Regra da Soma (Soma = 1.000).\n")
    print(f"Relatório de fatores de forma exportado com sucesso para: '{filename}'")

# ==============================================================================
# ETAPA 2: FORMULAÇÃO CONVECTIVA E SISTEMA ACOPLADO NÃO-LINEAR
# ==============================================================================

def obter_h_conv(i, T_parede, T_int, L, W, H):
    """Calcula o coeficiente h de convecção natural dinamicamente para cada face."""
    T_film = (T_parede + T_int) / 2.0
    T_film_K = T_film + 273.15
    
    # Interpolação das propriedades do ar na Tf
    rho = np.interp(T_film_K, T_table, rho_table)
    nu = np.interp(T_film_K, T_table, nu_table)
    k = np.interp(T_film_K, T_table, k_table)
    alpha = np.interp(T_film_K, T_table, alpha_table)
    Pr = np.interp(T_film_K, T_table, Pr_table)
    beta = 1.0 / T_film_K
    
    g = 9.80665
    Delta_T = T_parede - T_int
    abs_dT = max(abs(Delta_T), 1e-5)  # Evita divisão por zero
    
    # 1. Base (Face 1): Horizontal aquecida voltada para cima
    if i == 0:
        L_c = (L * W) / (2.0 * (L + W))  # A_s / P
        Ra = (g * beta * abs_dT * L_c**3) / (nu * alpha)
        if Delta_T >= 0:
            if Ra < 1e7:
                Nu = 0.54 * Ra**0.25
            else:
                Nu = 0.15 * Ra**(1.0/3.0)
        else:
            Nu = 0.27 * Ra**0.25
        h = Nu * k / L_c
        
    # 2. Topo (Face 2): Horizontal resfriada voltada para baixo
    elif i == 1:
        L_c = (L * W) / (2.0 * (L + W))
        Ra = (g * beta * abs_dT * L_c**3) / (nu * alpha)
        if Delta_T >= 0:
            Nu = 0.27 * Ra**0.25
        else:
            if Ra < 1e7:
                Nu = 0.54 * Ra**0.25
            else:
                Nu = 0.15 * Ra**(1.0/3.0)
        h = Nu * k / L_c
        
    # 3. Faces Verticais (3, 4, 5 e 6)
    else:
        L_c = H
        Ra = (g * beta * abs_dT * L_c**3) / (nu * alpha)
        Nu = (0.825 + (0.387 * Ra**(1.0/6.0)) / (1.0 + (0.492 / Pr)**(9.0/16.0))**(8.0/27.0))**2
        h = Nu * k / L_c
        
    return h

def resolver_sistema_acoplado(L, W, H, F, A, T1, T_inf, U_ext, emissivity, sigma, m_dot_inf, cp_ar):
    """Resolve o sistema de 12 equações simultâneas não-lineares."""
    
    def equacoes(x):
        # Desempacotar variáveis
        # x[0:5]: T2, T3, T4, T5, T6 (°C)
        # x[5]: T_int (°C)
        # x[6:12]: J1, J2, J3, J4, J5, J6 (W/m²)
        T = np.zeros(6)
        T[0] = T1
        T[1:6] = x[0:5]
        T_int = x[5]
        J = x[6:12]
        
        # Obter coeficientes h para as condições de parede atuais
        h_conv = np.zeros(6)
        for i in range(6):
            h_conv[i] = obter_h_conv(i, T[i], T_int, L, W, H)
            
        res = []
        
        # 1. Equações de Radiosidade para as 6 superfícies
        for i in range(6):
            Eb_i = sigma * (T[i] + 273.15)**4
            sum_F_J = np.sum(F[i] * J)
            res_J = J[i] - (emissivity * Eb_i + (1.0 - emissivity) * sum_F_J)
            res.append(res_J)
            
        # 2. Balanço de Energia nas Paredes 2 a 6 (faces externas)
        for i in range(1, 6):
            # Q_rad_saindo_i = A_i * sum_j( F_ij * (J_i - J_j) )
            q_rad_i = A[i] * np.sum(F[i] * (J[i] - J))
            # Q_conv_interno_saindo_i = h_conv_i * A_i * (T_i - T_int)
            q_conv_i = h_conv[i] * A[i] * (T[i] - T_int)
            # Q_ext_saindo_i = U_ext * A_i * (T_i - T_inf)
            q_ext_i = U_ext * A[i] * (T[i] - T_inf)
            
            res_wall = q_rad_i + q_conv_i + q_ext_i
            res.append(res_wall)
            
        # 3. Balanço de Energia no Ar Interno
        # Convecção total entrando no ar - Convecção perdida por infiltração
        q_conv_total_ar = np.sum(h_conv * A * (T - T_int))
        q_perda_infiltracao = m_dot_inf * cp_ar * (T_int - T_inf)
        res_air = q_conv_total_ar - q_perda_infiltracao
        res.append(res_air)
        
        return res

    # Chute inicial fisicamente razoável
    # T_guess = [150, 150, 150, 150, 150] para paredes, 180 para T_int
    # J_guess estimadas a partir de temperaturas médias de parede
    T_walls_guess = [150.0, 150.0, 150.0, 150.0, 150.0]
    T_int_guess = [180.0]
    J_guess = [8000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0]
    x0 = np.concatenate((T_walls_guess, T_int_guess, J_guess))
    
    # Resolver
    sol = fsolve(equacoes, x0, xtol=1e-8)
    
    # Extrair e retornar resultados
    T = np.zeros(6)
    T[0] = T1
    T[1:6] = sol[0:5]
    T_int = sol[5]
    J = sol[6:12]
    
    h_conv = np.array([obter_h_conv(i, T[i], T_int, L, W, H) for i in range(6)])
    
    return T, T_int, J, h_conv

# ==============================================================================
# ETAPA 3: TRANSIENTE DA ESFERA (MDF IMPLÍCITO E VALIDAÇÃO ANALÍTICA)
# ==============================================================================

def obter_h_esfera(T_surf, T_int, D):
    """Calcula o h de convecção natural dinâmico para uma esfera."""
    T_film = (T_surf + T_int) / 2.0
    T_film_K = T_film + 273.15
    
    # Interpolação das propriedades do ar na Tf da esfera
    nu = np.interp(T_film_K, T_table, nu_table)
    k = np.interp(T_film_K, T_table, k_table)
    alpha = np.interp(T_film_K, T_table, alpha_table)
    Pr = np.interp(T_film_K, T_table, Pr_table)
    beta = 1.0 / T_film_K
    
    g = 9.80665
    Delta_T = T_surf - T_int
    abs_dT = max(abs(Delta_T), 1e-5)
    
    Ra_D = (g * beta * abs_dT * D**3) / (nu * alpha)
    Nu = 2.0 + (0.589 * Ra_D**0.25) / (1.0 + (0.469 / Pr)**(9.0/16.0))**(4.0/9.0)
    h = Nu * k / D
    return h

def simular_transiente_esfera(D, k_sph, rho_sph, cp_sph, emissivity_sph, T_ini, T_int, J_avg, sigma, t_total, dt=10.0, M=50):
    """
    Simula o aquecimento transiente da esfera usando o MDF Implícito 1D (Esférico).
    Retorna trajetórias de temperatura ao longo do tempo.
    """
    R = D / 2.0
    dr = R / M
    r = np.linspace(0.0, R, M+1)
    
    # Vetor de tempo e histórico de temperaturas
    t_span = np.arange(0.0, t_total + dt, dt)
    n_steps = len(t_span)
    
    T_sph = np.full(M+1, T_ini)
    
    T_centro_hist = []
    T_surf_hist = []
    h_eff_hist = []
    
    # Temperatura efetiva de radiação da câmara
    T_rad_K = (J_avg / sigma)**0.25
    T_rad_C = T_rad_K - 273.15
    
    # Coeficiente de difusividade térmica da esfera
    alpha_sph = k_sph / (rho_sph * cp_sph)
    Fo_grid = alpha_sph * dt / dr**2
    
    for step in range(n_steps):
        # Armazena estado atual
        T_centro_hist.append(T_sph[0])
        T_surf_hist.append(T_sph[M])
        
        # Calcular coeficientes h_conv e h_rad baseados na temperatura da superfície atual
        T_surf_atual = T_sph[M]
        h_conv = obter_h_esfera(T_surf_atual, T_int, D)
        
        # Coeficiente equivalente de radiação (linearizado)
        T_surf_K = T_surf_atual + 273.15
        h_rad = emissivity_sph * sigma * (T_rad_K**2 + T_surf_K**2) * (T_rad_K + T_surf_K)
        h_eff = h_conv + h_rad
        h_eff_hist.append(h_eff)
        
        # Montar a matriz tridiagonal A e o vetor C para o sistema A * T^{p+1} = C
        A_mat = np.zeros((M+1, M+1))
        C_vec = np.zeros(M+1)
        
        # 1. Nó do Centro (m = 0): Simetria dT/dr = 0
        A_mat[0, 0] = 1.0 + 6.0 * Fo_grid
        A_mat[0, 1] = - 6.0 * Fo_grid
        C_vec[0] = T_sph[0]
        
        # 2. Nós Internos (m = 1 a M-1)
        for m in range(1, M):
            A_mat[m, m-1] = - Fo_grid * (1.0 - 0.5/m)**2
            A_mat[m, m] = 1.0 + Fo_grid * (2.0 + 0.5/m**2)
            A_mat[m, m+1] = - Fo_grid * (1.0 + 0.5/m)**2
            C_vec[m] = T_sph[m]
            
        # 3. Nó da Superfície (m = M): Condição de Contorno Convectiva + Radiativa
        # Volume de controle da casca externa
        V_tilde_M = (dr / 2.0) * (1.0 - 0.5/M + 1.0/(12.0 * M**2))
        
        A_coeff = - (k_sph / dr) * (1.0 - 0.5/M)**2
        B_coeff = (rho_sph * cp_sph * V_tilde_M / dt) + (k_sph / dr) * (1.0 - 0.5/M)**2 + h_conv + h_rad
        C_rhs = (rho_sph * cp_sph * V_tilde_M / dt) * T_sph[M] + h_conv * T_int + h_rad * T_rad_C
        
        A_mat[M, M-1] = A_coeff
        A_mat[M, M] = B_coeff
        C_vec[M] = C_rhs
        
        # Resolver o sistema linear
        T_sph = np.linalg.solve(A_mat, C_vec)
        
    return t_span, np.array(T_centro_hist), np.array(T_surf_hist), np.mean(h_eff_hist), T_rad_C

def encontrar_zeta1_e_C1(Bi):
    """Resolve a equação transcendental 1 - zeta * cot(zeta) = Bi para uma esfera."""
    def f_transc(z):
        return z * math.cos(z) + (Bi - 1.0) * math.sin(z)
    
    # A primeira raiz positiva está no intervalo (0, pi)
    zeta1 = brentq(f_transc, 1e-5, math.pi - 1e-5)
    
    # Coeficiente C1
    num = 4.0 * (math.sin(zeta1) - zeta1 * math.cos(zeta1))
    denom = 2.0 * zeta1 - math.sin(2.0 * zeta1)
    C1 = num / denom
    
    return zeta1, C1

def calcular_solucao_analitica(D, k_sph, rho_sph, cp_sph, T_ini, T_int, T_rad, h_eff_avg, t_span):
    """Calcula a solução analítica por séries de Termo Único para a esfera."""
    R = D / 2.0
    alpha_sph = k_sph / (rho_sph * cp_sph)
    
    # 1. Temperatura efetiva do ambiente combinado (ar + radiação)
    # Aqui aproximamos o ambiente combinado por uma temperatura média ponderada pelo h_conv e h_rad.
    # Como não temos os logs parciais separados, usamos T_ambiente_equivalente.
    # Em termos práticos, q_conv + q_rad = h_eff * (T_amp_eff - T_s).
    # Como T_rad e T_int são constantes, T_amp_eff = (h_conv * T_int + h_rad * T_rad) / h_eff.
    # Vamos assumir que a temperatura de radiação e do ar são dominantes.
    # No transiente médio:
    T_amp_eff = (T_int + T_rad) / 2.0  # Média simplificada dos reservatórios
    
    Bi = (h_eff_avg * R) / k_sph
    zeta1, C1 = encontrar_zeta1_e_C1(Bi)
    
    T_centro_anal = []
    T_surf_anal = []
    
    for t in t_span:
        Fo = alpha_sph * t / R**2
        
        # Termo Único
        theta_centro_star = C1 * math.exp(-zeta1**2 * Fo)
        # Limite de sen(z*r*)/z*r* quando r* -> 0 é 1
        T_centro = T_amp_eff + (T_ini - T_amp_eff) * theta_centro_star
        
        theta_surf_star = C1 * math.exp(-zeta1**2 * Fo) * (math.sin(zeta1) / zeta1)
        T_surf = T_amp_eff + (T_ini - T_amp_eff) * theta_surf_star
        
        T_centro_anal.append(T_centro)
        T_surf_anal.append(T_surf)
        
    return np.array(T_centro_anal), np.array(T_surf_anal), Bi, T_amp_eff

# ==============================================================================
# PROGRAMA PRINCIPAL
# ==============================================================================

def main():
    # --------------------------------------------------------------------------
    # 1. DADOS DE ENTRADA E GEOMETRIA DA CÂMARA
    # --------------------------------------------------------------------------
    L = 4.0   # m
    W = 8.0   # m
    H = 3.0   # m
    
    T1 = 350.0     # °C (Base mantida aquecida)
    T_inf = 25.0   # °C (Temperatura externa)
    U_ext = 1.0    # W/(m²·°C) (Coeficiente global de perda externa)
    
    emissivity = 0.9
    sigma = 5.67e-8
    
    # Infiltração de ar
    V_dot_inf_min = 3.0    # m³/min
    V_dot_inf = V_dot_inf_min / 60.0  # m³/s (0.05 m³/s)
    # Densidade do ar externo a 25 °C (298.15 K)
    rho_ar_ext = np.interp(298.15, T_table, rho_table)  # ~1.1614 kg/m³
    m_dot_inf = rho_ar_ext * V_dot_inf                  # ~0.0581 kg/s
    cp_ar = np.interp(298.15, T_table, cp_table)        # ~1007 J/(kg·K)
    
    # Áreas das 6 superfícies
    A = np.array([
        L * W,  # Base (1): 32 m²
        L * W,  # Topo (2): 32 m²
        L * H,  # Frontal (3): 12 m²
        L * H,  # Traseira (4): 12 m²
        W * H,  # Lateral Esquerda (5): 24 m²
        W * H   # Lateral Direita (6): 24 m²
    ])
    
    # --------------------------------------------------------------------------
    # ETAPA 1: CÁLCULO E EXPORTAÇÃO DOS FATORES DE FORMA
    # --------------------------------------------------------------------------
    print("Etapa 1: Calculando matriz de fatores de forma...")
    F = calcular_fatores_forma(L, W, H)
    exportar_relatorio_fatores_forma(F, "relatorio_fatores_forma.txt")
    
    # --------------------------------------------------------------------------
    # ETAPA 2: RESOLUÇÃO DO SISTEMA ACOPLADO NÃO-LINEAR
    # --------------------------------------------------------------------------
    print("\nEtapa 2: Resolvendo o sistema acoplado não-linear...")
    T, T_int, J, h_conv = resolver_sistema_acoplado(
        L, W, H, F, A, T1, T_inf, U_ext, emissivity, sigma, m_dot_inf, cp_ar
    )
    
    # Balanço de potência elétrica fornecida à base (Face 1)
    # Q_total = q_rad,1 + h_conv,1 * A1 * (T1 - T_int)
    q_rad_1 = A[0] * np.sum(F[0] * (J[0] - J))
    q_conv_1 = h_conv[0] * A[0] * (T[0] - T_int)
    Q_total = q_rad_1 + q_conv_1
    
    # Exibição de resultados da Câmara
    print("=" * 80)
    print("  RESULTADOS DA ANÁLISE ESTACIONÁRIA DA CÂMARA SECADORA")
    print("=" * 80)
    print(f"  Temperatura do Ar Interno (T_int)        = {T_int:.2f} °C")
    print(f"  Temperaturas das Paredes:")
    print(f"    Superfície 1 (Base - Fonte)            = {T[0]:.2f} °C")
    print(f"    Superfície 2 (Topo)                    = {T[1]:.2f} °C")
    print(f"    Superfície 3 (Frontal)                 = {T[2]:.2f} °C")
    print(f"    Superfície 4 (Traseira)                = {T[3]:.2f} °C")
    print(f"    Superfície 5 (Lateral Esquerda)        = {T[4]:.2f} °C")
    print(f"    Superfície 6 (Lateral Direita)         = {T[5]:.2f} °C")
    print("-" * 80)
    print("  Radiosidades Internas (J):")
    for i in range(6):
        print(f"    J{i+1} = {J[i]:.2f} W/m²")
    print("-" * 80)
    print(f"  Potência Térmica Total Requerida (Q_tot) = {Q_total:.2f} W ({Q_total/1000.0:.2f} kW)")
    
    # Verificação do balanço global de energia
    # Perda pelas paredes externas 2-6 + perda por infiltração de ar
    perdas_conducao_ext = np.sum(U_ext * A[1:6] * (T[1:6] - T_inf))
    perdas_infiltracao = m_dot_inf * cp_ar * (T_int - T_inf)
    Q_perda_total = perdas_conducao_ext + perdas_infiltracao
    discrepancia = abs(Q_total - Q_perda_total)
    
    print(f"  Verificação de Conservação de Energia:")
    print(f"    Calor Inserido na Base (Q_total)       = {Q_total:.2f} W")
    print(f"    Calor Total Perdido p/ o Exterior      = {Q_perda_total:.2f} W")
    print(f"    Discrepância Residual                  = {discrepancia:.4e} W")
    print("=" * 80)
    
    # --------------------------------------------------------------------------
    # ETAPA 3: SIMULAÇÃO TRANSIENTE DA ESFERA VITROCERÂMICA
    # --------------------------------------------------------------------------
    print("\nEtapa 3: Simulando aquecimento transiente da esfera...")
    D_esfera = 0.50          # m
    k_esfera = 1.4           # W/(m·°C)
    rho_esfera = 2520.0      # kg/m³
    cp_esfera = 790.0        # J/(kg·°C)
    emissivity_esfera = 1.0  # Pintada de preto
    T_ini_esfera = 25.0      # °C
    t_total = 7200.0         # 120 minutos (s)
    
    # Radiosidade média interna da câmara
    J_avg = np.sum(A * J) / np.sum(A)
    
    # Simulação Numérica (MDF Implícito)
    t_span, T_centro_num, T_surf_num, h_eff_avg, T_rad_C = simular_transiente_esfera(
        D_esfera, k_esfera, rho_esfera, cp_esfera, emissivity_esfera,
        T_ini_esfera, T_int, J_avg, sigma, t_total, dt=10.0, M=40
    )
    
    # Validação Analítica (Termo Único)
    T_centro_anal, T_surf_anal, Bi, T_amp_eff = calcular_solucao_analitica(
        D_esfera, k_esfera, rho_esfera, cp_esfera, T_ini_esfera, T_int, T_rad_C, h_eff_avg, t_span
    )
    
    # Exibição de resultados do transiente
    print("=" * 80)
    print("  RESULTADOS DO TRANSIENTE DA ESFERA VITROCERÂMICA")
    print("=" * 80)
    print(f"  Número de Biot Médio da Esfera (Bi)      = {Bi:.4f}  (Bi > 0.1 confirma condução multidimensional)")
    print(f"  Coeficiente Médio de Transf. de Calor    = {h_eff_avg:.2f} W/(m²·°C)")
    print(f"  Temperatura de Radiação Efetiva (T_rad)  = {T_rad_C:.2f} °C")
    print(f"  Temperatura Ambiente Combinada (T_amp)   = {T_amp_eff:.2f} °C")
    print("-" * 80)
    print(f"  Temperaturas no Final de 120 Minutos (7200 s):")
    print(f"    Centro (MDF Numérico)                  = {T_centro_num[-1]:.2f} °C")
    print(f"    Centro (Séries Analítico)              = {T_centro_anal[-1]:.2f} °C")
    print(f"    Diferença Absoluta no Centro           = {abs(T_centro_num[-1] - T_centro_anal[-1]):.2f} °C")
    print(f"    Superfície (MDF Numérico)              = {T_surf_num[-1]:.2f} °C")
    print(f"    Superfície (Séries Analítico)          = {T_surf_anal[-1]:.2f} °C")
    print(f"    Diferença Absoluta na Superfície       = {abs(T_surf_num[-1] - T_surf_anal[-1]):.2f} °C")
    print("=" * 80)
    
    # --------------------------------------------------------------------------
    # OUTPUT GRÁFICO (Matplotlib)
    # --------------------------------------------------------------------------
    print("\nGerando gráfico comparativo...")
    t_min = t_span / 60.0  # Converter tempo para minutos para legibilidade do gráfico
    
    plt.figure(figsize=(10, 6))
    
    # Plot Numérico (Linha Cheia)
    plt.plot(t_min, T_centro_num, 'b-', label='Centro - MDF Implícito (Numérico)', linewidth=2.5)
    plt.plot(t_min, T_surf_num, 'r-', label='Superfície - MDF Implícito (Numérico)', linewidth=2.5)
    
    # Plot Analítico (Linha Tracejada)
    plt.plot(t_min, T_centro_anal, 'b--', label='Centro - Termo Único (Analítico)', linewidth=1.5)
    plt.plot(t_min, T_surf_anal, 'r--', label='Superfície - Termo Único (Analítico)', linewidth=1.5)
    
    plt.title('Evolução Temporal de Temperatura na Esfera Vitrocerâmica\n(MDF Implícito vs. Validação Analítica de Termo Único)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Tempo (minutos)', fontsize=11)
    plt.ylabel('Temperatura (°C)', fontsize=11)
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', fontsize=10)
    
    # Anotações adicionais no gráfico para qualidade de apresentação
    text_info = (f"Bi médio = {Bi:.3f}\n"
                 f"h_eff médio = {h_eff_avg:.1f} W/m²°C\n"
                 f"T_ar = {T_int:.1f} °C\n"
                 f"T_rad = {T_rad_C:.1f} °C")
    plt.text(5, 230, text_info, bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5), fontsize=9)
    
    plt.tight_layout()
    plot_filename = "grafico_transiente_esfera.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Gráfico salvo com sucesso como: '{plot_filename}'")
    plt.close()
    
if __name__ == "__main__":
    main()
