"""
================================================================================
  SOLUCIONADOR DE TRANSFERÊNCIA DE CALOR POR RADIAÇÃO EM CÂMARA TRIDIMENSIONAL
================================================================================
  Autor: Juan 
  
  Descrição:
    Este script calcula os fatores de forma e as radiosidades para uma câmara
    tridimensional retangular (paralelepípedo). Ele determina a taxa líquida
    de transferência de calor por radiação na superfície da base (Superfície 1)
    utilizando o método matricial para resolver o sistema de equações lineares.
================================================================================
"""

import numpy as np
import math

def get_parallel_vf(a, b, c):
    """
    Calcula o fator de forma (F_12) entre dois retângulos idênticos, paralelos e 
    alinhados, com dimensões a x b, separados por uma distância c.
    
    Equação clássica de Howell / Incropera:
      F_12 = (2 / (pi * X * Y)) * { ln[ ( (1 + X^2)*(1 + Y^2) ) / ( 1 + X^2 + Y^2 ) ]^0.5
             + X * (1 + Y^2)^0.5 * arctan( X / (1 + Y^2)^0.5 )
             + Y * (1 + X^2)^0.5 * arctan( Y / (1 + X^2)^0.5 )
             - X * arctan(X) - Y * arctan(Y) }
      onde:
        X = a / c
        Y = b / c
    """
    X = a / c
    Y = b / c
    
    # Termo logarítmico: 0.5 * ln( (1 + X^2)*(1 + Y^2) / (1 + X^2 + Y^2) )
    term_ln = 0.5 * math.log(((1.0 + X**2) * (1.0 + Y**2)) / (1.0 + X**2 + Y**2))
    
    # Termos com arco tangente e raízes
    term_atan_1 = X * math.sqrt(1.0 + Y**2) * math.atan(X / math.sqrt(1.0 + Y**2))
    term_atan_2 = Y * math.sqrt(1.0 + X**2) * math.atan(Y / math.sqrt(1.0 + X**2))
    
    # Termos subtrativos simples
    term_sub_1 = X * math.atan(X)
    term_sub_2 = Y * math.atan(Y)
    
    # Fator de forma resultante
    F = (2.0 / (math.pi * X * Y)) * (term_ln + term_atan_1 + term_atan_2 - term_sub_1 - term_sub_2)
    return F

def get_perpendicular_vf(w, h, l):
    """
    Calcula o fator de forma (F_12) entre dois retângulos perpendiculares que
    compartilham uma aresta comum de comprimento l.
    
    l: comprimento da aresta de interseção.
    w: dimensão da superfície emissora (Área 1 = l * w) perpendicular à aresta.
    h: dimensão da superfície receptora (Área 2 = l * h) perpendicular à aresta.
    
    Equação do Incropera:
      F_12 = (1 / (pi * X)) * { X * arctan(1/X) + Y * arctan(1/Y) - (X^2 + Y^2)^0.5 * arctan(1/(X^2+Y^2)^0.5)
             + 1/4 * ln( [ (1 + X^2)*(1 + Y^2) / (1 + X^2 + Y^2) ]
                         * [ X^2 * (1 + X^2 + Y^2) / ( (1 + X^2)*(X^2 + Y^2) ) ]^(X^2)
                         * [ Y^2 * (1 + X^2 + Y^2) / ( (1 + Y^2)*(X^2 + Y^2) ) ]^(Y^2) ) }
      onde:
        X = w / l
        Y = h / l
    """
    X = w / l
    Y = h / l
    
    # Termos de arco tangente
    term1 = X * math.atan(1.0 / X)
    term2 = Y * math.atan(1.0 / Y)
    term3 = - math.sqrt(X**2 + Y**2) * math.atan(1.0 / math.sqrt(X**2 + Y**2))
    
    # Termos logarítmicos decompostos para evitar instabilidades e overflow
    num_common = 1.0 + X**2 + Y**2
    denom_common = X**2 + Y**2
    
    ln_term1 = 0.25 * math.log(((1.0 + X**2) * (1.0 + Y**2)) / num_common)
    
    val_x = (X**2 * num_common) / ((1.0 + X**2) * denom_common)
    ln_term2 = 0.25 * (X**2) * math.log(val_x)
    
    val_y = (Y**2 * num_common) / ((1.0 + Y**2) * denom_common)
    ln_term3 = 0.25 * (Y**2) * math.log(val_y)
    
    # Fator de forma resultante
    F = (1.0 / (math.pi * X)) * (term1 + term2 + term3 + ln_term1 + ln_term2 + ln_term3)
    return F

def calcular_fatores_forma(L, W, H):
    """
    Calcula a matriz completa de fatores de forma 6x6 da cavidade retangular 3D.
    Mapeamento das superfícies:
      1: Base (inferior, L x W)
      2: Topo (superior, L x W)
      3: Frontal (L x H)
      4: Traseira (L x H)
      5: Lateral Esquerda (W x H)
      6: Lateral Direita (W x H)
    """
    F = np.zeros((6, 6))
    
    # --------------------------------------------------------------------------
    # 1. Superfícies Paralelas
    # --------------------------------------------------------------------------
    # F12 e F21 (Base e Topo): área comum L x W, separados por H
    F[0, 1] = get_parallel_vf(L, W, H)
    F[1, 0] = F[0, 1]
    
    # F34 e F43 (Frontal e Traseira): área comum L x H, separados por W
    F[2, 3] = get_parallel_vf(L, H, W)
    F[3, 2] = F[2, 3]
    
    # F56 e F65 (Laterais Esquerda e Direita): área comum W x H, separados por L
    F[4, 5] = get_parallel_vf(W, H, L)
    F[5, 4] = F[4, 5]
    
    # --------------------------------------------------------------------------
    # 2. Superfícies Perpendiculares (com aresta comum)
    # --------------------------------------------------------------------------
    
    # Superfície 1 (Base, L x W) para as adjacentes:
    F[0, 2] = get_perpendicular_vf(W, H, L)  # 1 -> 3 (aresta L, envia W, recebe H)
    F[0, 3] = get_perpendicular_vf(W, H, L)  # 1 -> 4 (aresta L, envia W, recebe H)
    F[0, 4] = get_perpendicular_vf(L, H, W)  # 1 -> 5 (aresta W, envia L, recebe H)
    F[0, 5] = get_perpendicular_vf(L, H, W)  # 1 -> 6 (aresta W, envia L, recebe H)
    
    # Superfície 2 (Topo, L x W) para as adjacentes:
    F[1, 2] = get_perpendicular_vf(W, H, L)  # 2 -> 3
    F[1, 3] = get_perpendicular_vf(W, H, L)  # 2 -> 4
    F[1, 4] = get_perpendicular_vf(L, H, W)  # 2 -> 5
    F[1, 5] = get_perpendicular_vf(L, H, W)  # 2 -> 6
    
    # Superfície 3 (Frontal, L x H) para as adjacentes:
    F[2, 0] = get_perpendicular_vf(H, W, L)  # 3 -> 1 (aresta L, envia H, recebe W)
    F[2, 1] = get_perpendicular_vf(H, W, L)  # 3 -> 2 (aresta L, envia H, recebe W)
    F[2, 4] = get_perpendicular_vf(L, W, H)  # 3 -> 5 (aresta H, envia L, recebe W)
    F[2, 5] = get_perpendicular_vf(L, W, H)  # 3 -> 6 (aresta H, envia L, recebe W)
    
    # Superfície 4 (Traseira, L x H) para as adjacentes:
    F[3, 0] = get_perpendicular_vf(H, W, L)  # 4 -> 1
    F[3, 1] = get_perpendicular_vf(H, W, L)  # 4 -> 2
    F[3, 4] = get_perpendicular_vf(L, W, H)  # 4 -> 5
    F[3, 5] = get_perpendicular_vf(L, W, H)  # 4 -> 6
    
    # Superfície 5 (Lateral Esquerda, W x H) para as adjacentes:
    F[4, 0] = get_perpendicular_vf(H, L, W)  # 5 -> 1 (aresta W, envia H, recebe L)
    F[4, 1] = get_perpendicular_vf(H, L, W)  # 5 -> 2 (aresta W, envia H, recebe L)
    F[4, 2] = get_perpendicular_vf(W, L, H)  # 5 -> 3 (aresta H, envia W, recebe L)
    F[4, 3] = get_perpendicular_vf(W, L, H)  # 5 -> 4 (aresta H, envia W, recebe L)
    
    # Superfície 6 (Lateral Direita, W x H) para as adjacentes:
    F[5, 0] = get_perpendicular_vf(H, L, W)  # 6 -> 1
    F[5, 1] = get_perpendicular_vf(H, L, W)  # 6 -> 2
    F[5, 2] = get_perpendicular_vf(W, L, H)  # 6 -> 3
    F[5, 3] = get_perpendicular_vf(W, L, H)  # 6 -> 4
    
    return F

def resolver_radiosidades_e_fluxo(F, A, T_C, emissividades, sigma):
    """
    Monta e resolve o sistema de equações lineares para as radiosidades J_i 
    de cada uma das 6 superfícies.
    
    A equação de balanço de radiosidade para a superfície i é:
      J_i = eps_i * sigma * T_i^4 + (1 - eps_i) * sum_{j=1}^N ( F_ij * J_j )
      
    Reorganizando em termos do sistema linear M * J = B:
      J_i - (1 - eps_i) * sum_{j=1}^N ( F_ij * J_j ) = eps_i * sigma * T_i^4
      
    Portanto:
      M_ii = 1 - (1 - eps_i) * F_ii  (como F_ii = 0 para superfícies planas, M_ii = 1)
      M_ij = - (1 - eps_i) * F_ij    (para i != j)
      B_i = eps_i * sigma * T_i^4
    """
    N = len(A)
    T_K = T_C + 273.15  # Conversão de Celsius para Kelvin
    Eb = sigma * (T_K ** 4)  # Poder emissivo de corpo negro
    
    # Inicialização da matriz do sistema e do vetor de termos independentes
    M = np.zeros((N, N))
    B = np.zeros(N)
    
    for i in range(N):
        B[i] = emissividades[i] * Eb[i]
        for j in range(N):
            if i == j:
                M[i, j] = 1.0 - (1.0 - emissividades[i]) * F[i, j]
            else:
                M[i, j] = - (1.0 - emissividades[i]) * F[i, j]
                
    # Resolução do sistema linear M * J = B
    J = np.linalg.solve(M, B)
    
    # Cálculo da taxa líquida de calor Q na superfície 1 (Q1) em Watts
    # Método A: Através do balanço de resistência de superfície:
    #   Q_i = (A_i * eps_i / (1 - eps_i)) * (Eb_i - J_i)
    #   Nota: Se eps_i for 1.0, o termo de resistência é zero (corpo negro)
    if abs(emissividades[0] - 1.0) > 1e-9:
        Q1_metodoA = (A[0] * emissividades[0] / (1.0 - emissividades[0])) * (Eb[0] - J[0])
    else:
        Q1_metodoA = None
        
    # Método B: Através do balanço líquido de troca com as outras superfícies (verificação):
    #   Q_i = A_i * sum_{j=1}^N [ F_ij * (J_i - J_j) ]
    Q1_metodoB = 0.0
    for j in range(N):
        Q1_metodoB += F[0, j] * (J[0] - J[j])
    Q1_metodoB *= A[0]
    
    return J, Eb, T_K, Q1_metodoA, Q1_metodoB

def main():
    # --------------------------------------------------------------------------
    # DADOS DE ENTRADA DO PROBLEMA
    # --------------------------------------------------------------------------
    L = 4.0  # Largura (m)
    W = 8.0  # Comprimento (m)
    H = 3.0  # Altura (m)
    
    # Áreas das superfícies (m²)
    A = np.array([
        L * W,  # Superfície 1: Base (inferior) -> 32 m²
        L * W,  # Superfície 2: Topo (superior) -> 32 m²
        L * H,  # Superfície 3: Frontal -> 12 m²
        L * H,  # Superfície 4: Traseira -> 12 m²
        W * H,  # Superfície 5: Lateral Esquerda -> 24 m²
        W * H   # Superfície 6: Lateral Direita -> 24 m²
    ])
    
    # Temperaturas (°C)
    T_C = np.array([300.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    
    # Emissividades (adimensional)
    emissividades = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    
    # Constante de Stefan-Boltzmann (W/(m²·K⁴))
    sigma = 5.67e-8
    
    # --------------------------------------------------------------------------
    # EXECUÇÃO DO CÁLCULO
    # --------------------------------------------------------------------------
    
    # ITEM A: Fatores de Forma
    F = calcular_fatores_forma(L, W, H)
    
    # ITEM B: Radiosidades e Taxas Líquidas
    J, Eb, T_K, Q1_A, Q1_B = resolver_radiosidades_e_fluxo(F, A, T_C, emissividades, sigma)
    
    # --------------------------------------------------------------------------
    # FORMATAÇÃO E EXIBIÇÃO DE RESULTADOS
    # --------------------------------------------------------------------------
    print("=" * 70)
    print("  RESOLUÇÃO DO PROBLEMA DE RADIAÇÃO EM CAVIDADE PARALELEPÍPEDO 3D")
    print("=" * 70)
    print(f"  Geometria: Largura (L) = {L:.1f} m | Comprimento (W) = {W:.1f} m | Altura (H) = {H:.1f} m")
    print("  Áreas das Superfícies:")
    for i in range(6):
        print(f"    Superfície {i+1}: Área = {A[i]:.1f} m²")
    print("-" * 70)
    
    print("\n" + "=" * 70)
    print("  ITEM A - FATORES DE FORMA SOLICITADOS (formatados com 3 casas decimais)")
    print("=" * 70)
    
    pedidos = [
        ("F12", F[0, 1]), ("F13", F[0, 2]), ("F15", F[0, 4]),
        ("F31", F[2, 0]), ("F34", F[2, 3]), ("F35", F[2, 4]),
        ("F52", F[4, 1]), ("F56", F[4, 5]), ("F65", F[5, 4])
    ]
    
    for nome, valor in pedidos:
        print(f"    Fator de forma {nome} = {valor:.3f}")
        
    print("-" * 70)
    print("  MATRIZ COMPLETA DE FATORES DE FORMA (6x6):")
    print("-" * 70)
    header = "          " + "".join([f"    S{j+1}  " for j in range(6)])
    print(header)
    for i in range(6):
        row_str = f"    S{i+1}    " + "  ".join([f"{F[i, j]:.4f}" for j in range(6)])
        print(row_str)
        
    print("-" * 70)
    print("  VERIFICAÇÃO DA REGRA DA SOMA (Sum(F_ij) deve ser igual a 1.0):")
    for i in range(6):
        row_sum = np.sum(F[i])
        print(f"    Linha {i+1} (Superfície {i+1}): Soma = {row_sum:.6f}")
        
    print("\n" + "=" * 70)
    print("  ITEM B - CONDIÇÕES DE CONTORNO, RADIOSIDADES E TAXA LÍQUIDA Q1")
    print("=" * 70)
    print("  Temperaturas e Poderes Emissivos de Corpo Negro (Eb = sigma * T^4):")
    for i in range(6):
        print(f"    Superfície {i+1}: T = {T_C[i]:.1f} °C ({T_K[i]:.2f} K) | Eb = {Eb[i]:.2f} W/m²")
    print("-" * 70)
    
    print("  Radiosidades Solucionadas (J) via numpy.linalg.solve (em W/m²):")
    for i in range(6):
        print(f"    Radiosidade J{i+1} = {J[i]:.2f} W/m²")
    print("-" * 70)
    
    print("  TAXA LÍQUIDA DE RADIAÇÃO NA SUPERFÍCIE 1 (Q1):")
    print("-" * 70)
    if Q1_A is not None:
        print(f"    Método da Resistência de Superfície: Q1 = {Q1_A:.2f} W")
    print(f"    Método do Balanço Direto de Fluxo  : Q1 = {Q1_B:.2f} W")
    
    diferenca = abs(Q1_A - Q1_B) if Q1_A is not None else 0.0
    print(f"    Diferença absoluta entre métodos    : {diferenca:.4e} W")
    print("=" * 70)
    
if __name__ == "__main__":
    main()
