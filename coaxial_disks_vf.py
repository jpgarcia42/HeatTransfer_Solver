"""
================================================================================
  CÁLCULO DO FATOR DE FORMA ENTRE DOIS DISCOS PARALELOS COAXIAIS
================================================================================
  Autor: Juan
  
  Descrição:
    Este script calcula o fator de forma de radiação entre dois discos coaxiais
    paralelos utilizando a formulação analítica clássica apresentada no livro 
    "Fundamentos de Transferência de Calor e Massa" de Incropera et al.
    Também compara o resultado analítico com o valor numérico obtido de referência.
================================================================================
"""

import math

def calcular_fator_forma_discos_coaxiais(r_i, r_j, L):
    """
    Calcula o fator de forma F_ij entre dois discos paralelos coaxiais.
    
    Parâmetros:
      r_i: Raio do disco emissor i (m)
      r_j: Raio do disco receptor j (m)
      L: Distância de separação entre os discos (m)
      
    Retorna:
      F_ij: Fator de forma adimensional
      R_i: Parâmetro adimensional r_i / L
      R_j: Parâmetro adimensional r_j / L
      S: Parâmetro intermediário
    """
    # 1. Parâmetros adimensionais
    R_i = r_i / L
    R_j = r_j / L
    
    # 2. Parâmetro intermediário S
    S = 1.0 + (1.0 + R_j**2) / (R_i**2)
    
    # 3. Equação analítica do Fator de Forma F_ij (Incropera)
    termo_sub = S**2 - 4.0 * (r_j / r_i)**2
    
    # Tratamento de segurança para raiz de número negativo
    if termo_sub < 0:
        raise ValueError("O argumento sob a raiz quadrada é negativo. Verifique as dimensões inseridas.")
        
    F_ij = 0.5 * (S - math.sqrt(termo_sub))
    
    return F_ij, R_i, R_j, S

def main():
    # 1. Dados de entrada para o caso teste do relatório
    r_i = 3.0  # Raio do disco i (m)
    r_j = 3.0  # Raio do disco j (m)
    L = 5.0    # Distância de separação (m)
    
    # Valor de referência obtido via site do Heat Group
    f_ij_heat_group = 0.21937509
    
    # 2. Execução do cálculo analítico
    F_ij, R_i, R_j, S = calcular_fator_forma_discos_coaxiais(r_i, r_j, L)
    
    # 3. Cálculo dos erros relativos e absolutos
    diferenca_absoluta = abs(F_ij - f_ij_heat_group)
    erro_relativo_incropera = (diferenca_absoluta / F_ij) * 100.0
    erro_relativo_heat_group = (diferenca_absoluta / f_ij_heat_group) * 100.0
    
    # 4. Exibição dos resultados no terminal
    print("=" * 80)
    print("  CÁLCULO DO FATOR DE FORMA ENTRE DOIS DISCOS COAXIAIS PARALELOS")
    print("=" * 80)
    print("  Geometria:")
    print(f"    Raio do disco emissor (r_i)   = {r_i:.4f} m")
    print(f"    Raio do disco receptor (r_j)  = {r_j:.4f} m")
    print(f"    Distância de separação (L)    = {L:.4f} m")
    print("-" * 80)
    
    print("  Parâmetros Intermediários da Formulação:")
    print(f"    Razão dimensional R_i (r_i/L) = {R_i:.6f}")
    print(f"    Razão dimensional R_j (r_j/L) = {R_j:.6f}")
    print(f"    Parâmetro geométrico S        = {S:.6f}")
    print("-" * 80)
    
    print("  Fator de Forma Calculado (Incropera):")
    print(f"    F_ij (analítico)              = {F_ij:.8f}")
    print("-" * 80)
    
    print("  Comparação com o site Heat Group:")
    print(f"    F_ij (Heat Group)             = {f_ij_heat_group:.8f}")
    print(f"    Diferença Absoluta            = {diferenca_absoluta:.8f}")
    print(f"    Erro Relativo (base Incropera) = {erro_relativo_incropera:.4f} %")
    print(f"    Erro Relativo (base Heat Group)= {erro_relativo_heat_group:.4f} %")
    print("=" * 80)

if __name__ == "__main__":
    main()
