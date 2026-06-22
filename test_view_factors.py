import numpy as np
import math

def get_parallel_vf(a, b, c):
    """
    View factor between two identical parallel rectangles of dimensions a x b,
    separated by a distance c.
    """
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
    """
    View factor between two perpendicular rectangles sharing a common edge.
    l: length of the common edge.
    w: second dimension of the sending surface (area is l * w).
    h: second dimension of the receiving surface (area is l * h).
    Returns F_12 (from sending to receiving).
    """
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

# Chamber dimensions
L = 4.0
W = 8.0
H = 3.0

# Areas
A = np.zeros(6)
A[0] = L * W  # Base: 32 m^2
A[1] = L * W  # Top: 32 m^2
A[2] = L * H  # Front: 12 m^2
A[3] = L * H  # Rear: 12 m^2
A[4] = W * H  # Left: 24 m^2
A[5] = W * H  # Right: 24 m^2

# Initialize 6x6 View Factor Matrix
F = np.zeros((6, 6))

# Parallel pairs:
# F_12 and F_21 (Base and Top)
F[0, 1] = get_parallel_vf(L, W, H)
F[1, 0] = F[0, 1]

# F_34 and F_43 (Front and Rear)
F[2, 3] = get_parallel_vf(L, H, W)
F[3, 2] = F[2, 3]

# F_56 and F_65 (Left and Right)
F[4, 5] = get_parallel_vf(W, H, L)
F[5, 4] = F[4, 5]

# Perpendicular pairs:
# Surface 1 (Base) to adjacent:
F[0, 2] = get_perpendicular_vf(W, H, L) # 1 to 3 (common edge L, send W, recv H)
F[0, 3] = get_perpendicular_vf(W, H, L) # 1 to 4 (common edge L, send W, recv H)
F[0, 4] = get_perpendicular_vf(L, H, W) # 1 to 5 (common edge W, send L, recv H)
F[0, 5] = get_perpendicular_vf(L, H, W) # 1 to 6 (common edge W, send L, recv H)

# Surface 2 (Top) to adjacent:
F[1, 2] = get_perpendicular_vf(W, H, L) # 2 to 3
F[1, 3] = get_perpendicular_vf(W, H, L) # 2 to 4
F[1, 4] = get_perpendicular_vf(L, H, W) # 2 to 5
F[1, 5] = get_perpendicular_vf(L, H, W) # 2 to 6

# Surface 3 (Front) to adjacent:
F[2, 0] = get_perpendicular_vf(H, W, L) # 3 to 1 (common edge L, send H, recv W)
F[2, 1] = get_perpendicular_vf(H, W, L) # 3 to 2 (common edge L, send H, recv W)
F[2, 4] = get_perpendicular_vf(L, W, H) # 3 to 5 (common edge H, send L, recv W)
F[2, 5] = get_perpendicular_vf(L, W, H) # 3 to 6 (common edge H, send L, recv W)

# Surface 4 (Rear) to adjacent:
F[3, 0] = get_perpendicular_vf(H, W, L) # 4 to 1
F[3, 1] = get_perpendicular_vf(H, W, L) # 4 to 2
F[3, 4] = get_perpendicular_vf(L, W, H) # 4 to 5
F[3, 5] = get_perpendicular_vf(L, W, H) # 4 to 6

# Surface 5 (Left) to adjacent:
F[4, 0] = get_perpendicular_vf(H, L, W) # 5 to 1 (common edge W, send H, recv L)
F[4, 1] = get_perpendicular_vf(H, L, W) # 5 to 2 (common edge W, send H, recv L)
F[4, 2] = get_perpendicular_vf(W, L, H) # 5 to 3 (common edge H, send W, recv L)
F[4, 3] = get_perpendicular_vf(W, L, H) # 5 to 4 (common edge H, send W, recv L)

# Surface 6 (Right) to adjacent:
F[5, 0] = get_perpendicular_vf(H, L, W) # 6 to 1
F[5, 1] = get_perpendicular_vf(H, L, W) # 6 to 2
F[5, 2] = get_perpendicular_vf(W, L, H) # 6 to 3
F[5, 3] = get_perpendicular_vf(W, L, H) # 6 to 4

# Check row sums (should be 1.0)
print("=== VERIFICAÇÃO DA REGRA DA SOMA ===")
for i in range(6):
    row_sum = np.sum(F[i])
    print(f"Superfície {i+1}: Soma = {row_sum:.6f}")

# Check reciprocity
print("\n=== VERIFICAÇÃO DA RECIPROCIDADE ===")
recip_errors = 0
for i in range(6):
    for j in range(6):
        val1 = A[i] * F[i, j]
        val2 = A[j] * F[j, i]
        diff = abs(val1 - val2)
        if diff > 1e-6:
            print(f"Falha na reciprocidade ({i+1}->{j+1}): A{i+1}*F{i+1}{j+1} = {val1:.6f}, A{j+1}*F{j+1}{i+1} = {val2:.6f}, dif = {diff:.6e}")
            recip_errors += 1
if recip_errors == 0:
    print("Reciprocidade satisfeita para todos os pares!")

print("\n=== VALORES PEDIDOS PELO ENUNCIADO ===")
pedidos = [
    ("F12", F[0, 1]), ("F13", F[0, 2]), ("F15", F[0, 4]),
    ("F31", F[2, 0]), ("F34", F[2, 3]), ("F35", F[2, 4]),
    ("F52", F[4, 1]), ("F56", F[4, 5]), ("F65", F[5, 4])
]
for nome, val in pedidos:
    print(f"{nome} = {val:.4f}")
