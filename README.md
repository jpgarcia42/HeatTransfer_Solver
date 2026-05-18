# 🌡️ Solucionador de Transferência de Calor — USP

Resolve o teste de Transferência de Calor e Massa baseado nos **três últimos dígitos do Nº USP**.

O arquivo transcal_solver.py é mais preciso que o site apresentado abaixo:
🔗 **[Abrir online (GitHub Pages)](https://jpgarcia42.github.io/HeatTransfer_Solver/)**

## O que calcula

| Item | Resultado |
|------|-----------|
| **1** | Razão h_forçado / h_natural (adimensional) |
| **2** | Velocidade mínima v_mín (m/s) para convecção natural desprezível |
| **3** | Fluxo de calor médio q̄'' (W/m²) |

## Modos de operação

| Modo | Propriedades do fluido | Como usar |
|------|----------------------|-----------|
| **Online** (GitHub Pages) | Correlações polinomiais (Incropera, Holman) | Abrir o link acima |
| **Local + CoolProp** | Valores exatos via CoolProp | Rodar `iniciar_servidor.bat` e abrir `index.html` |

## Como rodar localmente com CoolProp

```bash
pip install CoolProp django
```

Depois clique duas vezes em **`iniciar_servidor.bat`** e abra `index.html` no navegador.

## Correlações implementadas

**Convecção Natural:**
- Churchill & Chu — Placa Plana Vertical
- Churchill & Chu — Cilindro Horizontal
- Churchill (1983) — Esfera

**Convecção Forçada:**
- Laminar (0.664 Re¹/² Pr¹/³) — Placa Plana
- Churchill & Bernstein — Cilindro
- Whitaker (com μ_s) — Esfera

**Item 2:** Nu³_comb = Nu³_nat + Nu³_forç (mesmo sentido), resolvido por bisseção.
