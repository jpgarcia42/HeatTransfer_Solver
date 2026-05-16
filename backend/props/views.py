"""
Django view that returns CoolProp fluid properties as JSON.
Endpoint: GET /props/?fluid=Air&T=40&P=100000
           GET /props/?fluid=Water&T=55&P=100000
"""
import json
from django.http import JsonResponse, HttpResponseBadRequest
from CoolProp.CoolProp import PropsSI


def fluid_props(request):
    # ── CORS headers (allow the HTML file opened locally to call us) ──
    fluid = request.GET.get('fluid', 'Air')
    T_str = request.GET.get('T', None)      # °C
    P_str = request.GET.get('P', '100000')  # Pa

    if T_str is None:
        return HttpResponseBadRequest('Parâmetro T (temperatura em °C) obrigatório.')

    try:
        T_C = float(T_str)
        P   = float(P_str)
    except ValueError:
        return HttpResponseBadRequest('T e P devem ser números.')

    # Map frontend name to CoolProp name
    fluid_map = {'Water': 'Water', 'Air': 'Air'}
    cp_fluid = fluid_map.get(fluid)
    if cp_fluid is None:
        return HttpResponseBadRequest(f'Fluido "{fluid}" não suportado. Use Air ou Water.')

    T_K = T_C + 273.15

    try:
        rho   = PropsSI('D',  'T', T_K, 'P', P, cp_fluid)   # kg/m³
        mu    = PropsSI('V',  'T', T_K, 'P', P, cp_fluid)   # Pa·s
        k     = PropsSI('L',  'T', T_K, 'P', P, cp_fluid)   # W/(m·K)
        cp_v  = PropsSI('C',  'T', T_K, 'P', P, cp_fluid)   # J/(kg·K)
        beta  = PropsSI('isobaric_expansion_coefficient',
                              'T', T_K, 'P', P, cp_fluid)   # 1/K
        nu    = mu / rho                                      # m²/s
        alpha = k / (rho * cp_v)                             # m²/s
        Pr    = nu / alpha

        data = {
            'fluid':  fluid,
            'T_C':    T_C,
            'T_K':    T_K,
            'P':      P,
            'rho':    rho,
            'mu':     mu,
            'k':      k,
            'cp':     cp_v,
            'beta':   beta,
            'nu':     nu,
            'alpha':  alpha,
            'Pr':     Pr,
        }
    except Exception as e:
        return HttpResponseBadRequest(f'Erro CoolProp: {e}')

    response = JsonResponse(data)
    # Allow cross-origin requests from file:// pages
    response['Access-Control-Allow-Origin'] = '*'
    return response
