"""
MMSD v3.2.3 (RVT Engine) - Código de Referência e Validação
Autor: Jamaico Silva Ribeiro
Descrição: Motor analítico para estimativa de PGA baseado na Teoria das
           Vibrações Aleatórias (RVT) e integração da Equação Autoral
           de Amplificação de Solo (760/VS30)^0.43.
Licença: Creative Commons Attribution (CC BY 4.0)
"""

import numpy as np
import time

def mmsd_v323_calcular_pga(mw, dist_hipo_km, vs30, dsigma_pa=26.33e6, 
                           q0=180.0, eta=0.45, kappa0=0.035, 
                           rho=2700.0, vs=3500.0, rad=0.55):
    """
    Calcula a Aceleração de Pico do Solo (PGA em cm/s²) via RVT.
    """
    # 1. Momento Sísmico (N.m)
    m0 = 10.0 ** (1.5 * mw + 9.1)
    
    # 2. Saturação Geométrica (Boore & Atkinson)
    h0 = np.exp(-0.59 + 0.29 * mw)
    r_eq_m = np.sqrt(dist_hipo_km ** 2 + h0 ** 2) * 1000.0
    
    # 3. Frequência de Canto de Brune (Hz)
    fc = 0.49 * vs * ((dsigma_pa / m0) ** (1.0 / 3.0))
    
    # 4. Equação Autoral de Amplificação de Solo (Gama = 0.43)
    gamma = 0.43
    fator_solo = (760.0 / vs30) ** gamma
    
    # 5. Vetor de Frequências para Integração Espectral (0.1 a 50 Hz)
    freqs = np.linspace(0.1, 50.0, 500)
    df = freqs[1] - freqs[0]
    
    # Termos do Espectro de Fourier
    termo_fonte = m0 / (1.0 + (freqs / fc) ** 2)
    
    # Saturação de área de fonte para grandes magnitudes (Mw > 5.5)
    if mw > 5.5:
        fator_saturacao = 1.0 / (1.0 + 10.0 ** (0.75 * (mw - 5.5)))
        termo_fonte *= fator_saturacao
        
    termo_geo = (rad * 2.0) / (4.0 * np.pi * rho * (vs ** 3) * r_eq_m)
    
    q_f = q0 * (freqs ** eta)
    termo_atenuacao = np.exp(-(np.pi * freqs * r_eq_m) / (q_f * vs)) * np.exp(-np.pi * kappa0 * freqs)
    
    # Espectro de Aceleração (m/s²) com a Equação Autoral de Solo
    a_f = ((2.0 * np.pi * freqs) ** 2) * termo_geo * termo_fonte * termo_atenuacao * fator_solo
    
    # 6. Momentos Espectrais RVT (Parseval)
    m0_espectral = 2.0 * np.sum((a_f ** 2) * df)
    m2_espectral = 2.0 * np.sum(((2.0 * np.pi * freqs) ** 2) * (a_f ** 2) * df)
    
    if m0_espectral <= 0.0 or m2_espectral <= 0.0:
        return 0.0, fator_solo
        
    # 7. Duração Estocástica e Fator de Pico (Cartwright & Longuet-Higgins)
    t_dura = (1.0 / fc) + 0.05 * (r_eq_m / 1000.0)
    f_zero = (1.0 / (2.0 * np.pi)) * np.sqrt(m2_espectral / m0_espectral)
    n_e = max(2.0, 2.0 * f_zero * t_dura)
    
    fator_pico = np.sqrt(2.0 * np.log(n_e)) + (0.5772156 / np.sqrt(2.0 * np.log(n_e)))
    
    # Aceleração em m/s² convertida para cm/s²
    pga_m_s2 = np.sqrt(m0_espectral) * fator_pico
    pga_cm_s2 = pga_m_s2 * 100.0
    
    return pga_cm_s2, fator_solo

def executar_suite_testes():
    """Executa a bateria de testes e benchmarks do modelo."""
    print("=" * 70)
    print(" SUÍTE DE TESTES E VALIDAÇÃO - ENGINE MMSD v3.2.3 (RVT)")
    print("=" * 70)
    
    cenarios = [
        {"nome": "Caso Base ESM (Solo Macio)", "mw": 4.4, "dist": 9.45, "vs30": 190.0, "obs": 17.53},
        {"nome": "Rocha Firme (Referência)", "mw": 4.4, "dist": 9.45, "vs30": 760.0, "obs": None},
        {"nome": "Campo Próximo (Severo)", "mw": 5.0, "dist": 2.00, "vs30": 250.0, "obs": None},
        {"nome": "Grande Sismo (Mw 6.5)", "mw": 6.5, "dist": 15.00, "vs30": 360.0, "obs": None},
        {"nome": "Campo Distante (50 km)", "mw": 5.5, "dist": 50.00, "vs30": 270.0, "obs": None},
    ]
    
    print(f"{'Cenário':<28} | {'VS30':<6} | {'Fator Solo':<10} | {'PGA Sim (cm/s²)':<16} | {'Latência':<8}")
    print("-" * 75)
    
    for c in cenarios:
        t0 = time.perf_counter()
        pga, f_solo = mmsd_v323_calcular_pga(c["mw"], c["dist"], c["vs30"])
        dt_ms = (time.perf_counter() - t0) * 1000.0
        
        print(f"{c['nome']:<28} | {c['vs30']:<6.0f} | {f_solo:<10.3f} | {pga:<16.2f} | {dt_ms:<6.2f} ms")
        
        if c["obs"] is not None:
            res_log = np.log(pga) - np.log(c["obs"])
            print(f"  └─> [Validação ESM] Observado: {c['obs']} cm/s² | Resíduo Log: {res_log:+.4f}")
            
    print("=" * 70)

if __name__ == "__main__":
    executar_suite_testes()
