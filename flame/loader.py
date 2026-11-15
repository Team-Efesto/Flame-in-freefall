"""
EFESTO — Custom Loading Indicator (Hammer & Anvil Pure SVG + CSS)
Modulo per l'animazione di forgiatura dati della piattaforma EFESTO.
Zero dipendenze esterne, 100% offline e compatibile con Streamlit.
"""

def get_efesto_loader_html(message: str = "Forging the data...", height: int = 190) -> str:
    """Restituisce il frammento HTML/CSS con l'animazione del martello che batte sull'incudine."""
    return f"""
<div class="efesto-forge-container" style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding:1.2rem; margin:1rem 0; width:100%;">
  <style>
    @keyframes hammerSwing {{
      0% {{
        transform: rotate(-38deg);
      }}
      30% {{
        transform: rotate(-44deg);
      }}
      46% {{
        transform: rotate(0deg);
      }}
      52% {{
        transform: rotate(-10deg);
      }}
      60% {{
        transform: rotate(0deg);
      }}
      78% {{
        transform: rotate(-34deg);
      }}
      100% {{
        transform: rotate(-38deg);
      }}
    }}

    @keyframes sparkBurst1 {{
      0%, 45% {{ opacity: 0; transform: translate(0, 0) scale(0.2); }}
      47% {{ opacity: 1; transform: translate(-14px, -18px) scale(1.4); }}
      65% {{ opacity: 0; transform: translate(-28px, -28px) scale(0.4); }}
      100% {{ opacity: 0; }}
    }}

    @keyframes sparkBurst2 {{
      0%, 45% {{ opacity: 0; transform: translate(0, 0) scale(0.2); }}
      47% {{ opacity: 1; transform: translate(14px, -20px) scale(1.3); }}
      65% {{ opacity: 0; transform: translate(26px, -32px) scale(0.3); }}
      100% {{ opacity: 0; }}
    }}

    @keyframes sparkBurst3 {{
      0%, 45% {{ opacity: 0; transform: translate(0, 0) scale(0.2); }}
      47% {{ opacity: 1; transform: translate(-4px, -26px) scale(1.5); }}
      65% {{ opacity: 0; transform: translate(-8px, -42px) scale(0.2); }}
      100% {{ opacity: 0; }}
    }}

    @keyframes sparkBurst4 {{
      0%, 45% {{ opacity: 0; transform: translate(0, 0) scale(0.2); }}
      47% {{ opacity: 1; transform: translate(22px, -10px) scale(1.2); }}
      66% {{ opacity: 0; transform: translate(38px, -18px) scale(0.2); }}
      100% {{ opacity: 0; }}
    }}

    @keyframes impactFlash {{
      0%, 44% {{ opacity: 0; transform: scale(0.4); }}
      46% {{ opacity: 1; transform: scale(1.5); }}
      56% {{ opacity: 0; transform: scale(2.2); }}
      100% {{ opacity: 0; }}
    }}

    @keyframes anvilGlow {{
      0%, 44% {{ filter: drop-shadow(0 0 4px rgba(245, 158, 11, 0.2)); }}
      47% {{ filter: drop-shadow(0 0 16px rgba(245, 158, 11, 0.95)) drop-shadow(0 0 28px rgba(239, 68, 68, 0.8)); }}
      70% {{ filter: drop-shadow(0 0 6px rgba(245, 158, 11, 0.3)); }}
      100% {{ filter: drop-shadow(0 0 4px rgba(245, 158, 11, 0.2)); }}
    }}

    @keyframes textPulse {{
      0%, 100% {{ opacity: 0.82; color: #f59e0b; text-shadow: 0 0 8px rgba(245,158,11,0.25); }}
      50% {{ opacity: 1; color: #fbbf24; text-shadow: 0 0 16px rgba(245,158,11,0.65); }}
    }}

    .efesto-hammer-group {{
      transform-origin: 148px 48px;
      animation: hammerSwing 1.35s cubic-bezier(0.25, 0.1, 0.25, 1) infinite;
    }}

    .efesto-anvil-path {{
      animation: anvilGlow 1.35s ease-in-out infinite;
    }}

    .efesto-spark-1 {{ animation: sparkBurst1 1.35s ease-out infinite; }}
    .efesto-spark-2 {{ animation: sparkBurst2 1.35s ease-out infinite; }}
    .efesto-spark-3 {{ animation: sparkBurst3 1.35s ease-out infinite; }}
    .efesto-spark-4 {{ animation: sparkBurst4 1.35s ease-out infinite; }}
    .efesto-flash {{ animation: impactFlash 1.35s ease-out infinite; transform-origin: 104px 92px; }}

    .efesto-loader-label {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      margin-top: 0.85rem;
      animation: textPulse 2s ease-in-out infinite;
      text-align: center;
    }}
  </style>

  <svg viewBox="0 0 220 150" width="165" height="112" style="overflow: visible;">
    <defs>
      <!-- Gradient Incudine -->
      <linearGradient id="anvilGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#334155" />
        <stop offset="50%" stop-color="#1e293b" />
        <stop offset="100%" stop-color="#0f172a" />
      </linearGradient>

      <!-- Gradient Piano Superiore Incudine (Incandescente) -->
      <linearGradient id="anvilTopGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#f59e0b" />
        <stop offset="45%" stop-color="#ef4444" />
        <stop offset="70%" stop-color="#f97316" />
        <stop offset="100%" stop-color="#f59e0b" />
      </linearGradient>

      <!-- Gradient Testa Martello -->
      <linearGradient id="hammerHeadGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#94a3b8" />
        <stop offset="40%" stop-color="#475569" />
        <stop offset="100%" stop-color="#1e293b" />
      </linearGradient>

      <!-- Gradient Manico -->
      <linearGradient id="handleGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#d97706" />
        <stop offset="50%" stop-color="#b45309" />
        <stop offset="100%" stop-color="#78350f" />
      </linearGradient>

      <!-- Bagliore Flash Impatto -->
      <radialGradient id="flashGrad" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#ffffff" stop-opacity="1" />
        <stop offset="35%" stop-color="#fbbf24" stop-opacity="0.9" />
        <stop offset="70%" stop-color="#ef4444" stop-opacity="0.5" />
        <stop offset="100%" stop-color="#f59e0b" stop-opacity="0" />
      </radialGradient>
    </defs>

    <!-- Ombra Base -->
    <ellipse cx="110" cy="140" rx="65" ry="7" fill="#030712" opacity="0.6" />

    <!-- 1. INCUDINE STILIZZATA -->
    <g class="efesto-anvil-path">
      <!-- Base dell'incudine -->
      <path d="M 60 136 L 160 136 L 150 120 L 70 120 Z" fill="url(#anvilGrad)" stroke="#475569" stroke-width="1.5" />
      <!-- Fusto centrale -->
      <path d="M 85 120 L 135 120 L 138 98 L 82 98 Z" fill="url(#anvilGrad)" />
      <!-- Corpo superiore e corno sinistro -->
      <path d="M 50 94 Q 68 96 80 98 L 145 98 Q 155 98 158 104 L 162 104 Q 165 92 145 92 L 68 92 Q 45 92 34 88 Q 45 93 50 94 Z" fill="url(#anvilGrad)" stroke="#475569" stroke-width="1.2" />
      <!-- Piano battente dell'incudine (superficie forgiatura) -->
      <rect x="74" y="90" width="62" height="4.5" rx="1.5" fill="url(#anvilTopGrad)" stroke="#fbbf24" stroke-width="0.8" />
      <!-- Zona calda luminescente -->
      <line x1="88" y1="91" x2="120" y2="91" stroke="#ffffff" stroke-width="1.6" stroke-linecap="round" opacity="0.9" />
    </g>

    <!-- 2. SCINTILLE & FLASH ALL'IMPATTO (punto d'impatto x=104, y=92) -->
    <circle class="efesto-flash" cx="104" cy="91" r="14" fill="url(#flashGrad)" />
    <circle class="efesto-spark-1" cx="104" cy="91" r="2.8" fill="#ffffff" />
    <circle class="efesto-spark-2" cx="104" cy="91" r="2.4" fill="#fbbf24" />
    <circle class="efesto-spark-3" cx="104" cy="91" r="2.6" fill="#f97316" />
    <circle class="efesto-spark-4" cx="104" cy="91" r="2.2" fill="#ef4444" />

    <!-- 3. MARTELLO OSCILLANTE (pivot x=148, y=48) -->
    <g class="efesto-hammer-group">
      <!-- Manico in legno forgiato -->
      <rect x="100" y="44" width="68" height="6.5" rx="3" fill="url(#handleGrad)" stroke="#92400e" stroke-width="0.8" transform="rotate(-18 100 44)" />
      <!-- Impugnatura fascettata -->
      <rect x="142" y="32" width="22" height="7.5" rx="2" fill="#451a03" stroke="#f59e0b" stroke-width="0.7" transform="rotate(-18 142 32)" />
      <!-- Cuneo / Collare di fissaggio -->
      <rect x="98" y="48" width="6" height="12" rx="1" fill="#64748b" stroke="#334155" stroke-width="0.8" />
      <!-- Testa pesante del martello -->
      <path d="M 88 43 L 114 43 L 117 62 L 85 62 Z" fill="url(#hammerHeadGrad)" stroke="#64748b" stroke-width="1.4" />
      <!-- Superficie battente inferiore del martello -->
      <rect x="86" y="60" width="29" height="4" rx="1" fill="#f59e0b" stroke="#fbbf24" stroke-width="0.6" />
      <line x1="90" y1="62" x2="111" y2="62" stroke="#ffffff" stroke-width="1.2" stroke-linecap="round" opacity="0.85" />
      <!-- Penna / Becco posteriore del martello -->
      <path d="M 88 43 L 80 47 L 85 58 L 88 62 Z" fill="#334155" />
    </g>
  </svg>

  <div class="efesto-loader-label">
    <span>{message}</span>
  </div>
</div>
"""
