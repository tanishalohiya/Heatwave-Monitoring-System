"""
Design tokens and shared styling for the Climate Intelligence System.

Usage in app.py:

    from services.theme import CUSTOM_CSS, PLOTLY_TEMPLATE_NAME, STATUS_COLORS, status_badge

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)   # once, near the top

    # in every fig.update_layout(...) call, replace template="plotly_dark" with:
    fig.update_layout(template=PLOTLY_TEMPLATE_NAME, ...)

    # for inline status pills:
    st.markdown(status_badge("Heatwave"), unsafe_allow_html=True)
"""

import plotly.graph_objects as go
import plotly.io as pio


# ------------------------------------------------------------------
# Color tokens
# ------------------------------------------------------------------

COLORS = {
    "bg": "#0B0D12",           # near-black indigo (desert night sky)
    "surface": "#12151C",
    "surface_alt": "#171B24",
    "border": "#232838",
    "text": "#EDEFF4",
    "text_muted": "#8B92A6",
    "accent": "#E8A33D",       # marigold - heat, warnings, primary actions
    "severe": "#D64545",       # deep terracotta - severe heatwave
    "normal": "#4FA37B",       # oasis teal-green - safe/normal
    "info": "#5B8DEF",         # Jodhpur blue - secondary/info accents
}

STATUS_COLORS = {
    "Normal": COLORS["normal"],
    "Heatwave": COLORS["accent"],
    "Severe Heatwave": COLORS["severe"],
}


# ------------------------------------------------------------------
# Shared Plotly template - every chart in the app should use this
# instead of the built-in "plotly_dark" so charts match the CSS theme
# ------------------------------------------------------------------

def register_plotly_template(name="climate_intel"):
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(family="Inter, -apple-system, sans-serif", color=COLORS["text"], size=13),
        title=dict(font=dict(family="Fraunces, Georgia, serif", size=18, color=COLORS["text"])),
        colorway=[COLORS["accent"], COLORS["info"], COLORS["severe"], COLORS["normal"], "#B98AD1", "#6FC7C1"],
        xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"], linecolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"], linecolor=COLORS["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=50, l=10, r=10, b=10),
    )
    pio.templates[name] = template
    return name


PLOTLY_TEMPLATE_NAME = register_plotly_template()


# ------------------------------------------------------------------
# CSS
# ------------------------------------------------------------------

CUSTOM_CSS = f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

:root {{
    --bg: {COLORS['bg']};
    --surface: {COLORS['surface']};
    --surface-alt: {COLORS['surface_alt']};
    --border: {COLORS['border']};
    --text: {COLORS['text']};
    --text-muted: {COLORS['text_muted']};
    --accent: {COLORS['accent']};
    --severe: {COLORS['severe']};
    --normal: {COLORS['normal']};
    --info: {COLORS['info']};
}}

.stApp {{
    background-color: var(--bg);
    color: var(--text);
}}

body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
}}

section[data-testid="stSidebar"] {{
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}}

section[data-testid="stSidebar"] .stRadio label {{
    font-size: 14px;
}}

/* Headline */
.main-title {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600;
    font-size: 44px;
    letter-spacing: -0.01em;
    color: var(--text);
    margin-bottom: 6px;
}}

.subtitle {{
    font-size: 16px;
    color: var(--text-muted);
    margin-bottom: 32px;
}}

.section-title {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600;
    font-size: 22px;
    color: var(--text);
    margin-top: 32px;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}}

/* Metric cards */
div[data-testid="stMetric"] {{
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    padding: 18px 20px;
    border-radius: 8px;
}}

div[data-testid="stMetric"] label {{
    color: var(--text-muted) !important;
    font-size: 13px;
}}

div[data-testid="stMetricValue"] {{
    font-family: 'Inter', sans-serif;
    font-variant-numeric: tabular-nums;
    color: var(--text);
}}

/* Status badges */
.status-badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 500;
}}

.status-badge.normal {{
    background-color: rgba(79, 163, 123, 0.15);
    color: var(--normal);
    border: 1px solid rgba(79, 163, 123, 0.35);
}}

.status-badge.heatwave {{
    background-color: rgba(232, 163, 61, 0.15);
    color: var(--accent);
    border: 1px solid rgba(232, 163, 61, 0.35);
}}

.status-badge.severe {{
    background-color: rgba(214, 69, 69, 0.15);
    color: var(--severe);
    border: 1px solid rgba(214, 69, 69, 0.35);
}}

/* Buttons */
.stButton > button {{
    background-color: var(--accent);
    color: #1a1300;
    border: none;
    border-radius: 6px;
    font-weight: 600;
}}

.stButton > button:hover {{
    background-color: #f0b356;
    color: #1a1300;
}}

/* Dataframes / tables */
[data-testid="stDataFrame"] {{
    border: 1px solid var(--border);
    border-radius: 8px;
}}

</style>
"""


def status_badge(label):
    """
    Return an HTML span for a status pill. label must be one of
    'Normal', 'Heatwave', 'Severe Heatwave'.
    Render with: st.markdown(status_badge("Heatwave"), unsafe_allow_html=True)
    """
    css_class = {
        "Normal": "normal",
        "Heatwave": "heatwave",
        "Severe Heatwave": "severe",
    }.get(label, "normal")

    return f'<span class="status-badge {css_class}">{label}</span>'