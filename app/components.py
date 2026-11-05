"""
components.py - Reusable UI Components, Styling & Professional Icon System

Minimal, modern B2B SaaS design system (inspired by Linear, Stripe, and Notion)
for Resume Job Compatibility Analyzer.
Zero emojis, strict professional typography, clean neutral borders, and deep muted green accents.
"""

from typing import Dict, List, Optional, Any
import streamlit as st


# ==============================================================================
# PROFESSIONAL SVG ICON DICTIONARY (Lucide-compatible, 2px stroke, scalable)
# ==============================================================================
LUCIDE_ICONS = {
    "layout-dashboard": '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>',
    "file-search": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><circle cx="11.5" cy="14.5" r="2.5"/><path d="m16 19-2.2-2.2"/>',
    "chart-no-axes-combined": '<line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "activity": '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.48 12H2"/>',
    "file-text": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "briefcase-business": '<rect width="20" height="14" x="2" y="7" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',
    "upload-cloud": '<path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M12 12v9"/><path d="m16 16-4-4-4 4"/>',
    "settings": '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>',
    "circle-check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "alert-circle": '<circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/>',
    "monitor": '<rect width="20" height="14" x="2" y="3" rx="2"/><line x1="8" x2="16" y1="21" y2="21"/><line x1="12" x2="12" y1="17" y2="21"/>',
    "brain-circuit": '<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/><path d="M17.5 8a2.5 2.5 0 0 0-5 0"/><path d="M6.5 8a2.5 2.5 0 0 1 5 0"/><path d="M12 18v4"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "arrow-right": '<line x1="5" x2="19" y1="12" y2="12"/><polyline points="12 5 19 12 12 19"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "x": '<line x1="18" x2="6" y1="6" y2="18"/><line x1="6" x2="18" y1="6" y2="18"/>',
    "plus": '<line x1="12" x2="12" y1="5" y2="19"/><line x1="5" x2="19" y1="12" y2="12"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "filter": '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "sliders": '<line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="1" x2="7" y1="14" y2="14"/><line x1="9" x2="15" y1="8" y2="8"/><line x1="17" x2="23" y1="16" y2="16"/>'
}


def get_icon(name: str, size: int = 16, color: str = "currentColor", stroke_width: float = 2.0, extra_style: str = "") -> str:
    """Returns an inline SVG HTML snippet for a Lucide icon."""
    path_data = LUCIDE_ICONS.get(name, LUCIDE_ICONS["info"])
    style = f"vertical-align: -0.15em; display: inline-block; {extra_style}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" '
        f'style="{style}">{path_data}</svg>'
    )


# ==============================================================================
# MINIMAL B2B SAAS APPLICATION STYLES (Light, Crisp, Professional)
# ==============================================================================
CUSTOM_CSS = """
<style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: #F8FAFC !important;
        color: #172B25 !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }

    /* Minimal Top Header (Replaces Heavy Gradient Banner) */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .app-header-left {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }
    .app-header-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #245B4A;
        background-color: #EAF3EF;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        width: fit-content;
    }
    .app-header-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #172B25;
        margin: 0.2rem 0 0 0;
        letter-spacing: -0.01em;
    }
    .app-header-subtitle {
        font-size: 0.875rem;
        color: #64748B;
        margin: 0;
    }

    /* Clean Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        height: 100vh !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] .block-container {
        display: flex !important;
        flex-direction: column !important;
        min-height: calc(100vh - 2rem) !important;
        padding: 1.25rem 1rem 1rem 1rem !important;
        background-color: #FFFFFF !important;
    }

    /* Sidebar ML Lab Brand Header */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        padding-bottom: 0.85rem;
        margin-bottom: 1.15rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .sidebar-brand-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background-color: #EAF3EF;
        color: #245B4A;
        flex-shrink: 0;
    }
    .sidebar-brand-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #172B25;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    .sidebar-brand-sub {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 0.1rem;
    }

    /* Sidebar Navigation Radio Styling */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] {
        flex: 1 0 auto !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 0.25rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background-color: transparent !important;
        border-radius: 6px !important;
        padding: 0.55rem 0.75rem !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: background-color 0.15s ease, color 0.15s ease !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
    }
    /* Hide the default radio circle */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
        display: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] p {
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: #172B25 !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        background-color: #F8FAFC !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        background-color: #EAF3EF !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
        color: #245B4A !important;
        font-weight: 600 !important;
    }

    /* Consistent Lucide Icons for Navigation via pure CSS mask */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-of-type(1) p::before {
        content: '';
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 0.65rem;
        vertical-align: -2px;
        background-color: currentColor;
        -webkit-mask: url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22black%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Crect%20width%3D%227%22%20height%3D%229%22%20x%3D%223%22%20y%3D%223%22%20rx%3D%221%22/%3E%3Crect%20width%3D%227%22%20height%3D%225%22%20x%3D%2214%22%20y%3D%223%22%20rx%3D%221%22/%3E%3Crect%20width%3D%227%22%20height%3D%229%22%20x%3D%2214%22%20y%3D%2212%22%20rx%3D%221%22/%3E%3Crect%20width%3D%227%22%20height%3D%225%22%20x%3D%223%22%20y%3D%2216%22%20rx%3D%221%22/%3E%3C/svg%3E") no-repeat center / contain;
        mask: url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22black%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Crect%20width%3D%227%22%20height%3D%229%22%20x%3D%223%22%20y%3D%223%22%20rx%3D%221%22/%3E%3Crect%20width%3D%227%22%20height%3D%225%22%20x%3D%2214%22%20y%3D%223%22%20rx%3D%221%22/%3E%3Crect%20width%3D%227%22%20height%3D%229%22%20x%3D%2214%22%20y%3D%2212%22%20rx%3D%221%22/%3E%3Crect%20width%3D%227%22%20height%3D%225%22%20x%3D%223%22%20y%3D%2216%22%20rx%3D%221%22/%3E%3C/svg%3E") no-repeat center / contain;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-of-type(2) p::before {
        content: '';
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 0.65rem;
        vertical-align: -2px;
        background-color: currentColor;
        -webkit-mask: url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22black%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M14%202H6a2%202%200%200%200-2%202v16a2%202%200%200%200%202%202h12a2%202%200%200%200%202-2V8z%22/%3E%3Cpolyline%20points%3D%2214%202%2014%208%2020%208%22/%3E%3Ccircle%20cx%3D%2211.5%22%20cy%3D%2214.5%22%20r%3D%222.5%22/%3E%3Cpath%20d%3D%22m16%2019-2.2-2.2%22/%3E%3C/svg%3E") no-repeat center / contain;
        mask: url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22black%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M14%202H6a2%202%200%200%200-2%202v16a2%202%200%200%200%202%202h12a2%202%200%200%200%202-2V8z%22/%3E%3Cpolyline%20points%3D%2214%202%2014%208%2020%208%22/%3E%3Ccircle%20cx%3D%2211.5%22%20cy%3D%2214.5%22%20r%3D%222.5%22/%3E%3Cpath%20d%3D%22m16%2019-2.2-2.2%22/%3E%3C/svg%3E") no-repeat center / contain;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-of-type(3) p::before {
        content: '';
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 0.65rem;
        vertical-align: -2px;
        background-color: currentColor;
        -webkit-mask: url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22black%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cline%20x1%3D%2218%22%20x2%3D%2218%22%20y1%3D%2220%22%20y2%3D%2210%22/%3E%3Cline%20x1%3D%2212%22%20x2%3D%2212%22%20y1%3D%2220%22%20y2%3D%224%22/%3E%3Cline%20x1%3D%226%22%20x2%3D%226%22%20y1%3D%2220%22%20y2%3D%2214%22/%3E%3C/svg%3E") no-repeat center / contain;
        mask: url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22black%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cline%20x1%3D%2218%22%20x2%3D%2218%22%20y1%3D%2220%22%20y2%3D%2210%22/%3E%3Cline%20x1%3D%2212%22%20x2%3D%2212%22%20y1%3D%2220%22%20y2%3D%224%22/%3E%3Cline%20x1%3D%226%22%20x2%3D%226%22%20y1%3D%2220%22%20y2%3D%2214%22/%3E%3C/svg%3E") no-repeat center / contain;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-of-type(4) p::before {
        content: '';
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 0.65rem;
        vertical-align: -2px;
        background-color: currentColor;
        -webkit-mask: url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22black%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Ccircle%20cx%3D%2212%22%20cy%3D%2212%22%20r%3D%2210%22/%3E%3Cpath%20d%3D%22M12%2016v-4%22/%3E%3Cpath%20d%3D%22M12%208h.01%22/%3E%3C/svg%3E") no-repeat center / contain;
        mask: url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22black%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Ccircle%20cx%3D%2212%22%20cy%3D%2212%22%20r%3D%2210%22/%3E%3Cpath%20d%3D%22M12%2016v-4%22/%3E%3Cpath%20d%3D%22M12%208h.01%22/%3E%3C/svg%3E") no-repeat center / contain;
    }

    /* Subtle Footer Pinned at Bottom of Sidebar */
    .sidebar-footer {
        margin-top: auto !important;
        padding-top: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-top: 1px solid #E2E8F0 !important;
    }
    .sidebar-footer-content {
        display: inline-flex !important;
        align-items: center !important;
        gap: 0.45rem !important;
        font-size: 0.8rem !important;
        color: #245B4A !important;
        font-weight: 500 !important;
        letter-spacing: -0.01em !important;
    }

    /* Clean Card Container */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.85rem;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .metric-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    .metric-title {
        color: #64748B;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #172B25;
        letter-spacing: -0.02em;
    }
    .metric-subtitle {
        color: #64748B;
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }

    /* Refined Compatibility Badges */
    .badge-strong {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background-color: #EAF3EF;
        color: #245B4A;
        border: 1px solid #D1E5DC;
        padding: 0.4rem 0.9rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .badge-medium {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #E2E8F0;
        padding: 0.4rem 0.9rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .badge-weak {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FEE2E2;
        padding: 0.4rem 0.9rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* Minimal Skill Chips */
    .skill-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.2rem 0.55rem;
        margin: 0.15rem 0.2rem 0.15rem 0;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .chip-matched {
        background-color: #EAF3EF;
        color: #245B4A;
        border: 1px solid #D1E5DC;
    }
    .chip-missing {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FEE2E2;
    }
    .chip-additional {
        background-color: #F8FAFC;
        color: #64748B;
        border: 1px solid #E2E8F0;
    }

    /* Subdued Academic Notice Box */
    .academic-disclaimer {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #245B4A;
        padding: 0.85rem 1.1rem;
        border-radius: 6px;
        margin: 1.25rem 0;
        font-size: 0.82rem;
        color: #475569;
        line-height: 1.45;
    }
    .disclaimer-title {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-weight: 600;
        color: #172B25;
        margin-bottom: 0.25rem;
        font-size: 0.85rem;
    }

    /* Primary Buttons (Deep Muted Green) */
    div.stButton > button[kind="primary"] {
        background-color: #245B4A !important;
        color: #FFFFFF !important;
        border: 1px solid #245B4A !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.25rem !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1E4B3D !important;
        border-color: #1E4B3D !important;
        color: #FFFFFF !important;
    }
    div.stButton > button[kind="primary"]:focus {
        box-shadow: 0 0 0 2px rgba(36, 91, 74, 0.25) !important;
    }

    /* Secondary Buttons */
    div.stButton > button:not([kind="primary"]) {
        background-color: #FFFFFF !important;
        color: #172B25 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
    }
    div.stButton > button:not([kind="primary"]):hover {
        background-color: #F8FAFC !important;
        border-color: #CBD5E1 !important;
    }

    /* Clean Input & Text Area Fields */
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"]:focus-within > div,
    div[data-baseweb="textarea"]:focus-within > div,
    div[data-baseweb="select"]:focus-within > div {
        border-color: #245B4A !important;
        box-shadow: 0 0 0 1px #245B4A !important;
    }

    /* File Uploader Refinement */
    div[data-testid="stFileUploader"] section {
        border: 1px dashed #CBD5E1 !important;
        border-radius: 8px !important;
        background-color: #FFFFFF !important;
        padding: 1rem !important;
    }

    /* Headings hierarchy */
    h1, h2, h3, h4, h5, h6 {
        color: #172B25 !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }
    h3 {
        font-size: 1.15rem !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    h4 {
        font-size: 0.98rem !important;
        margin-top: 0.75rem !important;
        margin-bottom: 0.4rem !important;
    }

    /* Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem !important;
        border-bottom: 1px solid #E2E8F0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #64748B !important;
        padding: 0.5rem 0.85rem !important;
        border-radius: 6px 6px 0 0 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #245B4A !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #245B4A !important;
        background-color: transparent !important;
    }

    /* Section divider */
    hr {
        margin: 1.25rem 0 !important;
        border-color: #E2E8F0 !important;
    }

    /* Workflow Step Cards */
    .workflow-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        height: 100%;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .workflow-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    }
    .workflow-card-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.45rem;
    }
    .workflow-step-badge {
        font-size: 0.68rem;
        font-weight: 700;
        color: #245B4A;
        background-color: #EAF3EF;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .workflow-card-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #172B25;
        margin-bottom: 0.25rem;
    }
    .workflow-card-desc {
        font-size: 0.8rem;
        color: #64748B;
        line-height: 1.45;
        margin: 0;
    }

    /* Primary Model Highlight Tag */
    .primary-model-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background-color: #EAF3EF;
        color: #245B4A;
        border: 1px solid #D1E5DC;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
    }
</style>
"""


def apply_custom_styles():
    """Injects minimal B2B SaaS styles into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(
    subtitle: str = "Classical Machine Learning Pipeline &bull; Candidate-Grouped Validation",
    title: str = "Resume Job Compatibility Analyzer",
    tag: str = "ML Lab Evaluation System",
    icon_name: str = "activity"
):
    """
    Renders a clean, minimal header without heavy gradients.
    """
    tag_icon = get_icon(icon_name, size=13, color="#245B4A")
    st.markdown(f"""
    <div class="app-header">
        <div class="app-header-left">
            <div class="app-header-tag">{tag_icon} {tag}</div>
            <h1 class="app-header-title">{title}</h1>
            <p class="app-header-subtitle">{subtitle}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_workflow_step(step_num: int, title: str, description: str, icon_name: str = "layers"):
    """Renders a single step card in the workflow pipeline."""
    icon_svg = get_icon(icon_name, size=15, color="#245B4A")
    st.markdown(f"""
    <div class="workflow-card">
        <div class="workflow-card-top">
            <span class="workflow-step-badge">Step {step_num:02d}</span>
            {icon_svg}
        </div>
        <div class="workflow-card-title">{title}</div>
        <div class="workflow-card-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def render_compatibility_badge(label: str) -> str:
    """Returns HTML markup for a styled compatibility category badge using clean SVG icons."""
    label_clean = str(label).strip().lower()
    if label_clean == "strong_positive":
        icon_svg = get_icon("circle-check", size=15, color="#245B4A")
        return f'<span class="badge-strong">{icon_svg} Strong Positive Match</span>'
    elif label_clean == "medium_positive":
        icon_svg = get_icon("info", size=15, color="#334155")
        return f'<span class="badge-medium">{icon_svg} Medium Positive Match</span>'
    elif label_clean == "weak_positive":
        icon_svg = get_icon("alert-circle", size=15, color="#991B1B")
        return f'<span class="badge-weak">{icon_svg} Weak Positive Match</span>'
    
    icon_svg = get_icon("info", size=15, color="#334155")
    return f'<span class="badge-medium">{icon_svg} {label}</span>'


def render_metric_card(title: str, value: str, subtitle: str = "", icon_name: Optional[str] = None):
    """Renders a structured, clean KPI card."""
    icon_html = f"{get_icon(icon_name, size=14, color='#64748B')} " if icon_name else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{icon_html}{title}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_skill_chips(skills: List[str], chip_type: str = "matched", max_display: int = 40):
    """
    Renders styled skill chips.
    chip_type: 'matched', 'missing', or 'additional'
    """
    if not skills:
        st.markdown("<span style='color: #94a3b8; font-style: italic; font-size: 0.8rem;'>None detected</span>", unsafe_allow_html=True)
        return

    icon_map = {
        "matched": get_icon("check", size=12, color="#245B4A"),
        "missing": get_icon("x", size=12, color="#991B1B"),
        "additional": get_icon("plus", size=12, color="#64748B"),
    }
    icon_svg = icon_map.get(chip_type, "")

    css_class = f"skill-chip chip-{chip_type}"
    chips_html = []
    
    for s in skills[:max_display]:
        chips_html.append(f'<span class="{css_class}">{icon_svg} {s}</span>')

    if len(skills) > max_display:
        chips_html.append(f'<span class="skill-chip chip-additional">+{len(skills) - max_display} more</span>')

    st.markdown("".join(chips_html), unsafe_allow_html=True)


def render_academic_disclaimer():
    """Renders the academic disclaimer regarding dataset approximations with a clean icon."""
    info_icon = get_icon("info", size=15, color="#245B4A")
    st.markdown(f"""
    <div class="academic-disclaimer">
        <div class="disclaimer-title">{info_icon} Academic Evaluation Notice</div>
        Model predictions reflect algorithmic approximations of the underlying Hugging Face dataset's scoring rubric 
        (binned from synthetic compatibility formulas). They serve strictly for academic ML lab evaluation and do not 
        substitute for real-world hiring decisions or human judgment. Probability values are displayed only for models 
        with calibrated probabilistic outputs (e.g., KNN, Random Forest, Logistic Regression); non-probabilistic models 
        (e.g., Linear SVM) display direct geometric hyperplane classifications.
    </div>
    """, unsafe_allow_html=True)
