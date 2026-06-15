"""
Streamlit UI for the Agentic RAG Financial Chatbot.
Premium ChatGPT/Codex-style design — no-scroll layout, Inter font, full viewport.


"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, HumanMessage

from backend import DB_URI, MODEL_NAME, build_chatbot


st.set_page_config(
    page_title="Varsity Finance AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --t-xs: 12px;
        --t-sm: 14px;
        --t-md: 16px;
        --t-lg: 24px;

        --bg:          #0d0d0d;
        --sidebar-bg:  #171717;
        --panel:       #1e1e1e;
        --panel-alt:   #262626;
        --ink:         #ececec;
        --ink-muted:   #8e8e8e;
        --ink-faint:   #555555;
        --border:      #2a2a2a;
        --border-mid:  #333333;
        --accent:      #10a37f;
        --accent-dim:  #0d7a60;
        --accent-glow: rgba(16,163,127,.12);
        --user-bubble: #1a1a1a;
        --ai-bubble:   #111111;

        /* ── Sidebar width is a CSS variable so JS can update it via
           setProperty('--sidebar-width', ...) and override !important rules ── */
        --sidebar-width: 284px;
        --radius:    8px;
        --radius-lg: 12px;
    }

    *, *::before, *::after { box-sizing: border-box; }

    html, body {
        height: auto !important;
        overflow-x: hidden;
        overflow-y: auto !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: var(--t-md);
        line-height: 1.6;
        color: var(--ink);
        background: var(--bg);
        -webkit-font-smoothing: antialiased;
    }

    #MainMenu, footer,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        display: none !important;
    }

    header,
    [data-testid="stHeader"] {
        display: block !important;
        height: 0 !important;
        min-height: 0 !important;
        background: transparent !important;
        overflow: visible !important;
        pointer-events: none !important;
        z-index: 999998 !important;
    }

    header button,
    header [role="button"],
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] [role="button"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"] {
        pointer-events: auto !important;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"] {
        height:  auto !important;
        min-height: 100vh !important;
        overflow: visible !important;
        background: var(--bg) !important;
    }

    /* ── Main content: flex column for header+messages+input layout.
       padding-top: 2px prevents the header text from being clipped
       at the top edge when overflow:hidden is active. ── */
    .main .block-container,
    [data-testid="stMain"] > div,
    [data-testid="stMain"] {
        height: auto !important;
        min-height: 100vh !important;
        overflow-y: auto !important;
        padding: 0 !important;
        padding-top: 2px !important;
        # display: flex !important;
        flex-direction: column !important;
    }

    /* ── Sidebar width driven by CSS variable — JS updates --sidebar-width
       to resize; using var() inside !important lets JS win via setProperty ── */
    section[data-testid="stSidebar"] {
        width:     var(--sidebar-width) !important;
        min-width: var(--sidebar-width) !important;
        max-width: var(--sidebar-width) !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
        background: #111111 !important;
        position: relative !important;
        transition: width 0.05s ease !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        width:     var(--sidebar-width) !important;
        min-width: var(--sidebar-width) !important;
        max-width: var(--sidebar-width) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 1rem 0.875rem 1rem 0.875rem !important;
    }

    section[data-testid="stSidebar"] .element-container,
    section[data-testid="stSidebar"] .stButton,
    section[data-testid="stSidebar"] .stTextInput,
    section[data-testid="stSidebar"] [data-testid="stTextInputRootElement"] {
        width: 100% !important;
        max-width: 100% !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        border-radius: 14px !important;
    }

    /* ── stSidebarContent: overflow-y auto enables scrolling.
       The old overflow:hidden was silently blocking all scroll. ── */
    [data-testid="stSidebarContent"] {
        height: 100% !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        display: flex !important;
        flex-direction: column !important;
        padding: 14px 12px !important;
        gap: 0;
        scrollbar-width: thin;
        scrollbar-color: #333 transparent;
    }
    [data-testid="stSidebarContent"]::-webkit-scrollbar { width: 4px; }
    [data-testid="stSidebarContent"]::-webkit-scrollbar-track { background: transparent; }
    [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 4px;
    }

    [data-testid="stSidebar"] * {
        font-family: 'Inter', sans-serif !important;
        color: var(--ink);
    }

    /* When sidebar is OPEN */
    body.sidebar-expanded 
    [data-testid="stAppViewContainer"] 
    [data-testid="stMain"] 
    .block-container {
        max-width: 1120px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* When sidebar is CLOSED → true center */
    body.sidebar-collapsed 
    [data-testid="stAppViewContainer"] 
    [data-testid="stMain"] 
    .block-container {
        max-width: 900px !important;   /* slightly tighter like ChatGPT */
        margin-left: auto !important;
        margin-right: auto !important;
    }

    body.sidebar-collapsed div[data-testid="stChatMessage"] {
        max-width: 900px;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    .home-shell,
    .chat-shell,
    .hero-shell,
    .suggestions-shell,
    .composer-shell {
        width: 100%;
        max-width: 1120px;
        margin-left: auto;
        margin-right: auto;
    }

    /* ── FIX 1: Hide native Streamlit collapse button completely ──
       The native button shows "keyboard_double_arrow_left" as raw text
       when the Material Symbols font fails to load. We use our own
       custom JS toggle button instead, so hide the native one entirely. */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] button {
        display: none !important;
        visibility: hidden !important;
    }

    /* ── SIDEBAR TOGGLE BUTTON ── */
    [data-testid="baseButton-header"] {
        background: transparent !important;
        border:        none            !important;
        border-radius: var(--radius)   !important;
        padding:       7px 10px        !important;
        color: #b3b3b3 !important;
        cursor:        pointer         !important;
        display:       flex            !important;
        align-items:   center          !important;
        justify-content: center        !important;
        visibility:    visible         !important;
        opacity:       1               !important;
        z-index:       999999          !important;
        box-shadow: none !important;
        transition:    background 0.15s !important;
    }
    [data-testid="baseButton-header"]:hover {
        background: rgba(255,255,255,0.08) !important;
        color: #ffffff !important;
    }
    [data-testid="baseButton-header"] svg,
    [data-testid="baseButton-header"] svg *,
    [data-testid="baseButton-header"] path,
    [data-testid="baseButton-header"] polyline,
    [data-testid="baseButton-header"] line {
        fill:   #ffffff !important;
        stroke: #ffffff !important;
        color:  #ffffff !important;
    }

    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"] {
        display:       flex      !important;
        visibility:    visible   !important;
        opacity:       1         !important;
        z-index:       999999    !important;
        background:    transparent !important;
        position:      fixed     !important;
        top:           12px      !important;
        left:          12px      !important;
        width:         auto      !important;
        height:        auto      !important;
    }

    [data-testid="stExpandSidebarButton"] {
        align-items:     center     !important;
        justify-content: center     !important;
        width:           40px       !important;
        height:          40px       !important;
        padding:         0          !important;
        border:          none       !important;
        border-radius:   var(--radius) !important;
        background: transparent !important;
        color: #b3b3b3 !important;
        box-shadow: none !important;
        cursor:          pointer    !important;
    }
    [data-testid="stExpandSidebarButton"]:hover {
        background: rgba(255,255,255,0.08) !important;
        color: #ffffff !important;
    }
    [data-testid="stExpandSidebarButton"] *,
    [data-testid="stExpandSidebarButton"] span {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }

    button[aria-label="Open sidebar"],
    button[aria-label="Expand sidebar"] {
        background:    var(--accent) !important;
        border:        none          !important;
        border-radius: var(--radius) !important;
        color:         #ffffff       !important;
        z-index:       999999        !important;
    }
    button[aria-label="Open sidebar"] svg *,
    button[aria-label="Expand sidebar"] svg * {
        fill: #ffffff !important; stroke: #ffffff !important;
    }

    /* ── Sidebar: New Chat button ── */
    [data-testid="stSidebar"] .stButton:first-of-type > button {
        background: var(--panel) !important;
        border: 1px solid var(--border-mid) !important;
        color: var(--ink) !important;
        font-size: var(--t-sm) !important;
        font-weight: 500 !important;
        height: 36px !important;
        border-radius: var(--radius) !important;
        width: 100% !important;
        cursor: pointer;
        transition: background 0.15s, border-color 0.15s;
        padding: 0 12px !important;
        letter-spacing: 0;
    }
    [data-testid="stSidebar"] .stButton:first-of-type > button:hover {
        background: var(--panel-alt) !important;
        border-color: var(--border-mid) !important;
    }

    .sidebar-label {
        font-size: var(--t-xs);
        font-weight: 600;
        color: var(--ink-faint);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 10px 4px 4px 4px;
        margin: 0;
    }

    /* ══════════════════════════════════════════════════════════════════════
       SIDEBAR THREAD BUTTONS — ChatGPT-style
       Streamlit attaches data-testid="baseButton-primary" (active thread)
       and data-testid="baseButton-secondary" (inactive thread) to the button
       element itself. The old button[kind="primary"] selector was wrong and
       never matched anything, causing the grey Streamlit default to bleed through.
       We target both data-testid variants with high specificity here.
       ══════════════════════════════════════════════════════════════════════ */

    /* Base reset — every sidebar thread button */
    [data-testid="stSidebar"] [data-testid="baseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="baseButton-primary"] {
        background:       transparent !important;
        border:           none        !important;
        outline:          none        !important;
        box-shadow:       none        !important;
        border-radius:    6px         !important;
        color:            var(--ink-muted) !important;
        font-size:        var(--t-sm) !important;
        font-weight:      400         !important;
        height:           34px        !important;
        width:            100%        !important;
        text-align:       left        !important;
        justify-content:  flex-start  !important;
        padding:          0 10px 0 12px !important;
        white-space:      nowrap      !important;
        overflow:         hidden      !important;
        text-overflow:    ellipsis    !important;
        transition:       background 0.12s, color 0.12s !important;
        position:         relative    !important;
    }

    /* Hover — both states */
    [data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover,
    [data-testid="stSidebar"] [data-testid="baseButton-primary"]:hover {
        background: rgba(255,255,255,0.06) !important;
        color:      var(--ink) !important;
        border:     none !important;
        box-shadow: none !important;
    }

    /* Active thread — green left accent + slightly lit background + white text */
    [data-testid="stSidebar"] [data-testid="baseButton-primary"] {
        background:    rgba(255,255,255,0.08) !important;
        color:         var(--ink)            !important;
        font-weight:   500                   !important;
        border-left:   3px solid var(--accent) !important;
        padding-left:  10px                  !important;
        border-radius: 0 6px 6px 0           !important;
    }
    [data-testid="stSidebar"] [data-testid="baseButton-primary"]:hover {
        background: rgba(255,255,255,0.11) !important;
    }

    /* ── FIX 2: Scrollable thread list — fixed max-height so it scrolls ── */
    .thread-scroll-area {
        flex: 1 1 auto;
        overflow-y: auto;
        overflow-x: hidden;
        min-height: 0;
        max-height: 340px;
        padding-right: 2px;
        margin: 0 -2px;
    }
    .thread-scroll-area::-webkit-scrollbar { width: 4px; }
    .thread-scroll-area::-webkit-scrollbar-track { background: transparent; }
    .thread-scroll-area::-webkit-scrollbar-thumb { background: var(--border-mid); border-radius: 4px; }

    .status-row {
        display: flex;
        align-items: center;
        gap: 7px;
        font-size: var(--t-xs);
        color: var(--ink-muted);
        padding: 3px 0;
    }
    .dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .dot-green  { background: #3fb950; }
    .dot-red    { background: #f85149; }
    .dot-amber  { background: #d29922; }

    .id-pill {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 5px 9px;
        font-size: var(--t-xs);
        color: var(--ink-muted);
        word-break: break-all;
        font-family: 'SF Mono', 'Fira Code', monospace !important;
        line-height: 1.4;
        margin-top: 3px;
    }

    [data-testid="stSidebar"] hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 8px 0;
    }

    [data-testid="stSidebar"] input {
        background: var(--panel) !important;
        border: 1px solid var(--border-mid) !important;
        color: var(--ink) !important;
        font-size: var(--t-sm) !important;
        border-radius: 6px !important;
        height: 32px !important;
        padding: 0 10px !important;
    }
    [data-testid="stSidebar"] label {
        font-size: var(--t-xs) !important;
        color: var(--ink-faint) !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stSidebar"] .stTextInput {
        margin-bottom: 4px;
    }

    .session-date {
        font-size: var(--t-xs);
        color: var(--ink-faint);
        padding: 6px 4px 0 4px;
        border-top: 1px solid var(--border);
        margin-top: 4px;
    }

    [data-testid="stSidebar"] .stButton:last-of-type > button {
        background: transparent !important;
        border: 1px solid var(--border-mid) !important;
        color: var(--ink-muted) !important;
        font-size: var(--t-xs) !important;
        font-weight: 500 !important;
        height: 30px !important;
        border-radius: 6px !important;
    }
    [data-testid="stSidebar"] .stButton:last-of-type > button:hover {
        border-color: var(--accent) !important;
        color: var(--ink) !important;
    }

    /* Remove top space from main container */
    .block-container {
        padding-top: 0px !important;
    }

    /* Also remove extra top spacing from main */
    [data-testid="stMain"] {
        padding-top: 0px !important;
    }

    /* Remove extra bottom space in chat area */
    .chat-scroll-zone {
        padding-bottom: 0px !important;
        margin-bottom: 0px !important;
    }
    /* Pull input bar closer to content */
    .input-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
    }   

    /* Remove space created by last element */
    .chat-scroll-zone > *:last-child {
        margin-bottom: 0px !important;
    }
    .chat-scroll-zone {
    padding-bottom: 90px !important;  /* 🔥 match input height */
}
    }
    .main .block-container {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;   /* 🔥 THIS changes everything */
    }
    .main-header {
        flex: 0 0 auto;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 0px 24px 6px 24px;
        border-bottom: 1px solid var(--border);
        margin-top: -10px;
        background: var(--bg);
        transform: translateY(-6px);
    }
    .main-title {
        font-size: var(--t-lg);
        font-weight: 700;
        color: var(--ink);
        letter-spacing: -0.3px;
        margin: 0;
        margin-top: 0px !important;
        margin-bottom: 4px !important;
        line-height: 1.2;
    }
    .main-sub {
        font-size: var(--t-sm);
        color: var(--ink-muted);
        margin: 4px 0 0 0;
        font-weight: 400;
    }

    .error-box {
        flex: 0 0 auto;
        margin: 8px 20px;
        border: 1px solid #5f2b2b;
        background: #1a0d0d;
        color: #ffd7d7;
        border-radius: var(--radius);
        padding: 10px 14px;
        font-size: var(--t-sm);
        line-height: 1.5;
    }

    .chat-scroll-zone {
        flex: 1 1 auto;
        min-height: 0;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 16px 0 0 0;
    }
    .chat-scroll-zone::-webkit-scrollbar { width: 5px; }
    .chat-scroll-zone::-webkit-scrollbar-track { background: transparent; }
    .chat-scroll-zone::-webkit-scrollbar-thumb { background: var(--border-mid); border-radius: 4px; }

    div[data-testid="stChatMessage"] {
        border-radius: 0 !important;
        border: none !important;
        padding: 16px 24px !important;
        margin: 0 !important;
        max-width: 100%;
        font-size: var(--t-md) !important;
        line-height: 1.7 !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: var(--user-bubble) !important;
        border-bottom: 1px solid var(--border) !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: var(--ai-bubble) !important;
        border-bottom: 1px solid var(--border) !important;
    }
    div[data-testid="stChatMessage"] p {
        font-size: var(--t-md) !important;
        line-height: 1.7 !important;
        color: var(--ink) !important;
        margin: 0;
    }
    div[data-testid="stChatMessage"] code {
        font-size: 14px !important;
        background: var(--panel-alt);
        color: #a5d6ff;
        padding: 1px 5px;
        border-radius: 4px;
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    }
    div[data-testid="stChatMessage"] pre code {
        font-size: 14px !important;
        background: transparent;
        padding: 0;
    }
    div[data-testid="stChatMessage"] pre {
        background: var(--panel-alt);
        border: 1px solid var(--border-mid);
        border-radius: var(--radius);
        padding: 12px 14px;
        overflow-x: auto;
        margin: 10px 0;
    }

    .empty-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: calc(100vh - 140px);
        padding: 0 20px 40px 20px;
        text-align: center;
    }
    .empty-icon {
        font-size: 40px;
        margin-bottom: 12px;
        line-height: 1;
    }
    .empty-title {
        font-size: 20px;
        font-weight: 600;
        color: var(--ink);
        margin: 0 0 8px 0;
    }
    .empty-sub {
        font-size: var(--t-sm);
        color: var(--ink-muted);
        margin: 0 0 24px 0;
        max-width: 380px;
        line-height: 1.6;
    }
    .suggestion-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        max-width: 580px;
        width: 100%;
    }
    .suggestion-card {
        background: var(--panel);
        border: 1px solid var(--border-mid);
        border-radius: var(--radius);
        padding: 13px 16px;
        font-size: var(--t-sm);
        color: var(--ink-muted);
        cursor: pointer;
        transition: border-color 0.15s, background 0.15s, color 0.15s;
        text-align: left;
        line-height: 1.5;
    }
    .suggestion-card:hover {
        border-color: var(--accent);
        background: var(--accent-glow);
        color: var(--ink);
    }
    .suggestion-card .s-icon {
        display: block;
        font-size: 17px;
        margin-bottom: 5px;
    }
    .suggestion-card .s-text {
        font-weight: 500;
        color: var(--ink);
        font-size: var(--t-sm);
    }
    .suggestion-card .s-sub {
        font-size: var(--t-xs);
        color: var(--ink-faint);
        margin-top: 3px;
    }

    .input-bar {
        flex: 0 0 auto;
        border-top: 1px solid var(--border);
        background: #1e1e1e;
        padding: 10px 20px 12px 20px;
    }
    [data-testid="stChatInput"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    [data-testid="stChatInput"] > div {
        background: #1e1e1e;
        border: 1px solid var(--border-mid) !important;
        border-radius: var(--radius-lg) !important;
        padding: 0 !important;
        transition: border-color 0.15s;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: var(--ink) !important;
        font-size: var(--t-md) !important;
        font-family: 'Inter', sans-serif !important;
        border: none !important;
        resize: none;
        padding: 11px 16px !important;
        line-height: 1.5;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--ink-faint) !important;
        font-size: var(--t-sm) !important;
    }

    .thinking-dot {
        display: inline-block;
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--accent);
        animation: pulse 1.2s ease-in-out infinite;
        margin: 0 2px;
    }
    .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes pulse {
        0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
        40%            { opacity: 1;   transform: scale(1);   }
    }

    [data-testid="stSpinner"] {
        display: none !important;
    }

    .stMarkdown, .stMarkdown p, .stMarkdown li {
        color: var(--ink);
        font-size: var(--t-md);
    }
    [data-testid="stMarkdownContainer"] p { color: var(--ink); }

    .stCaptionContainer, [data-testid="stCaptionContainer"] {
        color: var(--ink-faint) !important;
        font-size: var(--t-xs) !important;
    }

    body.sidebar-expanded [data-testid="stAppViewContainer"] [data-testid="stMain"] .block-container {
        max-width: 1120px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    body.sidebar-collapsed [data-testid="stAppViewContainer"] [data-testid="stMain"] .block-container {
        max-width: 1240px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    .suggestions-shell,
    .cards-shell {
        max-width: 900px;
        margin: 0 auto;
    }

    .composer-shell,
    .chat-input-shell {
        max-width: 1120px;
        margin: 0 auto;
        width: 100%;
    }

    .composer-shell .stTextInput,
    .composer-shell [data-testid="stTextInputRootElement"] {
        width: 100% !important;
    }

    /* FIX 3: Resize handle styling */
    #vf-resize-handle {
        position: absolute !important;
        right: -3px !important;
        top: 0 !important;
        width: 6px !important;
        height: 100% !important;
        cursor: ew-resize !important;
        z-index: 9999 !important;
        background: transparent !important;
        transition: background 0.15s !important;
    }
    #vf-resize-handle:hover {
        background: rgba(16,163,127,0.35) !important;
    }
    #vf-resize-handle.dragging {
        background: rgba(16,163,127,0.5) !important;
    }

    * { scrollbar-width: thin; scrollbar-color: var(--border-mid) transparent; }

    body.sidebar-collapsed div[data-testid="stChatMessage"] {
        max-width: 900px;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    body.sidebar-collapsed [data-testid="stChatInput"] {
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }

    .block-container,
        [data-testid="stChatInput"],
        div[data-testid="stChatMessage"] {
            transition: all 0.2s ease;
        }

    body.sidebar-collapsed .chat-scroll-zone {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    body.sidebar-collapsed .chat-scroll-zone > * {
        width: 100%;
        max-width: 900px;
    }

    body.sidebar-collapsed .main-header {
        align-items: center !important;
        text-align: center;
    }

    body.sidebar-collapsed .input-bar {
        display: flex;
        justify-content: center;
    }

    body.sidebar-collapsed .input-bar > * {
        width: 100%;
        max-width: 900px;
    }

    body.sidebar-collapsed section[data-testid="stSidebar"] {
        width: 0px !important;
        min-width: 0px !important;
        max-width: 0px !important;
    }
    body.sidebar-collapsed section[data-testid="stSidebar"] > div {
        display: none !important;
    }
    body.sidebar-collapsed [data-testid="stAppViewContainer"] {
        margin-left: 0 !important;
        padding-left: 0 !important;
    }

    /* ===== FINAL OVERRIDE FIX (DO NOT MODIFY ABOVE CSS) ===== */

    html, body {
        overflow-y: auto !important;
        height: auto !important;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stMain"],
    .main .block-container {
        height: auto !important;
        min-height: 100vh !important;
        overflow: visible !important;
    }

    /* Prevent top clipping */
    .main-header {
        margin-top: 20px !important;
    }

    /* Ensure chat area never clips */
    .chat-scroll-zone {
        overflow-y: auto !important;
        min-height: 0 !important;
    }
    /* ── Force dark on Streamlit's bottom bar (light mode fix) ── */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottom"] > div > div {
        background-color: #0d0d0d !important;
        border-color: #2a2a2a !important;
    }

    /* ── Force dark on chat input inner wrapper ── */
    [data-testid="stChatInput"] > div {
        background: #1e1e1e !important;
        border: 1px solid #333333 !important;
    }

    /* ── White text + caret in textarea ── */
    [data-testid="stChatInput"] textarea {
        color: #ececec !important;
        caret-color: #ececec !important;
        -webkit-text-fill-color: #ececec !important;
    }

    /* ── Send button dark styling ── */
    [data-testid="stChatInput"] button {
        background: #262626 !important;
        border-color: #333333 !important;
    }
    [data-testid="stChatInput"] button svg,
    [data-testid="stChatInput"] button svg * {
        fill: #ececec !important;
        stroke: #ececec !important;
    }

    /* ── App-wide dark background override (beats light theme) ── */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stAppViewBlockContainer"] {
        background-color: #0d0d0d !important;
    }
    /* FORCE dark input container (strong override) */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div {
        background-color: #1e1e1e !important;
        color: #ececec !important;
    }

    /* FORCE textarea */
    [data-testid="stChatInput"] textarea {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        caret-color: #ffffff !important;
    }

    /* Placeholder fix */
    [data-testid="stChatInput"] textarea::placeholder {
        color: #888 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

components.html(
    """
    <script>
    (() => {
        const STORAGE_KEY = 'vf_user_id';
        const url = new URL(window.parent.location.href);
        const inUrl = url.searchParams.get('user_id');
        const inStorage = localStorage.getItem(STORAGE_KEY);

        if (inUrl) {
            localStorage.setItem(STORAGE_KEY, inUrl);
        } else if (inStorage) {
            url.searchParams.set('user_id', inStorage);
            window.parent.location.replace(url.toString());
        }

        const doc = window.parent.document;
        // ── Custom sidebar toggle button ──────────────────────────────────────
        if (window.parent.__vfSidebarInterval) {
            window.parent.clearInterval(window.parent.__vfSidebarInterval);
        }
        const existing = doc.getElementById("vf-sidebar-toggle");
        if (existing) existing.remove();

        const button = doc.createElement("button");
        button.id = "vf-sidebar-toggle";
        button.type = "button";
        button.title = "Toggle sidebar";
        button.setAttribute("aria-label", "Toggle sidebar");
        button.innerHTML = "&#9776;";
        Object.assign(button.style, {
            position:       "fixed",
            top:            "12px",
            left:           "12px",
            width:          "40px",
            height:         "40px",
            border:         "0",
            borderRadius:   "8px",
            background: "transparent",
            color: "#b3b3b3",
            fontSize:       "22px",
            fontWeight: "500",
            lineHeight:     "40px",
            textAlign:      "center",
            cursor:         "pointer",
            zIndex:         "2147483647",
            boxShadow: "none",
        });
        button.onmouseenter = () => {
            button.style.background = "rgba(255,255,255,0.08)";
            button.style.color = "#ffffff";
        };

        button.onmouseleave = () => {
            button.style.background = "transparent";
            button.style.color = "#b3b3b3";
};

        const isSidebarOpen = () => {
            const sidebar = doc.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) return false;
            const rect = sidebar.getBoundingClientRect();
            return rect.width > 40 && rect.right > 40;
        };

        const clickNativeToggle = () => {
            const expand  = doc.querySelector('[data-testid="stExpandSidebarButton"]');
            const collapse = doc.querySelector('[data-testid="stSidebarCollapseButton"] button')
                          || doc.querySelector('[data-testid="stBaseButton-headerNoPadding"]');
            const target = isSidebarOpen() ? collapse : expand;
            if (target) target.click();
        };

        const refreshIcon = () => {
            const nextIcon = isSidebarOpen() ? "&#8249;" : "&#9776;";
            if (button.innerHTML !== nextIcon) button.innerHTML = nextIcon;
        };

        button.addEventListener("click", () => {
            clickNativeToggle();
            window.setTimeout(refreshIcon, 250);
        });

        doc.body.appendChild(button);
        refreshIcon();
        window.parent.__vfSidebarInterval = window.parent.setInterval(refreshIcon, 500);

        // ── Sidebar state class sync ──────────────────────────────────────────
        function syncSidebarState() {
            const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
            if (!sidebar) return;
            const expanded = sidebar.getAttribute('aria-expanded') === 'true';
            doc.body.classList.toggle('sidebar-expanded', expanded);
            doc.body.classList.toggle('sidebar-collapsed', !expanded);
        }
        syncSidebarState();
        const observer = new MutationObserver(syncSidebarState);
        observer.observe(doc.body, { subtree: true, attributes: true, attributeFilter: ['aria-expanded'] });

        // ── Reinforce sidebar scroll after every Streamlit re-render ──────────
        // Streamlit may reset overflow on its containers after each rerun;
        // this interval keeps the sidebar scrollable at all times.
        function enforceSidebarScroll() {
            const sc = doc.querySelector('[data-testid="stSidebarContent"]');
            if (sc && sc.style.overflowY !== 'auto') {
                sc.style.overflowY = 'auto';
                sc.style.overflowX = 'hidden';
            }
        }
        enforceSidebarScroll();
        window.setInterval(enforceSidebarScroll, 800);

        // ── FIX 3: Drag-to-resize sidebar ────────────────────────────────────
        // Hover the right edge of the sidebar to see the green resize indicator,
        // then drag left/right. Min width: 180px, Max width: 520px.
        (function attachResizeHandle() {
            const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
            if (!sidebar) {
                // Sidebar not mounted yet — retry in 400 ms
                window.setTimeout(attachResizeHandle, 400);
                return;
            }

            if (doc.getElementById('vf-resize-handle')) return; // already attached

            sidebar.style.position = 'relative';

            const handle = doc.createElement('div');
            handle.id = 'vf-resize-handle';
            Object.assign(handle.style, {
                position:   'absolute',
                right:      '-3px',
                top:        '0',
                width:      '6px',
                height:     '100%',
                cursor:     'ew-resize',
                zIndex:     '9999',
                background: 'transparent',
                transition: 'background 0.15s',
            });

            // Visual feedback on hover
            handle.addEventListener('mouseenter', () => {
                if (!isResizing) handle.style.background = 'rgba(16,163,127,0.35)';
            });
            handle.addEventListener('mouseleave', () => {
                if (!isResizing) handle.style.background = 'transparent';
            });

            let isResizing = false;
            let startX     = 0;
            let startWidth = 0;

            handle.addEventListener('mousedown', (e) => {
                isResizing = true;
                startX     = e.clientX;
                startWidth = sidebar.getBoundingClientRect().width;
                handle.style.background        = 'rgba(16,163,127,0.55)';
                doc.body.style.userSelect      = 'none';
                doc.body.style.cursor          = 'ew-resize';
                e.preventDefault();
            });

            doc.addEventListener('mousemove', (e) => {
                if (!isResizing) return;
                const delta = e.clientX - startX;
                const newW  = Math.min(Math.max(startWidth + delta, 180), 520);
                // ── KEY FIX: update the CSS variable, NOT inline styles.
                // Inline styles lose to CSS !important; CSS variables inside
                // !important rules are resolved first, so setProperty wins. ──
                doc.documentElement.style.setProperty('--sidebar-width', newW + 'px');
            });

            doc.addEventListener('mouseup', () => {
                if (!isResizing) return;
                isResizing                     = false;
                handle.style.background        = 'transparent';
                doc.body.style.userSelect      = '';
                doc.body.style.cursor          = '';
            });

            sidebar.appendChild(handle);
        })();

    })();
    </script>
    """,
    height=0,
    width=0,
)


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _query_value(name: str) -> str | None:
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value

def _new_user_id() -> str:
    return f"user_{uuid.uuid4().hex[:8]}"

def _new_thread_id() -> str:
    return f"thread_{uuid.uuid4().hex}"

def _clean_id(value: str | None, fallback: str) -> str:
    value = (value or "").strip()
    return value if value else fallback

# def init_ids() -> None:
#     if "user_id" not in st.session_state:
#         st.session_state.user_id = _clean_id(_query_value("user_id"), _new_user_id())
#     if "thread_id" not in st.session_state:
#         st.session_state.thread_id = _clean_id(_query_value("thread_id"), _new_thread_id())
#     sync_query_params()

def init_ids() -> None:
    if "user_id" not in st.session_state:
        from_url = _clean_id(_query_value("user_id"), "")
        if from_url:
            # Returning user — restore from URL
            st.session_state.user_id = from_url
            st.session_state.setup_done = True
        else:
            # Brand new visitor — needs to enter name
            st.session_state.user_id = ""
            st.session_state.setup_done = False

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = _clean_id(
            _query_value("thread_id"), _new_thread_id()
        )

    if st.session_state.get("setup_done", False):
        sync_query_params()

def sync_query_params() -> None:
    st.query_params["user_id"]   = st.session_state.user_id
    st.query_params["thread_id"] = st.session_state.thread_id

def config_for(thread_id: str | None = None, user_id: str | None = None) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": thread_id or st.session_state.thread_id,
            "user_id":   user_id   or st.session_state.user_id,
        }
    }

def uri_with_timeout(uri: str, seconds: int = 5) -> str:
    if "connect_timeout=" in uri:
        return uri
    separator = "&" if "?" in uri else "?"
    return f"{uri}{separator}connect_timeout={seconds}"

@st.cache_resource(show_spinner=False)
def get_runtime() -> dict[str, Any]:
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.store.postgres import PostgresStore

    probe = psycopg.connect(uri_with_timeout(DB_URI))
    probe.close()

    checkpointer_cm = PostgresSaver.from_conn_string(DB_URI)
    store_cm        = PostgresStore.from_conn_string(DB_URI)
    checkpointer    = checkpointer_cm.__enter__()
    store           = store_cm.__enter__()

    checkpointer.setup()
    store.setup()

    chatbot = build_chatbot(checkpointer=checkpointer, store=store)
    return {
        "checkpointer_cm": checkpointer_cm,
        "store_cm":        store_cm,
        "checkpointer":    checkpointer,
        "store":           store,
        "chatbot":         chatbot,
    }

def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content).strip()

def display_role(message: Any) -> str | None:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        text = content_to_text(message.content)
        if text:
            return "assistant"
    return None

def display_messages_from_snapshot(snapshot: Any) -> list[dict[str, str]]:
    values = getattr(snapshot, "values", None) or {}
    raw_messages = values.get("messages") or []
    messages: list[dict[str, str]] = []
    for message in raw_messages:
        role = display_role(message)
        text = content_to_text(getattr(message, "content", ""))
        if role and text:
            message_id = getattr(message, "id", None) or f"{role}:{hash(text)}"
            messages.append({"id": str(message_id), "role": role, "content": text})
    return messages

def load_thread_messages(chatbot: Any, config: dict[str, Any]) -> list[dict[str, str]]:
    seen: set[str]    = set()
    ordered: list[dict[str, str]] = []
    try:
        history = list(chatbot.get_state_history(config, limit=200))
    except Exception:
        history = []
    if history:
        for snapshot in reversed(history):
            for message in display_messages_from_snapshot(snapshot):
                if message["id"] not in seen:
                    seen.add(message["id"])
                    ordered.append(message)
        return ordered
    try:
        snapshot = chatbot.get_state(config)
    except Exception:
        return []
    return display_messages_from_snapshot(snapshot)

import psycopg

def get_user_threads(user_id: str, limit: int = 50) -> list[dict]:
    try:
        with psycopg.connect(DB_URI) as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT ON (c.thread_id)
                    c.thread_id,
                    c.metadata->>'title'   AS title,
                    c.checkpoint_id        AS updated_at
                FROM   checkpoints c
                WHERE  c.metadata->>'user_id' = %s
                   OR  c.thread_id LIKE %s
                ORDER  BY c.thread_id, c.checkpoint_id DESC
                """,
                (user_id, f"%{user_id}%"),
            ).fetchall()
    except Exception:
        return []

    threads = []
    for row in rows:
        threads.append({
            "thread_id":  row[0],
            "title":      row[1] or _short_title(row[0]),
            "updated_at": row[2],
        })
    threads.sort(key=lambda x: x["updated_at"] or "", reverse=True)
    return threads[:limit]

def _short_title(thread_id: str) -> str:
    return f"Chat {thread_id[-6:]}"

def save_thread_title(thread_id: str, user_id: str, first_message: str) -> None:
    title = first_message[:60].strip().replace("\n", " ")
    try:
        with psycopg.connect(DB_URI) as conn:
            conn.execute(
                """
                UPDATE checkpoints
                SET    metadata = metadata || jsonb_build_object('title', %s, 'user_id', %s)
                WHERE  thread_id = %s
                  AND  checkpoint_id = (
                      SELECT MAX(checkpoint_id) FROM checkpoints WHERE thread_id = %s
                  )
                """,
                (title, user_id, thread_id, thread_id),
            )
            conn.commit()
    except Exception:
        pass

def final_ai_text(result: dict[str, Any]) -> str:
    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage):
            text = content_to_text(message.content)
            if text:
                return text
    return "I couldn't generate a response. Please try again."

def stream_words(text: str):
    words = text.split(" ")
    for index, word in enumerate(words):
        yield word if index == 0 else f" {word}"
        time.sleep(0.012)

def render_dot(label: str, ok: bool | None) -> str:
    cls = "dot-green" if ok is True else "dot-red" if ok is False else "dot-amber"
    return (
        f'<div class="status-row">'
        f'<span class="dot {cls}"></span>'
        f'<span>{label}</span>'
        f'</div>'
    )


# ──────────────────────────────────────────────────────────────────────────────
# BOOT
# ──────────────────────────────────────────────────────────────────────────────
init_ids()

db_ok         = False
runtime_error = ""
runtime: dict[str, Any] | None = None

try:
    runtime = get_runtime()
    db_ok   = True
except Exception as exc:
    runtime_error = str(exc)

chatbot        = runtime["chatbot"] if runtime else None
current_config = config_for()
thread_messages = load_thread_messages(chatbot, current_config) if chatbot else []

# ── WELCOME GATE — shown only on first ever visit ──
if not st.session_state.get("setup_done", False):
    st.markdown(
        """
        <div style="display:flex;flex-direction:column;align-items:center;
        justify-content:center;min-height:80vh;text-align:center;padding:20px;">
            <div style="font-size:40px;margin-bottom:16px;">📈</div>
            <div style="font-size:24px;font-weight:700;color:#ececec;
            margin-bottom:8px;">Welcome to Varsity Finance AI</div>
            <div style="font-size:14px;color:#8e8e8e;margin-bottom:32px;">
            Enter a username to start. Use the same name to resume your chats later.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input(
            "Choose your username",
            placeholder="e.g. prashant",
            key="welcome_input",
        )
        if st.button("Start Chatting →", use_container_width=True):
            if username.strip():
                st.session_state.user_id = username.strip().lower()
                st.session_state.setup_done = True
                sync_query_params()
                st.rerun()
            else:
                st.error("Please enter a username")
    st.stop()   # ← blocks rest of app from rendering until name is set
# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    # Brand
    st.markdown(
        '<div style="padding:6px 4px 10px 4px;">'
        '<div style="font-size:16px;font-weight:700;color:#ececec;letter-spacing:-0.2px;">'
        '📈 Varsity Finance AI'
        '</div>'
        '<div style="font-size:12px;color:#555;margin-top:2px;">'
        'Agentic RAG · LangGraph · Postgres'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # New Chat
    if st.button("＋  New Chat", use_container_width=True):
        st.session_state.thread_id = _new_thread_id()
        sync_query_params()
        st.rerun()

    st.divider()

    # Recent chats section label
    st.markdown('<div class="sidebar-label">Recent</div>', unsafe_allow_html=True)

    # Thread list — scrollability is handled by JS injecting overflow-y:auto
    # on [data-testid="stSidebarContent"] and the CSS change from hidden to auto.
    # st.markdown div wrappers do NOT wrap Streamlit buttons (each element gets
    # its own container), so scroll must be driven by CSS/JS on the parent.
    user_threads = get_user_threads(st.session_state.user_id)

    if not user_threads:
        st.markdown(
            '<div style="font-size:13px;color:#555;padding:6px 4px;">No chats yet</div>',
            unsafe_allow_html=True,
        )
    else:
        for t in user_threads[:18]:
            is_active = (t["thread_id"] == st.session_state.thread_id)
            label = (" ▶ " if is_active else " ") + t["title"]   # active state shown via CSS (green left border + bg)
            if st.button(
                label,
                key=f"thread_btn_{t['thread_id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.thread_id = t["thread_id"]
                sync_query_params()
                st.rerun()

    st.divider()

    # User ID input
    entered_user_id = st.text_input(
        "User ID",
        value=st.session_state.user_id,
        key="user_id_input",
    )
    cleaned_uid = entered_user_id.strip()
    if cleaned_uid and cleaned_uid != st.session_state.user_id:
        st.session_state.user_id = cleaned_uid
        sync_query_params()
        st.rerun()

    # Thread ID pill
    st.markdown(
        '<div style="font-size:12px;color:#555;text-transform:uppercase;'
        'letter-spacing:0.06em;padding:6px 0 2px 0;">Thread</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="id-pill">{st.session_state.thread_id}</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # System status
    st.markdown(
        '<div style="font-size:12px;color:#555;text-transform:uppercase;'
        'letter-spacing:0.06em;padding-bottom:4px;">Status</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        render_dot("DB connected" if db_ok else "DB offline", db_ok) +
        render_dot(f"Model: {MODEL_NAME}", True) +
        render_dot("LangGraph ready" if db_ok else "Waiting for DB", db_ok),
        unsafe_allow_html=True,
    )

    if not db_ok:
        st.markdown(
            '<div style="font-size:12px;color:#8e8e8e;padding:4px 4px 0 4px;">'
            'Start Postgres on localhost:5442 and refresh.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="margin-top:6px;"></div>', unsafe_allow_html=True)

    if st.button("↺  Refresh connection", use_container_width=True):
        get_runtime.clear()
        st.rerun()

    # Session date pinned at bottom
    st.markdown(
        f'<div class="session-date">{datetime.now().strftime("%d %b %Y, %H:%M")}</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PANEL
# ──────────────────────────────────────────────────────────────────────────────

# ── Header strip ──
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">📈 Varsity Finance AI</div>
        <div class="main-sub">Research stocks · Understand markets · Learn investing</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Error banner ──
if not db_ok:
    st.markdown(
        f"""
        <div class="error-box">
            <strong>Postgres is not connected.</strong><br>
            <span style="font-size:12px;color:#8e8e8e;">DB URI: {DB_URI}</span><br><br>
            {runtime_error}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Chat messages ──
if db_ok and not thread_messages:
    st.markdown(
        """
        <div class="empty-wrap">
            <div class="empty-icon">📈</div>
            <div class="empty-title">What would you like to explore?</div>
            <div class="empty-sub">Ask about markets — stocks, options, mutual funds, technicals or news — think with an Innerworth mindset.</div>
            <div class="suggestion-grid">
                <div class="suggestion-card">
                    <span class="s-icon">📈</span>
                    <div class="s-text">Reliance stock price today?</div>
                    <div class="s-sub">Live price & movement</div>
                </div>
                <div class="suggestion-card">
                    <span class="s-icon">🧾</span>
                    <div class="s-text">Tax rules for market gains in India?</div>
                    <div class="s-sub">Taxation & regulations</div>
                </div>
                <div class="suggestion-card">
                    <span class="s-icon">🌐</span>
                    <div class="s-text">Find Nifty volatility latest news?</div>
                    <div class="s-sub">Market news & volatility</div>
                </div>
                <div class="suggestion-card">
                    <span class="s-icon">🎯</span>
                    <div class="s-text">When to use a Bull Call Spread?</div>
                    <div class="s-sub">Options strategies</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for message in thread_messages:
        avatar = "🧑" if message["role"] == "user" else "📈"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# ── Chat input ──
prompt = st.chat_input(
    "Ask about markets, stocks, news, or finance concepts...",
    disabled=(
        not db_ok
        or not bool(st.session_state.user_id.strip())
        or not bool(st.session_state.thread_id.strip())
    ),
)

if prompt and prompt.strip() and chatbot:
    user_input = prompt.strip()

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="📈"):
        thinking = st.empty()
        thinking.markdown(
            '<div style="display:flex;align-items:center;gap:6px;padding:4px 0;">'
            '<span class="thinking-dot"></span>'
            '<span class="thinking-dot"></span>'
            '<span class="thinking-dot"></span>'
            '</div>',
            unsafe_allow_html=True,
        )

        try:
            result = chatbot.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={
                    "configurable": {
                        "thread_id": st.session_state.thread_id,
                        "user_id":   st.session_state.user_id,
                    }
                },
            )
            ai_text = final_ai_text(result)

            if len(thread_messages) == 0:
                save_thread_title(
                    st.session_state.thread_id,
                    st.session_state.user_id,
                    user_input,
                )

        except Exception as exc:
            import traceback
            traceback.print_exc()
            ai_text = f"Agent error: {str(exc).encode('utf-8', errors='replace').decode('utf-8')}"

        thinking.empty()
        st.write_stream(stream_words(ai_text))