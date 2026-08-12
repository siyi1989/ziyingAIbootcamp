import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="Vendors@Gov Billing Assistant",
    page_icon="💬",
    layout="wide",
)

# ---------- Required Disclaimer (shown on every page) ----------
with st.expander("⚠️ Required Disclaimer — Please Read", expanded=False):
    st.markdown(
        """
**IMPORTANT NOTICE:** This web application is developed as a proof-of-concept
prototype. The information provided here is **NOT intended for actual usage**
and should not be relied upon for making any decisions, especially those
related to financial or billing matters.

**This assistant looks up records from a vendor data file and may return
incomplete or incorrect results if the underlying data is outdated or
mislabelled. You assume full responsibility for how you use any output
generated here.**

Always verify customer codes and billing details against the official
Vendors@Gov system before use.
        """
    )

# ---------- Session state defaults ----------
for key, default in {
    "authenticated": False,
    "role": None,
    "username": None,
    "login_error": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def check_login(username: str, password: str):
    admin_user = os.getenv("ADMIN_USERNAME", "admin").strip()
    admin_pass = os.getenv("ADMIN_PASSWORD", "admin123").strip()
    username = (username or "").strip()
    password = (password or "").strip()
    if username == admin_user and password == admin_pass:
        return "Admin"
    return None


def do_login():
    username = st.session_state.get("login_username", "")
    password = st.session_state.get("login_password", "")
    role = check_login(username, password)
    if role:
        st.session_state.authenticated = True
        st.session_state.role = role
        st.session_state.username = username
        st.session_state.login_error = ""
    else:
        st.session_state.authenticated = False
        st.session_state.role = None
        st.session_state.username = None
        st.session_state.login_error = "Invalid username or password."
    st.rerun()


# ---------- Admin login lives in the sidebar ----------
with st.sidebar:
    if not st.session_state.authenticated:
        with st.expander("🔐 Admin Login"):
            with st.form("login_form"):
                st.text_input("Username", key="login_username")
                st.text_input("Password", type="password", key="login_password")
                st.form_submit_button("Log in", on_click=do_login)
                if st.session_state.login_error:
                    st.error(st.session_state.login_error)
    else:
        st.success(f"Logged in as **{st.session_state.username}** (Admin)")
        if st.button("Log out"):
            st.session_state.authenticated = False
            st.session_state.role = None
            st.session_state.username = None
            st.rerun()

# ---------- Pages ----------
chat_page = st.Page(
    "pages/1_Chat_Assistant.py",
    title="Chat Assistant",
    icon="💬",
    default=True,
)

pages = [chat_page]

if st.session_state.role == "Admin":
    admin_page = st.Page(
        "pages/2_Admin_Upload.py",
        title="Admin Upload",
        icon="📤",
    )
    pages.append(admin_page)

pg = st.navigation(pages)
pg.run()
