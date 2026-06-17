import streamlit as st
from core.utils import get_supabase_client

def render_auth_ui():
    """Renders the login and signup forms with hArI's exact styling and SVG logo."""
    
        # 1. Custom CSS just for the Auth Screen
    st.markdown("""
    <style>
        /* 1. Hide the Deploy Button */
        .stDeployButton {display: none !important;}
        
        /* 2. Hide the Hamburger Menu & Running Indicator */
        [data-testid="stToolbar"] {display: none !important;}
        
        /* 3. Make the Header transparent so it doesn't block the logo */
        [data-testid="stHeader"] {background-color: transparent !important;}
        
        /* 4. Hide Streamlit footer */
        footer {display: none !important;}

        /* Override Streamlit Tab Colors (Replace Red with Purple) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            color: #888 !important;
            font-family: monospace;
        }
        .stTabs [aria-selected="true"] {
            color: #7F77DD !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #7F77DD !important;
        }
        
        /* Style the Form Container */
        [data-testid="stForm"] {
            background-color: #0e0e0e !important;
            border: 0.5px solid #2a2a2a !important;
            border-radius: 12px;
            padding: 20px;
        }

        /* Style the Submit Buttons */
        [data-testid="stFormSubmitButton"] > button {
            width: 100%;
            background-color: #141414 !important;
            border: 0.5px solid #3a3a3a !important;
            color: #c8c8c8 !important;
            padding: 10px;
            transition: all 0.2s ease-in-out;
        }
        [data-testid="stFormSubmitButton"] > button:hover {
            border-color: #7F77DD !important;
            color: #fff !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # 2. Render YOUR exact Header (Centered for the login screen)
    st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; margin-top: 80px; margin-bottom: 30px;">
            <div style="display:flex; align-items:center; gap:14px;">
                <svg width="42" height="42" viewBox="0 0 36 36" fill="none">
                    <rect width="36" height="36" rx="8" fill="#1a1a1a"/>
                    <circle cx="18" cy="18" r="3" fill="#7F77DD"/>
                    <line x1="18" y1="8"  x2="18" y2="13" stroke="#534AB7" stroke-width="1.5" stroke-linecap="round"/>
                    <line x1="18" y1="23" x2="18" y2="28" stroke="#534AB7" stroke-width="1.5" stroke-linecap="round"/>
                    <line x1="8"  y1="18" x2="13" y2="18" stroke="#534AB7" stroke-width="1.5" stroke-linecap="round"/>
                    <line x1="23" y1="18" x2="28" y2="18" stroke="#534AB7" stroke-width="1.5" stroke-linecap="round"/>
                    <line x1="11.5" y1="11.5" x2="14.9" y2="14.9" stroke="#3C3489" stroke-width="1.2" stroke-linecap="round"/>
                    <line x1="21.1" y1="21.1" x2="24.5" y2="24.5" stroke="#3C3489" stroke-width="1.2" stroke-linecap="round"/>
                    <line x1="24.5" y1="11.5" x2="21.1" y2="14.9" stroke="#3C3489" stroke-width="1.2" stroke-linecap="round"/>
                    <line x1="14.9" y1="21.1" x2="11.5" y2="24.5" stroke="#3C3489" stroke-width="1.2" stroke-linecap="round"/>
                </svg>
                <div>
                    <div style="font-size:32px; font-weight:500; color:#e8e8e8; letter-spacing:-0.5px; line-height:1.1">
                        h<span style="color:#7F77DD">A</span>r<span style="color:#7F77DD">I</span>
                    </div>
                    <div style="font-size:12px; color:#666; margin-top:2px; font-family: monospace;">document intelligence</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    supabase = get_supabase_client()

    # 3. Create the Tabs inside a centered layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="admin@hari.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submit = st.form_submit_button("Authenticate ➔")

                if submit:
                    try:
                        response = supabase.auth.sign_in_with_password({
                            "email": email,
                            "password": password
                        })
                        # Save the user AND the auth tokens to memory!
                        st.session_state["user"] = response.user
                        st.session_state["access_token"] = response.session.access_token
                        st.session_state["refresh_token"] = response.session.refresh_token
                        st.rerun()
                    except Exception as e:
                        st.error(f"Login failed: Check your credentials.")

        with tab_signup:
            with st.form("signup_form"):
                new_email = st.text_input("Email", placeholder="you@company.com")
                new_password = st.text_input("Password", type="password", placeholder="Create a strong password")
                submit_signup = st.form_submit_button("Create Account")

                if submit_signup:
                    try:
                        response = supabase.auth.sign_up({
                            "email": new_email,
                            "password": new_password
                        })
                        st.success("Account created! You can now log in.")
                    except Exception as e:
                        st.error(f"Signup failed: {e}")