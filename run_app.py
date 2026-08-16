"""
Run Streamlit Dashboard
=======================
Launches the Streamlit Web Application on http://localhost:8501
"""

import sys
from streamlit.web import cli as stcli

if __name__ == "__main__":
    print("Launching Karachi AQI Streamlit Dashboard on http://localhost:8501 ...")
    sys.argv = ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    sys.exit(stcli.main())
