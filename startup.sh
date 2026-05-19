#!/bin/bash
pip install -r requirements.txt

streamlit run streamlit_prod_frontend.py \
--server.port 8000 \
--server.address 0.0.0.0 \
--server.enableCORS false \
--server.enableXsrfProtection false