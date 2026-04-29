#!/bin/bash
# GROWW_API_KEY must be the TOTP JWT (from the "TOTP token" modal, NOT "API key and secret" modal)
TOKEN=$(grep GROWW_API_KEY .env | cut -d= -f2-)
SECRET=$(grep GROWW_TOTP_SECRET .env | cut -d= -f2-)
CODE=$(python3 -c "import pyotp; print(pyotp.TOTP('$SECRET').now())")
echo "Code: $CODE"
curl -s -X POST "https://api.groww.in/v1/token/api/access" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d "{\"key_type\":\"totp\",\"totp\":\"$CODE\"}"
