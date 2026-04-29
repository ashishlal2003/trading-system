#!/bin/bash
TOKEN=$(grep GROWW_API_KEY .env | cut -d= -f2-)
CODE=$(python3 -c "import pyotp; print(pyotp.TOTP('C2HKQPPGHXYSQ7HJWU3HD7D2JMM7ULXE').now())")
echo "Code: $CODE"
curl -s -X POST "https://api.groww.in/v1/token/api/access" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d "{\"key_type\":\"twofa\",\"totp\":\"$CODE\"}"
