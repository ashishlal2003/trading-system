#!/usr/bin/env python3
"""
Verify Groww credentials and do the TOTP exchange.
Run each morning after pasting fresh API key from groww.in/trade-api.

    python scripts/check_groww_key.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.broker.groww_client import GrowwClient, GrowwAPIError


async def main() -> None:
    print(f"Groww base URL   : {settings.GROWW_BASE_URL}")
    print(f"API key prefix   : {settings.GROWW_API_KEY[:20]}...")
    print(f"TOTP configured  : {'YES' if settings.GROWW_TOTP_SECRET else 'NO'}\n")

    async with GrowwClient(
        api_key=settings.GROWW_API_KEY,
        api_secret=settings.GROWW_API_SECRET,
        base_url=settings.GROWW_BASE_URL,
        totp_secret=settings.GROWW_TOTP_SECRET,
    ) as client:

        print("Exchanging API key for access token via TOTP...")
        try:
            token = await client.refresh_access_token()
            print(f"✅  TOTP exchange OK — access token prefix: {token[:20]}...\n")
        except GrowwAPIError as e:
            print(f"❌  TOTP exchange FAILED (HTTP {e.status_code}): {e.message}")
            print("\nCheck GROWW_API_KEY (fresh from Groww dashboard) and GROWW_TOTP_SECRET in .env")
            sys.exit(1)
        except Exception as e:
            print(f"⚠️  Error: {e}")
            sys.exit(1)

        print("Testing live API access via /holdings/user ...")
        try:
            data = await client.get("/holdings/user")
            print(f"✅  API access OK — holdings: {str(data)[:200]}")
        except GrowwAPIError as e:
            print(f"❌  API call failed (HTTP {e.status_code}): {e.message}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
