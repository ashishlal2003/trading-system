#!/usr/bin/env python3
"""
Quick sanity-check: verifies the Groww API key / TOTP in .env is still valid.
Run this on the EC2 before market open each morning.

    python scripts/check_groww_key.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.broker.groww_client import GrowwClient, GrowwAPIError


async def main() -> None:
    print(f"Groww base URL  : {settings.GROWW_BASE_URL}")
    print(f"API key prefix  : {settings.GROWW_API_KEY[:12]}...")
    totp_configured = bool(settings.GROWW_TOTP_SECRET)
    print(f"TOTP configured : {'YES' if totp_configured else 'NO (using static key)'}\n")

    async with GrowwClient(
        api_key=settings.GROWW_API_KEY,
        api_secret=settings.GROWW_API_SECRET,
        base_url=settings.GROWW_BASE_URL,
        totp_secret=settings.GROWW_TOTP_SECRET,
    ) as client:

        # If TOTP is set, refresh first then test
        if totp_configured:
            print("Refreshing token via TOTP...")
            try:
                token = await client.refresh_access_token()
                print(f"✅  TOTP refresh OK — new token prefix: {token[:12]}...\n")
            except GrowwAPIError as e:
                print(f"❌  TOTP refresh FAILED (HTTP {e.status_code}): {e.message}")
                print("\nCheck that GROWW_TOTP_SECRET is the 32-char secret from Groww API dashboard.")
                sys.exit(1)
            except Exception as e:
                print(f"⚠️  TOTP refresh error: {e}")
                sys.exit(1)

        # Test the token (refreshed or static) against a real endpoint
        print("Testing API access...")
        try:
            data = await client.get("/user/profile")
            print(f"✅  API key is VALID — profile: {data}")
        except GrowwAPIError as e:
            if e.status_code in (401, 403):
                print(f"❌  Token EXPIRED or INVALID (HTTP {e.status_code}): {e.message}")
                if not totp_configured:
                    print("\nFix: set GROWW_TOTP_SECRET in .env to enable auto-refresh, or manually renew GROWW_API_KEY.")
                sys.exit(1)
            else:
                print(f"⚠️  API returned HTTP {e.status_code}: {e.message}")
                sys.exit(1)
        except Exception as e:
            print(f"⚠️  Connection error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
