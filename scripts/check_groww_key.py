#!/usr/bin/env python3
"""
Quick sanity-check: verifies the Groww API key in .env is still valid.
Run this on the server before market open each morning after pasting the
fresh API key from the Groww dashboard.

    python scripts/check_groww_key.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.broker.groww_client import GrowwClient, GrowwAPIError


async def main() -> None:
    print(f"Groww base URL : {settings.GROWW_BASE_URL}")
    print(f"API key prefix : {settings.GROWW_API_KEY[:20]}...\n")

    async with GrowwClient(
        api_key=settings.GROWW_API_KEY,
        api_secret=settings.GROWW_API_SECRET,
        base_url=settings.GROWW_BASE_URL,
    ) as client:

        print("Testing API access via /holdings/user ...")
        try:
            data = await client.get("/holdings/user")
            print(f"✅  API key is VALID — holdings response: {str(data)[:200]}")
        except GrowwAPIError as e:
            if e.status_code in (401, 403):
                print(f"❌  Token EXPIRED or INVALID (HTTP {e.status_code}): {e.message}")
                print("\nFix: paste today's fresh API key from groww.in/trade-api into GROWW_API_KEY in .env")
            else:
                print(f"⚠️  API returned HTTP {e.status_code}: {e.message}")
            sys.exit(1)
        except Exception as e:
            print(f"⚠️  Connection error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
