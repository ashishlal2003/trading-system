#!/usr/bin/env python3
"""
Quick sanity-check: verifies the Groww API key in .env is still valid.
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
    print(f"Groww base URL : {settings.GROWW_BASE_URL}")
    print(f"API key prefix : {settings.GROWW_API_KEY[:12]}...")
    print("Testing API key...\n")

    async with GrowwClient(
        api_key=settings.GROWW_API_KEY,
        api_secret=settings.GROWW_API_SECRET,
        base_url=settings.GROWW_BASE_URL,
    ) as client:
        try:
            data = await client.get("/user/profile")
            print(f"✅  Key is VALID — profile response: {data}")
        except GrowwAPIError as e:
            if e.status_code in (401, 403):
                print(f"❌  Key is EXPIRED or INVALID (HTTP {e.status_code}): {e.message}")
                print("\nFix: log in to Groww, get a new API token, update GROWW_API_KEY in .env")
                sys.exit(1)
            else:
                print(f"⚠️  API returned HTTP {e.status_code}: {e.message}")
                sys.exit(1)
        except Exception as e:
            print(f"⚠️  Connection error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
