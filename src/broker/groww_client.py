import httpx
import pyotp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Groww auth endpoint for TOTP-based token exchange
_AUTH_URL = "https://api.groww.in/v1/token/api/access"


class GrowwAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Groww API error {status_code}: {message}")


class GrowwClient:
    """
    Async authenticated client for Groww REST API.

    Two auth modes:
    - Static token (legacy): pass api_key as the Bearer token. Expires 6 AM IST daily,
      requires manual renewal each morning.
    - TOTP (recommended): pass totp_secret (32-char string from Groww API dashboard).
      Call refresh_access_token() at startup and then daily at 06:05 IST to get a
      fresh token automatically — no manual steps needed.

    Must be used as async context manager.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str,
        totp_secret: str = "",
    ):
        self.api_key = api_key          # holds the current live Bearer token
        self.api_secret = api_secret
        self.base_url = base_url
        self.totp_secret = totp_secret  # empty string = TOTP disabled
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "GrowwClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        logger.info("groww_client_ready", base_url=self.base_url, totp_enabled=bool(self.totp_secret))
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # TOTP token refresh
    # ------------------------------------------------------------------

    async def refresh_access_token(self) -> str:
        """
        Use the TOTP secret to get a fresh Groww access token.

        Generates the current 6-digit TOTP code from totp_secret, posts it to
        Groww's auth endpoint along with api_key (which is the Groww API key,
        not the access token), and stores the returned access token in
        self.api_key so all subsequent requests use it automatically.

        Raises GrowwAPIError if auth fails (wrong secret, network error, etc.).
        """
        if not self.totp_secret:
            raise RuntimeError(
                "TOTP refresh called but GROWW_TOTP_SECRET is not set in .env"
            )

        totp = pyotp.TOTP(self.totp_secret)
        current_code = totp.now()

        logger.info("groww_token_refresh.attempt")

        # Use a short-lived client that hits the root auth URL (not base_url)
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as auth_client:
            response = await auth_client.post(
                _AUTH_URL,
                json={
                    "key_type": "totp",
                    "totp": current_code,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )

        if response.status_code >= 400:
            try:
                msg = response.json().get("message", response.text)
            except Exception:
                msg = response.text
            raise GrowwAPIError(response.status_code, msg)

        data = response.json()
        # Groww returns token under payload.token
        token = (
            (data.get("payload") or {}).get("token")
            or data.get("token")
            or data.get("access_token")
            or ""
        )
        if not token:
            raise GrowwAPIError(200, f"No access_token in response: {data}")

        self.api_key = token
        logger.info("groww_token_refresh.success", token_prefix=token[:12])
        return token

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-API-VERSION": "1.0",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _handle_response(self, response: httpx.Response) -> dict:
        logger.debug("groww_response", status=response.status_code, url=str(response.url))
        if response.status_code >= 400:
            try:
                msg = response.json().get("message", response.text)
            except Exception:
                msg = response.text
            raise GrowwAPIError(response.status_code, msg)
        return response.json()

    # ------------------------------------------------------------------
    # HTTP verbs
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.TransportError),
        reraise=True,
    )
    async def get(self, endpoint: str, params: dict | None = None) -> dict:
        logger.debug("groww_get", endpoint=endpoint, params=params)
        response = await self._client.get(endpoint, params=params, headers=self._headers())
        return self._handle_response(response)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.TransportError),
        reraise=True,
    )
    async def post(self, endpoint: str, payload: dict) -> dict:
        logger.debug("groww_post", endpoint=endpoint)
        response = await self._client.post(endpoint, json=payload, headers=self._headers())
        return self._handle_response(response)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.TransportError),
        reraise=True,
    )
    async def delete(self, endpoint: str) -> dict:
        response = await self._client.delete(endpoint, headers=self._headers())
        return self._handle_response(response)
