import httpx
import pyotp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.utils.logger import get_logger

logger = get_logger(__name__)

_AUTH_URL = "https://api.groww.in/v1/token/api/access"


class GrowwAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Groww API error {status_code}: {message}")


class GrowwClient:
    """
    Async authenticated client for Groww REST API.
    GROWW_API_KEY is the intermediate JWT from the dashboard.
    GROWW_TOTP_SECRET exchanges it for a live access token at startup and 06:05 IST daily.
    Must be used as async context manager.
    """

    def __init__(self, api_key: str, api_secret: str, base_url: str, totp_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.totp_secret = totp_secret
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "GrowwClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        logger.info("groww_client_ready", base_url=self.base_url)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def refresh_access_token(self) -> str:
        """Exchange the TOTP intermediate JWT for a live access token."""
        if not self.totp_secret:
            raise RuntimeError("GROWW_TOTP_SECRET not set in .env")

        code = pyotp.TOTP(self.totp_secret).now()
        logger.info("groww_token_refresh.attempt", totp_code=code)

        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as auth_client:
            response = await auth_client.post(
                _AUTH_URL,
                json={"key_type": "twofa", "totp": code},
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )

        if response.status_code >= 400:
            raise GrowwAPIError(response.status_code, response.text)

        data = response.json()
        token = (data.get("payload") or {}).get("token") or data.get("token") or data.get("access_token") or ""
        if not token:
            raise GrowwAPIError(200, f"No token in response: {data}")

        self.api_key = token
        logger.info("groww_token_refresh.success", token_prefix=token[:12])
        return token

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
