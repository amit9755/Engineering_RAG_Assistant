"""
HTTP access to Jira and Confluence, Cloud or Server / Data Center.

Cloud (*.atlassian.net): account email + API token as Basic auth.
Server / Data Center (company-hosted, e.g. jira.sw.nxp.com): a Personal Access
Token sent as a Bearer token; if that is rejected and a username was given,
username + token (or password) as Basic auth is tried, as older servers expect.
"""

from urllib.parse import urlparse

import requests

from src.sources.jobs import IndexingError


def is_cloud(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host.endswith(".atlassian.net")


class AtlassianSession:
    def __init__(self, base_url: str, username: str, token: str, product: str, session=None):
        self.base_url = base_url.rstrip("/")
        self.username, self.token, self.product = (username or "").strip(), token, product
        self.cloud = is_cloud(self.base_url)
        self.http = session or requests.Session()

    def get(self, path: str, params=None, timeout: int = 60):
        """GET base_url + path. Raises IndexingError with a readable message for network / auth failures."""
        from src import network_policy
        url = f"{self.base_url}{path}"
        try:
            network_policy.check_url(url, f"a {self.product} request")
        except network_policy.ExternalNetworkBlocked as exc:
            raise IndexingError(str(exc)) from exc
        headers = {"Accept": "application/json"}
        try:
            if self.cloud:
                response = self.http.get(url, params=params, auth=(self.username, self.token),
                                         headers=headers, timeout=timeout)
            else:
                response = self.http.get(url, params=params, timeout=timeout,
                                         headers={**headers, "Authorization": f"Bearer {self.token}"})
                if response.status_code == 401 and self.username:
                    response = self.http.get(url, params=params, auth=(self.username, self.token),
                                             headers=headers, timeout=timeout)
        except requests.exceptions.SSLError as exc:
            raise IndexingError(f"SSL certificate check failed for {self.base_url}. On a company network, "
                                "install pip-system-certs in the app's Python environment.") from exc
        except requests.RequestException as exc:
            raise IndexingError(f"Could not reach {self.product} at {self.base_url}: {exc.__class__.__name__}. "
                                "Check the URL and that you are on the company network or VPN.") from exc
        if response.status_code == 401:
            raise IndexingError(self.auth_hint())
        if response.status_code == 403:
            raise IndexingError(f"The {self.product} account lacks permission for this request.")
        return response

    def auth_hint(self) -> str:
        if self.cloud:
            return (f"{self.product} rejected the credentials. For {self.product} Cloud use your account email "
                    "and an API token from id.atlassian.com > Security > API tokens.")
        return (f"{self.product} rejected the token. For a company-hosted {self.product} use a Personal Access "
                f"Token ({self.product}: your avatar > Profile > Personal Access Tokens > Create token); "
                "leave the username empty or use your login ID, not your email.")
