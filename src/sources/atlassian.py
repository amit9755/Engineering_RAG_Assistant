"""
HTTP access to Jira and Confluence, Cloud or Server / Data Center.

Cloud (*.atlassian.net): account email + API token as Basic auth.
Server / Data Center (company-hosted, e.g. jira.sw.nxp.com): a Personal Access
Token sent as a Bearer token; if that is rejected and a username was given,
username + token (or password) as Basic auth is tried, as older servers expect.
"""

import re
from html import unescape
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

    def post(self, path: str, body: dict, timeout: int = 60):
        """POST JSON to base_url + path (creating issues). Same auth, policy and errors as get()."""
        from src import network_policy
        url = f"{self.base_url}{path}"
        try:
            network_policy.check_url(url, f"a {self.product} request")
        except network_policy.ExternalNetworkBlocked as exc:
            raise IndexingError(str(exc)) from exc
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        try:
            if self.cloud:
                response = self.http.post(url, json=body, auth=(self.username, self.token), headers=headers,
                                          timeout=timeout)
            else:
                response = self.http.post(url, json=body, timeout=timeout,
                                          headers={**headers, "Authorization": f"Bearer {self.token}"})
                if response.status_code == 401 and self.username:
                    response = self.http.post(url, json=body, auth=(self.username, self.token), headers=headers,
                                              timeout=timeout)
        except requests.RequestException as exc:
            raise IndexingError(f"Could not reach {self.product} at {self.base_url}: {exc.__class__.__name__}.") from exc
        if response.status_code == 401:
            raise IndexingError(self.auth_hint())
        return response

    def json(self, response, what: str = "request"):
        """
        Parse a JSON response. A company server that answers with a web page (an SSO /
        login page, a proxy notice, or an error page for an unknown address) raises a
        readable IndexingError instead of a JSONDecodeError.
        """
        content_type = (response.headers.get("Content-Type") or "").lower() if hasattr(response, "headers") else ""
        try:
            return response.json()
        except ValueError:
            body = response.text or "" if isinstance(getattr(response, "text", None), str) else ""
            title = re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S)
            title = " ".join(unescape(title.group(1)).split())[:120] if title else ""
            final_url = getattr(response, "url", "") or ""
            login = bool(re.search(r"login|sso|saml|signin|auth", f"{title} {final_url}", re.I))
            hint = (" The server sent a login page: the token was not accepted for API access, or the request was "
                    "redirected to single sign-on. Check the token, and that the URL is the server's base address."
                    if login else " Check that the URL is the server's base address (as shown in your browser).")
            raise IndexingError(f"{self.product} returned a web page instead of data for the {what} "
                                f"(HTTP {response.status_code}{', ' + content_type.split(';')[0] if content_type else ''}"
                                f"{', page: ' + repr(title) if title else ''}).{hint}")

    def auth_hint(self) -> str:
        if self.cloud:
            return (f"{self.product} rejected the credentials. For {self.product} Cloud use your account email "
                    "and an API token from id.atlassian.com > Security > API tokens.")
        return (f"{self.product} rejected the token. For a company-hosted {self.product} use a Personal Access "
                f"Token ({self.product}: your avatar > Profile > Personal Access Tokens > Create token); "
                "leave the username empty or use your login ID, not your email.")
