#!/usr/bin/env python3
"""
claude-proxy: Lightweight OpenAI-compatible proxy for Claude models via Google's internal API.
Zero external dependencies — runs on stdlib alone (Python 3.10+).
Designed for Termux/Android where native extensions can't compile.

Usage:
    export GEMINI_AUTH_PASSWORD="your_password"
    python claude_proxy.py

First run opens a Google OAuth URL in terminal. Visit it, auth, paste the redirect URL back.
Credentials saved to oauth_creds.json for automatic refresh.
"""

import json
import os
import sys
import time
import uuid
import logging
import threading
import urllib.request
import urllib.parse
import urllib.error
import ssl
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone, timedelta
from io import BytesIO

# ─── Configuration ───────────────────────────────────────────────────────────

PORT = int(os.getenv("PORT", "7860"))
HOST = os.getenv("HOST", "0.0.0.0")
AUTH_PASSWORD = os.getenv("GEMINI_AUTH_PASSWORD", "123456")
CREDENTIAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oauth_creds.json")
CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"

# OAuth — same client ID as Gemini CLI / geminicli2api
CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
CLIENT_SECRET = "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"
SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]
TOKEN_URI = "https://oauth2.googleapis.com/token"
AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
OAUTH_CALLBACK_PORT = 8080

# Safety — BLOCK_NONE on everything
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_IMAGE_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_IMAGE_HATE", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_UNSPECIFIED", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_JAILBREAK", "threshold": "BLOCK_NONE"},
]

# ─── Model Mapping ──────────────────────────────────────────────────────────

CLAUDE_MODEL_MAP = {
    "claude-opus-4-6": "claude-opus-4-6@default",
    "claude-opus-4.6": "claude-opus-4-6@default",
    "claude-opus-4-5": "claude-opus-4-5@20251101",
    "claude-opus-4.5": "claude-opus-4-5@20251101",
    "claude-opus-4": "claude-opus-4@20250514",
    "claude-opus-4.0": "claude-opus-4@20250514",
    "claude-sonnet-4-6": "claude-sonnet-4-6@default",
    "claude-sonnet-4.6": "claude-sonnet-4-6@default",
    "claude-sonnet-4-5": "claude-sonnet-4-5@20250929",
    "claude-sonnet-4.5": "claude-sonnet-4-5@20250929",
    "claude-sonnet-4": "claude-sonnet-4@20250514",
    "claude-sonnet-4.0": "claude-sonnet-4@20250514",
    "claude-haiku-4-5": "claude-haiku-4-5@20251001",
    "claude-haiku-4.5": "claude-haiku-4-5@20251001",
}

GEMINI_MODELS = [
    "gemini-2.5-pro", "gemini-2.5-pro-preview-06-05", "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-flash", "gemini-2.5-flash-preview-05-20",
    "gemini-3-pro-preview", "gemini-3-flash-preview",
]

ALL_MODELS = list(CLAUDE_MODEL_MAP.keys()) + GEMINI_MODELS


def resolve_model(name: str) -> str:
    """Resolve user-facing model name to Google's internal identifier."""
    clean = name.replace("models/", "").strip()
    if clean in CLAUDE_MODEL_MAP:
        return CLAUDE_MODEL_MAP[clean]
    return clean


def is_claude(name: str) -> bool:
    return "claude" in name.lower()


# ─── OAuth (Pure stdlib) ────────────────────────────────────────────────────

class TokenStore:
    """Manages OAuth tokens with auto-refresh. No google-auth dependency."""

    def __init__(self):
        self.access_token = None
        self.refresh_token = None
        self.expiry = None
        self.project_id = None
        self._lock = threading.Lock()
        self._onboarded = False

    def load(self) -> bool:
        """Load tokens from disk."""
        if not os.path.exists(CREDENTIAL_FILE):
            return False
        try:
            with open(CREDENTIAL_FILE, "r") as f:
                data = json.load(f)
            self.access_token = data.get("token") or data.get("access_token")
            self.refresh_token = data.get("refresh_token")
            self.project_id = data.get("project_id")
            exp = data.get("expiry")
            if exp:
                try:
                    self.expiry = datetime.fromisoformat(exp.replace("Z", "+00:00"))
                except Exception:
                    self.expiry = None
            return bool(self.refresh_token)
        except Exception as e:
            logging.error(f"Failed to load credentials: {e}")
            return False

    def save(self):
        """Persist tokens to disk."""
        data = {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_uri": TOKEN_URI,
            "scopes": SCOPES,
        }
        if self.expiry:
            data["expiry"] = self.expiry.isoformat()
        if self.project_id:
            data["project_id"] = self.project_id
        with open(CREDENTIAL_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def is_expired(self) -> bool:
        if not self.expiry:
            return True
        now = datetime.now(timezone.utc)
        return now >= self.expiry

    def refresh(self) -> bool:
        """Refresh the access token using the refresh token."""
        if not self.refresh_token:
            return False
        with self._lock:
            # Double-check after acquiring lock
            if not self.is_expired() and self.access_token:
                return True
            try:
                body = urllib.parse.urlencode({
                    "client_id": CLIENT_ID,
                    "client_secret": CLIENT_SECRET,
                    "refresh_token": self.refresh_token,
                    "grant_type": "refresh_token",
                }).encode()
                req = urllib.request.Request(TOKEN_URI, data=body, method="POST")
                req.add_header("Content-Type", "application/x-www-form-urlencoded")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                self.access_token = data["access_token"]
                expires_in = data.get("expires_in", 3600)
                self.expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
                if "refresh_token" in data:
                    self.refresh_token = data["refresh_token"]
                self.save()
                logging.info("Token refreshed successfully")
                return True
            except Exception as e:
                logging.error(f"Token refresh failed: {e}")
                return False

    def get_token(self) -> str | None:
        """Get a valid access token, refreshing if necessary."""
        if self.is_expired():
            if not self.refresh():
                return None
        return self.access_token

    def discover_project(self) -> str | None:
        """Discover project ID from Google's API."""
        # Check env vars first
        env_proj = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GEMINI_PROJECT_ID")
        if env_proj:
            self.project_id = env_proj
            self.save()
            logging.info(f"Using project ID from env: {env_proj}")
            return env_proj

        if self.project_id:
            return self.project_id

        token = self.get_token()
        if not token:
            return None

        try:
            payload = json.dumps({
                "metadata": _client_metadata(),
            }).encode()
            req = urllib.request.Request(
                f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
                data=payload, method="POST"
            )
            req.add_header("Authorization", f"Bearer {token}")
            req.add_header("Content-Type", "application/json")
            req.add_header("User-Agent", _user_agent())
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            proj = data.get("cloudaicompanionProject")
            if proj:
                self.project_id = proj
                self.save()
                logging.info(f"Discovered project: {proj}")
                return proj
            else:
                logging.error(f"loadCodeAssist response had no project: {json.dumps(data)}")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            logging.error(f"Project discovery HTTP {e.code}: {error_body}")
        except Exception as e:
            logging.error(f"Project discovery failed: {e}")
        return None

    def onboard(self) -> bool:
        """Onboard the user (one-time setup)."""
        if self._onboarded:
            return True
        token = self.get_token()
        proj = self.discover_project()
        if not token or not proj:
            return False
        try:
            payload = json.dumps({
                "cloudaicompanionProject": proj,
                "metadata": _client_metadata(proj),
            }).encode()
            req = urllib.request.Request(
                f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
                data=payload, method="POST"
            )
            req.add_header("Authorization", f"Bearer {token}")
            req.add_header("Content-Type", "application/json")
            req.add_header("User-Agent", _user_agent())
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            if data.get("currentTier"):
                self._onboarded = True
                logging.info("Already onboarded")
                return True

            # Need to onboard
            tier = None
            for t in data.get("allowedTiers", []):
                if t.get("isDefault"):
                    tier = t
                    break
            if not tier:
                tier = {"id": "legacy-tier", "userDefinedCloudaicompanionProject": True}

            onboard_payload = json.dumps({
                "tierId": tier.get("id"),
                "cloudaicompanionProject": proj,
                "metadata": _client_metadata(proj),
            }).encode()

            for _ in range(12):  # Max 60s
                req2 = urllib.request.Request(
                    f"{CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
                    data=onboard_payload, method="POST"
                )
                req2.add_header("Authorization", f"Bearer {token}")
                req2.add_header("Content-Type", "application/json")
                req2.add_header("User-Agent", _user_agent())
                with urllib.request.urlopen(req2, timeout=30) as resp2:
                    lro = json.loads(resp2.read())
                if lro.get("done"):
                    self._onboarded = True
                    logging.info("Onboarding complete")
                    return True
                time.sleep(5)

        except Exception as e:
            logging.error(f"Onboarding failed: {e}")
        return False


def _user_agent():
    import platform
    system = platform.system()
    arch = platform.machine()
    return "antigravity/1.11.9 windows/amd64"


def _get_platform():
    import platform
    system = platform.system().upper()
    arch = platform.machine().upper()
    if system == "DARWIN":
        return "DARWIN_ARM64" if arch in ("ARM64", "AARCH64") else "DARWIN_AMD64"
    elif system == "LINUX":
        return "LINUX_ARM64" if arch in ("ARM64", "AARCH64") else "LINUX_AMD64"
    elif system == "WINDOWS":
        return "WINDOWS_AMD64"
    return "PLATFORM_UNSPECIFIED"


def _client_metadata(project_id=None):
    return {
        "ideType": "IDE_UNSPECIFIED",
        "platform": _get_platform(),
        "pluginType": "GEMINI",
        "duetProject": project_id,
    }


# ─── OAuth Flow (first-time auth) ───────────────────────────────────────────

class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    auth_code = None

    def do_GET(self):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        code = query.get("code", [None])[0]
        if code:
            _OAuthCallbackHandler.auth_code = code
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Auth successful!</h1><p>You can close this tab. Check terminal.</p>")
        else:
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Auth failed.</h1><p>No code received.</p>")

    def log_message(self, format, *args):
        pass  # Suppress default logging


def run_oauth_flow(store: TokenStore) -> bool:
    """Run the interactive OAuth flow for first-time auth."""
    redirect_uri = f"http://localhost:{OAUTH_CALLBACK_PORT}"
    params = urllib.parse.urlencode({
        "client_id": CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
    })
    auth_url = f"{AUTH_URI}?{params}"

    print("\n" + "=" * 70)
    print("  AUTHENTICATION REQUIRED")
    print("=" * 70)
    print(f"\n  Open this URL in your browser:\n")
    print(f"  {auth_url}\n")
    print("=" * 70)
    print("  Waiting for OAuth callback on port", OAUTH_CALLBACK_PORT, "...")
    print("=" * 70 + "\n")

    # Start callback server
    try:
        server = HTTPServer(("", OAUTH_CALLBACK_PORT), _OAuthCallbackHandler)
        server.timeout = 300  # 5 min timeout
        server.handle_request()
    except OSError as e:
        logging.error(f"Could not start OAuth callback server: {e}")
        print(f"\nPort {OAUTH_CALLBACK_PORT} may be in use. Kill other processes or change OAUTH_CALLBACK_PORT.")
        return False

    code = _OAuthCallbackHandler.auth_code
    if not code:
        logging.error("No auth code received")
        return False

    # Exchange code for tokens
    try:
        body = urllib.parse.urlencode({
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }).encode()
        req = urllib.request.Request(TOKEN_URI, data=body, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        store.access_token = data["access_token"]
        store.refresh_token = data.get("refresh_token")
        expires_in = data.get("expires_in", 3600)
        store.expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        store.save()
        logging.info("OAuth flow completed successfully")
        print("\n  ✓ Authentication successful! Credentials saved.\n")
        return True

    except Exception as e:
        logging.error(f"Token exchange failed: {e}")
        return False


# ─── Request Transform (OpenAI → Gemini/Claude) ─────────────────────────────

def transform_openai_to_google(openai_body: dict) -> dict:
    """Transform an OpenAI chat completion request into Google's internal format."""
    model_raw = openai_body.get("model", "claude-opus-4-6")
    model = resolve_model(model_raw)
    messages = openai_body.get("messages", [])

    # Convert messages to Gemini contents format
    contents = []
    system_parts = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Collect system messages separately
        if role == "system":
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append({"text": part.get("text", "")})
                    elif isinstance(part, str):
                        system_parts.append({"text": part})
            else:
                system_parts.append({"text": str(content)})
            continue

        # Map roles
        gemini_role = "model" if role == "assistant" else "user"

        # Build parts
        parts = []
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            try:
                                header, b64data = url.split(",", 1)
                                mime = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
                                parts.append({"inlineData": {"mimeType": mime, "data": b64data}})
                            except Exception:
                                parts.append({"text": "[image]"})
                elif isinstance(part, str):
                    parts.append({"text": part})
        else:
            parts.append({"text": str(content)})

        contents.append({"role": gemini_role, "parts": parts})

    # Build generation config
    gen_config = {}
    if openai_body.get("temperature") is not None:
        gen_config["temperature"] = openai_body["temperature"]
    if openai_body.get("top_p") is not None:
        gen_config["topP"] = openai_body["top_p"]
    if openai_body.get("max_tokens") is not None:
        gen_config["maxOutputTokens"] = openai_body["max_tokens"]
    if openai_body.get("stop"):
        stop = openai_body["stop"]
        gen_config["stopSequences"] = [stop] if isinstance(stop, str) else stop
    if openai_body.get("frequency_penalty") is not None:
        gen_config["frequencyPenalty"] = openai_body["frequency_penalty"]
    if openai_body.get("presence_penalty") is not None:
        gen_config["presencePenalty"] = openai_body["presence_penalty"]

    # Build the inner request
    request_data = {
        "contents": contents,
        "safetySettings": SAFETY_SETTINGS,
        "generationConfig": gen_config,
    }

    # Add system instruction if present
    if system_parts:
        request_data["systemInstruction"] = {"parts": system_parts}

    # Add thinking config for Gemini models only
    if not is_claude(model_raw):
        gen_config["thinkingConfig"] = {
            "thinkingBudget": -1,
            "includeThoughts": True,
        }

    return {
        "model": f"models/{model}",
        "request": request_data,
    }


def transform_google_to_openai(google_resp: dict, model: str, stream: bool = False,
                                 chunk_index: int = 0, resp_id: str = None) -> dict:
    """Transform Google's response into OpenAI chat completion format."""
    resp_id = resp_id or f"chatcmpl-{uuid.uuid4()}"

    if stream:
        # Streaming chunk format
        text = ""
        candidates = google_resp.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part:
                    text += part["text"]

        return {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": text} if text else {},
                "finish_reason": _map_finish_reason(candidates[0].get("finishReason")) if candidates else None,
            }],
        }
    else:
        # Non-streaming format
        text = ""
        finish_reason = "stop"
        candidates = google_resp.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part:
                    text += part["text"]
            finish_reason = _map_finish_reason(candidates[0].get("finishReason", "STOP"))

        usage = google_resp.get("usageMetadata", {})
        return {
            "id": resp_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            },
        }


def _map_finish_reason(reason):
    mapping = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
    }
    return mapping.get(reason, "stop")


# ─── HTTP Server ─────────────────────────────────────────────────────────────

class ProxyHandler(BaseHTTPRequestHandler):
    """Handles OpenAI-compatible API requests."""

    token_store: TokenStore = None

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/health":
            self._json_response(200, {"status": "ok"})
            return

        if path in ("/v1/models", "/v1/models/"):
            if not self._authenticate():
                return
            models = []
            for name in ALL_MODELS:
                models.append({
                    "id": name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "google",
                })
            self._json_response(200, {"object": "list", "data": models})
            return

        self._json_response(404, {"error": {"message": "Not found", "type": "invalid_request_error"}})

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path

        if path != "/v1/chat/completions":
            self._json_response(404, {"error": {"message": "Not found"}})
            return

        if not self._authenticate():
            return

        # Read request body
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length > 0 else {}
        except Exception as e:
            self._json_response(400, {"error": {"message": f"Invalid JSON: {e}"}})
            return

        # Ensure we have a valid token and are onboarded
        store = self.__class__.token_store
        token = store.get_token()
        if not token:
            self._json_response(500, {"error": {"message": "No valid auth token. Restart proxy to re-authenticate."}})
            return

        store.onboard()
        proj = store.discover_project()
        if not proj:
            self._json_response(500, {"error": {"message": "Could not discover project ID."}})
            return

        # Transform request
        google_payload = transform_openai_to_google(body)
        google_payload["project"] = proj

        is_stream = body.get("stream", False)
        model_name = body.get("model", "claude-opus-4-6")
        resolved = resolve_model(model_name)

        logging.info(f"Request: model={model_name} -> {resolved}, stream={is_stream}")

        # Build Google API request
        action = "streamGenerateContent" if is_stream else "generateContent"
        url = f"{CODE_ASSIST_ENDPOINT}/v1internal:{action}"
        if is_stream:
            url += "?alt=sse"

        post_data = json.dumps(google_payload).encode()
        api_req = urllib.request.Request(url, data=post_data, method="POST")
        api_req.add_header("Authorization", f"Bearer {token}")
        api_req.add_header("Content-Type", "application/json")
        api_req.add_header("User-Agent", _user_agent())

        try:
            if is_stream:
                self._handle_stream(api_req, model_name)
            else:
                self._handle_non_stream(api_req, model_name)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            logging.error(f"Google API error {e.code}: {error_body}")
            try:
                err = json.loads(error_body)
                msg = err.get("error", {}).get("message", f"API error {e.code}")
            except Exception:
                msg = f"Google API returned {e.code}"
            self._json_response(e.code, {"error": {"message": msg, "type": "api_error", "code": e.code}})
        except Exception as e:
            logging.error(f"Request failed: {e}")
            self._json_response(500, {"error": {"message": str(e)}})

    def _handle_non_stream(self, api_req, model_name):
        """Handle non-streaming response."""
        with urllib.request.urlopen(api_req, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        # Google wraps response in {"response": {...}}
        if raw.startswith("data: "):
            raw = raw[6:]
        data = json.loads(raw)
        inner = data.get("response", data)

        openai_resp = transform_google_to_openai(inner, model_name, stream=False)
        self._json_response(200, openai_resp)

    def _handle_stream(self, api_req, model_name):
        """Handle streaming SSE response."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        resp_id = f"chatcmpl-{uuid.uuid4()}"
        chunk_idx = 0

        try:
            resp = urllib.request.urlopen(api_req, timeout=300)
            buffer = ""

            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    if not line or not line.startswith("data: "):
                        continue

                    json_str = line[6:].strip()
                    if not json_str:
                        continue

                    try:
                        google_data = json.loads(json_str)
                        inner = google_data.get("response", google_data)

                        openai_chunk = transform_google_to_openai(
                            inner, model_name, stream=True,
                            chunk_index=chunk_idx, resp_id=resp_id
                        )

                        # Only send chunks that have content
                        choices = openai_chunk.get("choices", [])
                        if choices and (choices[0].get("delta", {}).get("content") or choices[0].get("finish_reason")):
                            sse_line = f"data: {json.dumps(openai_chunk)}\n\n"
                            self.wfile.write(sse_line.encode())
                            self.wfile.flush()
                            chunk_idx += 1

                    except json.JSONDecodeError:
                        continue

            # Send [DONE]
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            resp.close()

        except BrokenPipeError:
            logging.info("Client disconnected during stream")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            logging.error(f"Streaming HTTP {e.code}: {error_body}")
            try:
                error_chunk = {
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": f"\n[HTTP {e.code}: {error_body[:500]}]"}, "finish_reason": "stop"}],
                }
                self.wfile.write(f"data: {json.dumps(error_chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
                pass
        except Exception as e:
            logging.error(f"Streaming error: {e}")
            try:
                error_chunk = {
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": f"\n[Error: {e}]"}, "finish_reason": "stop"}],
                }
                self.wfile.write(f"data: {json.dumps(error_chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
                pass

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, x-goog-api-key")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def _authenticate(self) -> bool:
        """Validate the incoming request auth."""
        # Check query param
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if query.get("key", [None])[0] == AUTH_PASSWORD:
            return True

        # Check headers
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and auth[7:] == AUTH_PASSWORD:
            return True

        api_key = self.headers.get("x-goog-api-key", "")
        if api_key == AUTH_PASSWORD:
            return True

        self._json_response(401, {"error": {"message": "Unauthorized", "type": "authentication_error"}})
        return False

    def _json_response(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        logging.info(f"{self.client_address[0]} - {format % args}")


# ─── Main ────────────────────────────────────────────────────────────────────

class ThreadedHTTPServer(HTTPServer):
    """Handle each request in a new thread for concurrent streaming."""
    def process_request(self, request, client_address):
        thread = threading.Thread(target=self.process_request_thread, args=(request, client_address))
        thread.daemon = True
        thread.start()

    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print(r"""
   ╔═══════════════════════════════════════════╗
   ║   Claude Proxy (geminicli2api-lite)       ║
   ║   Zero dependencies • Termux-ready        ║
   ╚═══════════════════════════════════════════╝
    """)

    store = TokenStore()

    # Try to load existing credentials
    if store.load():
        logging.info("Loaded existing credentials")
        if store.is_expired():
            logging.info("Token expired, refreshing...")
            if not store.refresh():
                logging.warning("Refresh failed, starting OAuth flow...")
                if not run_oauth_flow(store):
                    print("Authentication failed. Exiting.")
                    sys.exit(1)
        else:
            logging.info("Token still valid")
    else:
        logging.info("No credentials found, starting OAuth flow...")
        if not run_oauth_flow(store):
            print("Authentication failed. Exiting.")
            sys.exit(1)

    # Discover project and onboard
    logging.info("Discovering project ID...")
    proj = store.discover_project()
    if proj:
        logging.info(f"Project: {proj}")
    else:
        logging.warning("Could not discover project ID — will retry on first request")

    store.onboard()

    # Set token store on handler class
    ProxyHandler.token_store = store

    # Start server
    server = ThreadedHTTPServer((HOST, PORT), ProxyHandler)
    logging.info(f"Proxy running on http://{HOST}:{PORT}")
    logging.info(f"OpenAI endpoint: http://localhost:{PORT}/v1/chat/completions")
    logging.info(f"Models: {', '.join(ALL_MODELS[:5])}...")
    print(f"\n  → Ready! Point SillyTavern to: http://localhost:{PORT}/v1\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
