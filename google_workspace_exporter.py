from __future__ import annotations

import base64
import datetime
import json
import os
import re
import time
from dataclasses import asdict
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

import httpx

from monetary_cost_manager import OpenRouterUsageRecord


TOKEN_URL = "https://oauth2.googleapis.com/token"
DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
DOCS_DOCUMENTS_URL = "https://docs.googleapis.com/v1/documents"
SHEETS_SPREADSHEETS_URL = "https://sheets.googleapis.com/v4/spreadsheets"
SCOPES = (
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/spreadsheets",
)
DEFAULT_SHEETS_TAB = "Money Manager"
SHEETS_HEADERS = [
    "exported_at_utc",
    "github_run_id",
    "github_run_attempt",
    "github_workflow",
    "question_id",
    "post_id",
    "question_type",
    "title",
    "metaculus_post_url",
    "google_doc_url",
    "task_no",
    "task_name",
    "model_used",
    "input_characters",
    "input_tokens",
    "output_characters",
    "output_tokens",
    "total_tokens",
]


def google_export_is_configured() -> bool:
    try:
        service_account = _load_service_account_info(required=False)
    except RuntimeError:
        return True
    return bool(service_account and _spreadsheet_id() and _drive_folder_id())


def google_export_is_partially_configured() -> bool:
    try:
        service_account = _load_service_account_info(required=False)
    except RuntimeError:
        service_account = True
    return bool(service_account or _spreadsheet_id() or _drive_folder_id())


def export_question_result_to_google(
    *,
    question_id: int,
    post_id: int,
    title: str,
    question_type: str,
    llm_result_text: str,
    usage_records: list[OpenRouterUsageRecord],
    metaculus_post_url: str,
) -> str | None:
    """Upload one question's full LLM details to Docs and usage rows to Sheets."""
    if not google_export_is_configured():
        message = (
            "Google Workspace export skipped: set GOOGLE_SERVICE_ACCOUNT_JSON "
            "or GOOGLE_SERVICE_ACCOUNT_JSON_B64, GOOGLE_SHEETS_SPREADSHEET_ID, "
            "and GOOGLE_DRIVE_FOLDER_ID to enable it."
        )
        if google_export_is_partially_configured():
            _handle_export_problem(RuntimeError(message))
        else:
            print(message)
        return None

    try:
        token = _get_access_token()
        _verify_drive_folder_access(
            token=token,
            folder_id=_drive_folder_id(required=True),
        )
        doc_url = _create_google_doc(
            token=token,
            folder_id=_drive_folder_id(required=True),
            question_id=question_id,
            post_id=post_id,
            title=title,
            question_type=question_type,
            content=llm_result_text,
        )
        _append_usage_rows(
            token=token,
            spreadsheet_id=_spreadsheet_id(required=True),
            sheet_title=os.getenv("GOOGLE_SHEETS_TAB", DEFAULT_SHEETS_TAB).strip()
            or DEFAULT_SHEETS_TAB,
            rows=_usage_rows(
                question_id=question_id,
                post_id=post_id,
                title=title,
                question_type=question_type,
                metaculus_post_url=metaculus_post_url,
                google_doc_url=doc_url,
                usage_records=usage_records,
            ),
        )
    except Exception as exc:
        _handle_export_problem(exc)
        return None
    print(f"  [Google export saved] {doc_url}")
    return doc_url


def _handle_export_problem(exc: Exception) -> None:
    if _strict_export():
        raise exc
    print(f"Google Workspace export skipped/failed: {exc}")


def _strict_export() -> bool:
    return os.getenv("GOOGLE_EXPORT_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _load_service_account_info(*, required: bool = True) -> dict[str, Any] | None:
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    raw_b64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64", "").strip()
    if not raw and raw_b64:
        raw = base64.b64decode(raw_b64).decode("utf-8")
    if not raw:
        if required:
            raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is not configured.")
        return None

    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON.") from exc

    private_key = str(info.get("private_key", "")).replace("\\n", "\n")
    client_email = str(info.get("client_email", ""))
    if not private_key or not client_email:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON must include private_key and client_email."
        )
    info["private_key"] = private_key
    return info


def _get_access_token() -> str:
    service_account = _load_service_account_info(required=True)
    assert service_account is not None
    now = int(time.time())
    assertion = _sign_jwt(
        service_account["private_key"],
        {
            "iss": service_account["client_email"],
            "scope": " ".join(SCOPES),
            "aud": TOKEN_URL,
            "iat": now,
            "exp": now + 3600,
        },
    )
    response = httpx.post(
        TOKEN_URL,
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": assertion,
        },
        timeout=30.0,
    )
    _raise_for_status(response, "fetch Google OAuth access token")
    payload = response.json()
    access_token = payload.get("access_token")
    if not access_token:
        raise RuntimeError("Google OAuth token response did not include access_token.")
    return str(access_token)


def _verify_drive_folder_access(*, token: str, folder_id: str) -> None:
    response = httpx.get(
        f"{DRIVE_FILES_URL}/{folder_id}",
        headers=_auth_headers(token),
        params={
            "fields": "id,name,mimeType,driveId,capabilities(canAddChildren,canEdit,canShare)",
            "supportsAllDrives": "true",
        },
        timeout=30.0,
    )
    _raise_for_status(response, "verify Google Drive folder access")
    folder = response.json()
    if folder.get("mimeType") != "application/vnd.google-apps.folder":
        raise RuntimeError(
            "Google Workspace export failed: GOOGLE_DRIVE_FOLDER_ID does not point "
            f"to a Drive folder. Got mimeType={folder.get('mimeType')!r}."
        )
    capabilities = folder.get("capabilities") or {}
    if not capabilities.get("canAddChildren"):
        raise RuntimeError(
            "Google Workspace export failed: the service account can see the Drive "
            f"folder {folder.get('name')!r}, but Google says it cannot create files "
            f"inside it. Folder capabilities={capabilities!r}. Share the folder "
            "directly with the service account as Editor, or use a folder owned by "
            "the same account/domain."
        )


def _sign_jwt(private_key_pem: str, claims: dict[str, Any]) -> str:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    header = {"alg": "RS256", "typ": "JWT"}
    signing_input = ".".join(
        [
            _base64url_json(header),
            _base64url_json(claims),
        ]
    ).encode("ascii")
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode("utf-8"),
        password=None,
    )
    signature = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{signing_input.decode('ascii')}.{_base64url(signature)}"


def _create_google_doc(
    *,
    token: str,
    folder_id: str,
    question_id: int,
    post_id: int,
    title: str,
    question_type: str,
    content: str,
) -> str:
    doc_name = _doc_name(
        question_id=question_id,
        post_id=post_id,
        question_type=question_type,
        title=title,
    )
    headers = _auth_headers(token)
    create_response = httpx.post(
        DRIVE_FILES_URL,
        headers=headers,
        json={
            "name": doc_name,
            "mimeType": "application/vnd.google-apps.document",
            "parents": [folder_id],
        },
        params={"fields": "id,webViewLink", "supportsAllDrives": "true"},
        timeout=30.0,
    )
    _raise_for_status(create_response, "create Google Doc in Drive folder")
    file_info = create_response.json()
    document_id = file_info["id"]

    _insert_doc_text(token, document_id, _sanitize_document_text(content))
    return file_info.get("webViewLink") or f"https://docs.google.com/document/d/{document_id}/edit"


def _insert_doc_text(token: str, document_id: str, content: str) -> None:
    if not content:
        return
    chunk_size = 90_000
    chunks = [
        content[index : index + chunk_size]
        for index in range(0, len(content), chunk_size)
    ]
    for chunk in reversed(chunks):
        response = httpx.post(
            f"{DOCS_DOCUMENTS_URL}/{document_id}:batchUpdate",
            headers=_auth_headers(token),
            json={
                "requests": [
                    {
                        "insertText": {
                            "location": {"index": 1},
                            "text": chunk,
                        }
                    }
                ]
            },
            timeout=30.0,
        )
        _raise_for_status(response, "insert text into Google Doc")


def _append_usage_rows(
    *,
    token: str,
    spreadsheet_id: str,
    sheet_title: str,
    rows: list[list[Any]],
) -> None:
    _ensure_sheet(token, spreadsheet_id, sheet_title)
    _ensure_header_row(token, spreadsheet_id, sheet_title)
    range_name = f"{_quote_sheet_title(sheet_title)}!A:R"
    response = httpx.post(
        (
            f"{SHEETS_SPREADSHEETS_URL}/{spreadsheet_id}/values/"
            f"{quote(range_name, safe='')}:append"
        ),
        headers=_auth_headers(token),
        params={
            "valueInputOption": "USER_ENTERED",
            "insertDataOption": "INSERT_ROWS",
        },
        json={"values": rows},
        timeout=30.0,
    )
    _raise_for_status(response, "append money manager rows to Google Sheet")


def _ensure_sheet(token: str, spreadsheet_id: str, sheet_title: str) -> None:
    response = httpx.get(
        f"{SHEETS_SPREADSHEETS_URL}/{spreadsheet_id}",
        headers=_auth_headers(token),
        params={"fields": "sheets.properties.title"},
        timeout=30.0,
    )
    _raise_for_status(response, "read Google Sheet metadata")
    existing_titles = {
        sheet.get("properties", {}).get("title")
        for sheet in response.json().get("sheets", [])
    }
    if sheet_title in existing_titles:
        return

    create_response = httpx.post(
        f"{SHEETS_SPREADSHEETS_URL}/{spreadsheet_id}:batchUpdate",
        headers=_auth_headers(token),
        json={
            "requests": [
                {
                    "addSheet": {
                        "properties": {
                            "title": sheet_title,
                        }
                    }
                }
            ]
        },
        timeout=30.0,
    )
    _raise_for_status(create_response, "create Google Sheet tab")


def _ensure_header_row(token: str, spreadsheet_id: str, sheet_title: str) -> None:
    header_range = f"{_quote_sheet_title(sheet_title)}!A1:R1"
    get_response = httpx.get(
        (
            f"{SHEETS_SPREADSHEETS_URL}/{spreadsheet_id}/values/"
            f"{quote(header_range, safe='')}"
        ),
        headers=_auth_headers(token),
        timeout=30.0,
    )
    _raise_for_status(get_response, "read Google Sheet header row")
    if get_response.json().get("values"):
        return

    update_response = httpx.put(
        (
            f"{SHEETS_SPREADSHEETS_URL}/{spreadsheet_id}/values/"
            f"{quote(header_range, safe='')}"
        ),
        headers=_auth_headers(token),
        params={"valueInputOption": "USER_ENTERED"},
        json={"values": [SHEETS_HEADERS]},
        timeout=30.0,
    )
    _raise_for_status(update_response, "write Google Sheet header row")


def _usage_rows(
    *,
    question_id: int,
    post_id: int,
    title: str,
    question_type: str,
    metaculus_post_url: str,
    google_doc_url: str,
    usage_records: list[OpenRouterUsageRecord],
) -> list[list[Any]]:
    exported_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    records = usage_records or [
        OpenRouterUsageRecord(
            no=0,
            name_of_task="no OpenRouter calls recorded",
            input_characters=0,
            input_tokens=0,
            output_characters=0,
            output_tokens=0,
            model_used="",
        )
    ]
    rows: list[list[Any]] = []
    for record in records:
        record_dict = asdict(record)
        rows.append(
            [
                exported_at,
                os.getenv("GITHUB_RUN_ID", ""),
                os.getenv("GITHUB_RUN_ATTEMPT", ""),
                os.getenv("GITHUB_WORKFLOW", ""),
                question_id,
                post_id,
                question_type,
                title,
                metaculus_post_url,
                google_doc_url,
                record_dict["no"],
                record_dict["name_of_task"],
                record_dict["model_used"],
                record_dict["input_characters"],
                record_dict["input_tokens"],
                record_dict["output_characters"],
                record_dict["output_tokens"],
                record_dict["input_tokens"] + record_dict["output_tokens"],
            ]
        )
    return rows


def _spreadsheet_id(*, required: bool = False) -> str:
    return _id_from_env(
        "GOOGLE_SHEETS_SPREADSHEET_ID",
        patterns=(r"/spreadsheets/d/([^/?#]+)",),
        required=required,
    )


def _drive_folder_id(*, required: bool = False) -> str:
    return _id_from_env(
        "GOOGLE_DRIVE_FOLDER_ID",
        patterns=(r"/folders/([^/?#]+)",),
        required=required,
    )


def _id_from_env(name: str, *, patterns: tuple[str, ...], required: bool) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        if required:
            raise RuntimeError(f"{name} is not configured.")
        return ""
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    parsed = urlparse(value)
    query_id = parse_qs(parsed.query).get("id", [""])[0]
    return query_id or value


def _doc_name(
    *,
    question_id: int,
    post_id: int,
    question_type: str,
    title: str,
) -> str:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H%M UTC")
    run_id = os.getenv("GITHUB_RUN_ID", "local")
    attempt = os.getenv("GITHUB_RUN_ATTEMPT", "1")
    safe_title = re.sub(r"\s+", " ", re.sub(r"[^\w\s().,-]", "", title)).strip()
    safe_title = safe_title[:90].strip() or "Untitled question"
    return (
        f"{timestamp} - Q{question_id} P{post_id} - {question_type} - "
        f"{safe_title} - run {run_id}.{attempt}"
    )


def _sanitize_document_text(content: str) -> str:
    return "".join(
        char
        for char in content
        if char in "\n\r\t" or ord(char) >= 32
    )


def _quote_sheet_title(sheet_title: str) -> str:
    return f"'{sheet_title.replace(chr(39), chr(39) + chr(39))}'"


def _auth_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _raise_for_status(response: httpx.Response, action: str) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text.strip()
        if len(detail) > 1200:
            detail = detail[:1200] + "..."
        raise RuntimeError(
            f"Google Workspace export failed while trying to {action}. "
            f"Status={response.status_code}. Response={detail}"
        ) from exc


def _base64url_json(value: dict[str, Any]) -> str:
    return _base64url(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")
