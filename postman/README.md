# DocuMind Postman Setup & Quickstart

This directory contains the ready-to-import Postman Collection and Environments with automatic JWT authentication and variable management.

---

## Files Included

1. [`DocuMind.postman_collection.json`](./DocuMind.postman_collection.json): All API endpoints (`Health`, `Auth`, `Documents`, `Chat / RAG`).
2. [`DocuMind_Dev.postman_environment.json`](./DocuMind_Dev.postman_environment.json): Dev environment (`http://127.0.0.1:8000`).
3. [`DocuMind_Prod.postman_environment.json`](./DocuMind_Prod.postman_environment.json): Prod environment placeholder (`https://your-production-app.fly.dev`).

---

## How It Works (Zero Copy-Pasting)

- **Collection-Level Bearer Token:** The entire collection automatically passes `Authorization: Bearer {{access_token}}` on all protected endpoints (`/auth/me`, `/documents/*`, `/chat/*`).
- **Auto-Token Capture:**
  - Running **`Register`** or **`Login`** automatically saves `access_token` and `refresh_token` to your active Postman environment via test scripts.
  - Running **`Refresh Token`** automatically updates `access_token` and `refresh_token` in your active environment.
- **Auto-ID Capture:**
  - Running **`Upload Document`** automatically sets `{{document_id}}`.
  - Running **`Ask Question`** automatically sets `{{conversation_id}}`.

---

## Import Instructions

1. Open Postman.
2. Click **Import** (top left).
3. Select or drag-and-drop the 3 files from this folder:
   - `DocuMind.postman_collection.json`
   - `DocuMind_Dev.postman_environment.json`
   - `DocuMind_Prod.postman_environment.json`
4. In the top-right environment selector in Postman, choose **`DocuMind - Dev`** (or **`DocuMind - Prod`**).

---

## Quick Testing Flow

1. **Start the local server:**
   ```powershell
   python -m uvicorn app.main:app --reload
   ```
2. In Postman (with `DocuMind - Dev` selected):
   - Run **`Auth > Register`** (or **`Auth > Login`**).
   - Run **`Auth > Get Current User Profile (/me)`** to verify access without touching any token headers.
   - Run **`Documents > Upload Document`** (attach a `.pdf`, `.md`, or `.txt` file under Body -> form-data -> file).
   - Run **`Documents > Get Document Ingestion Status`** until `status` is `ready`.
   - Run **`Chat (RAG) > Ask Question (New Conversation)`** to receive a grounded answer with citations.
