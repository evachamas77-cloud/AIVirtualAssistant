    # AI Visual Assistant

    The local FastAPI backend analyzes a still image for sharpness, common objects, and visible text.

    ## Setup

    From the project root in PowerShell, use the existing Python 3.11 virtual environment:

    ```powershell
    .\.venv\Scripts\python.exe -m pip install --timeout 180 --retries 4 -r requirements.txt
    New-Item -ItemType Directory -Force models/tessdata
    Invoke-WebRequest -Uri https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/eng.traineddata -OutFile models/tessdata/eng.traineddata
    ```

    On 64-bit Windows the [tesserocr wheel](https://pypi.org/project/tesserocr/2.13.0/) includes the Tesseract runtime. The English language file above is a separate download from [Tesseract's fast models](https://github.com/tesseract-ocr/tessdata_fast).

    On the first valid upload, Ultralytics looks for `yolo11n.pt` in the project root and may download it if it is absent. For offline use, place [official YOLO11n weights](https://docs.ultralytics.com/models/yolo11/) there beforehand or set `YOLO_MODEL_PATH` to the local file. Use only trusted weights. Both engines are initialized lazily and reused within each server process after a successful load. Images are analyzed locally; no paid API is required.

    PyTorch is installed with Ultralytics and its wheel can take time to download. If an upload reports `analysis_errors`, check the full traceback in the Uvicorn terminal and verify the packages in the same environment used to launch the server:

    ```powershell
    .\.venv\Scripts\python.exe -m pip show ultralytics torch torchvision tesserocr
    .\.venv\Scripts\python.exe -m pip check
    ```

    Restart Uvicorn after installing packages. An HTTP 200 response alone does not mean the inference engines succeeded; check that `analysis_errors` is empty.

    ## Run

    ```powershell
    .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
    ```

    Open [Swagger UI](http://127.0.0.1:8000/docs), or run the following in another PowerShell window with a JPEG or PNG you have on disk:

    ```powershell
    curl.exe http://127.0.0.1:8000/health
    curl.exe -F "file=@C:/path/to/photo.jpg;type=image/jpeg" http://127.0.0.1:8000/analyze-frame
    ```

    For a PNG, use `type=image/png`. The upload limit is 10 MiB. Invalid, empty, and oversized uploads keep their Phase 1 error responses.

    ## Read Results

    - `width`, `height`, `sharpness_score`, and `quality` are unchanged. `quality` is a Laplacian-based blur estimate, not a measure of inference reliability.
    - `objects` contains a `name`, `confidence`, and `bbox` per detection. `ocr_results` contains `text`, `confidence`, and `bbox` per recognized word. Both box formats are `[left, top, right, bottom]` in original-image pixels, measured from the upper-left corner.
    - `ocr_text` joins retained words in the engine's reading order, with newlines between detected lines. An empty result with no corresponding error means the engine found nothing that passed its threshold.
    - `processing_time_ms` measures decode, Phase 1 analysis, and both inference stages, including any model initialization or download on that request. It excludes upload and response transmission.
    - `analysis_errors` is empty on full success. If an engine fails, the endpoint still returns HTTP 200 with the Phase 1 fields and any results from the other engine. The failing fields are empty, and `analysis_errors` contains `object_detection` and/or `ocr`. Check the server logs for the cause, such as a missing model or blocked download.

    Confidence is a model-specific score from 0 to 1, not a calibrated probability. YOLO11n only detects classes it was trained on and can miss or misidentify objects. OCR accuracy and reading order can degrade with blur, glare, small or stylized type, rotations, or complex layouts. An empty detection list does not establish that an environment is safe.

    ## Configuration

    Set environment variables in PowerShell before starting Uvicorn, then restart it when changing settings:

    | Variable | Default | Purpose |
    | --- | --- | --- |
    | `YOLO_MODEL_PATH` | `yolo11n.pt` in the project root | Pretrained weights name or local path; use an absolute path to override it from any working directory. Inference runs on CPU. |
    | `YOLO_CONFIDENCE_THRESHOLD` | `0.25` | Minimum object confidence (0 to 1). |
    | `OCR_TESSDATA_DIR` | `models/tessdata` | Directory containing OCR language files. |
    | `OCR_LANGUAGES` | `eng` | Tesseract language codes separated by `+`. |
    | `OCR_CONFIDENCE_THRESHOLD` | `0` | Minimum OCR word confidence (0 to 1). |

    To add French or Arabic, download `fra.traineddata` or `ara.traineddata` from [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast) into the same data directory. For example, set `$env:OCR_LANGUAGES = "eng+fra"` for English and French, or `$env:OCR_LANGUAGES = "ara"` for Arabic. Validate mixed-language reading order on representative images.

    ## Frame Archive

    Phase 3 uses Python's built-in SQLite support; no new packages are required. The database is `frames.sqlite3` inside the storage directory. Accepted frames keep their original JPEG or PNG bytes in `images/` under that directory. An `image_path` in a response is relative to the storage directory, not a public image URL. `POST /analyze-frame` still analyzes without saving.

    | Endpoint | Result |
    | --- | --- |
    | `POST /frames` | Submit a JPEG or PNG as multipart field `file`. Returns HTTP 201 and `saved: true` with metadata, or HTTP 200 and `saved: false` with a skip reason. Both include the Phase 2 output under `analysis`. |
    | `GET /frames` | Paginated metadata, newest first. Optional inclusive `start` and `end` accept timestamps with a time zone, e.g. `2026-03-01T10:00:00Z`. |
    | `GET /frames/search?q=...` | Case-insensitive **keyword search**, not semantic search. Matches a literal substring of OCR text or an object name. Supports the same dates and pagination. |
    | `GET /frames/{id}` | Metadata including server-assigned UTC timestamp, dimensions, sharpness, OCR text and word details, and detected objects with confidence and bounding boxes. |
    | `DELETE /frames/{id}` | Deletes the record and image; returns HTTP 204, or 404 for an unknown ID. |

    List and search return `items`, `limit`, `offset`, and `next_offset` (null on the last page). Use `limit` from 1 to 100 (default 20) and a nonnegative `offset` (default 0).

    Skipped frames never create a file or database row. Reasons are `analysis_failed` (see `analysis.analysis_errors`), `too_blurry`, `duplicate`, and `min_interval`. Duplicate detection compares exact decoded pixels against all retained frames, so changing PNG compression alone does not bypass it. It does not match near-identical images with changed pixels.

    Set these environment variables before starting the server:

    | Variable | Default | Purpose |
    | --- | --- | --- |
    | `FRAME_STORAGE_DIR` | `data/frames` in the project root | Local directory for the database and images. |
    | `FRAME_MIN_SHARPNESS` | `100` | Minimum sharpness score to archive. |
    | `FRAME_MIN_INTERVAL_SECONDS` | `5` | Minimum seconds since the most recently archived frame still in storage; 0 disables the interval filter. |
    | `FRAME_RETENTION_DAYS` | `7` | Automatically delete frames and images older than this many days; 0 disables cleanup. |
    | `FRAME_CLEANUP_INTERVAL_SECONDS` | `3600` | Seconds between cleanup runs. Cleanup also runs at startup when enabled. |

    ### PowerShell Walkthrough

    From the project root, start a separate demo server with a fresh archive:

    ```powershell
    $env:FRAME_STORAGE_DIR = Join-Path $env:TEMP ('ai-visual-frames-' + [guid]::NewGuid().ToString('N'))
    .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
    ```

    In another PowerShell window, run the following from the project root. Substitute an available JPEG with clear visible text if the smoke-test image is not present.

    ```powershell
    $base = 'http://127.0.0.1:8001'
    $file = 'file=@models/phase2-smoke-sign.jpg;type=image/jpeg'

    # Save, then submit the same pixels again to verify the duplicate filter
    $posted = curl.exe -fsS -F $file "$base/frames" | ConvertFrom-Json
    if (-not $posted.saved) { throw "Frame not saved: $($posted.reason). Check the server logs for analysis errors." }
    $posted.frame | ConvertTo-Json -Depth 8
    curl.exe -fsS -F $file "$base/frames"

    # List, filter, and search
    $start = [uri]::EscapeDataString([DateTimeOffset]::UtcNow.AddHours(-1).ToString('o'))
    curl.exe -fsS "$base/frames?limit=10&offset=0"
    curl.exe -fsS "$base/frames?start=$start&limit=10"
    curl.exe -fsS "$base/frames/search?q=HELLO&limit=10"

    # Retrieve and delete the saved frame
    $id = $posted.frame.id
    curl.exe -fsS "$base/frames/$id"
    curl.exe -sS -X DELETE -o NUL -w '%{http_code}' "$base/frames/$id"
    ```

## Live Camera and Dashboard

Phase 4 uses [aiortc](https://aiortc.readthedocs.io/en/latest/api.html) for WebRTC. The phone sends video to FastAPI, and FastAPI relays it over WebRTC to the dashboard. Same-origin HTTP requests exchange complete SDP offers and answers; they do not carry video. The WebRTC stream carries no audio and uses no cloud service, public STUN server, or TURN relay. Phase 5 assistant requests and speech controls are separate. Use one Uvicorn worker: the camera slot, pairing sessions, and viewers are in process memory.

From the project root, install only the new WebRTC dependency if it is not already present:

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: --timeout 90 --retries 1 "aiortc>=1.14,<2"
.\.venv\Scripts\python.exe -c "import aiortc, av; print('WebRTC runtime ready')"
```

This does not reinstall PyTorch, YOLO, or OCR. The PyAV wheel is the largest additional download. If installation fails, the existing Phase 1-3 APIs continue to work without live streaming; check the pip error and the Uvicorn terminal.

### Trusted HTTPS on Windows

[Camera access requires a secure context](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia). HTTP localhost is suitable for a laptop-only check, but HTTP on the laptop's LAN IP is not suitable for the phone. Use a private network and [mkcert](https://github.com/FiloSottile/mkcert#mobile-devices) for a development certificate.

```powershell
# Skip installation if mkcert is already on PATH. Open a new terminal afterward if needed.
winget install --exact --id FiloSottile.mkcert

Set-Location 'C:\Users\khate\Documents\AIVisualAssistant'
Get-NetIPConfiguration | Where-Object IPv4DefaultGateway | Format-Table InterfaceAlias,IPv4Address
$lan = Read-Host 'Enter the laptop IPv4 address on the same Wi-Fi as the phone'
New-Item -ItemType Directory -Force .\certs, .\certs\phone-root | Out-Null
mkcert -install
mkcert -cert-file .\certs\lan.pem -key-file .\certs\lan-key.pem localhost 127.0.0.1 $lan
$ca = (mkcert -CAROOT).Trim()
certutil.exe -f -decode (Join-Path $ca 'rootCA.pem') .\certs\phone-root\visual-assistant-root.cer

$env:LIVE_SESSION_CODE = .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(24))"
Write-Host "Pairing code: $env:LIVE_SESSION_CODE"
Write-Host "Laptop dashboard: https://localhost:8443/"
Write-Host "Phone camera: https://${lan}:8443/phone"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8443 --workers 1 --no-proxy-headers --ssl-certfile .\certs\lan.pem --ssl-keyfile .\certs\lan-key.pem
```

Keep the pairing code and private keys private. Transfer **only** `certs\phone-root\visual-assistant-root.cer` to your own phone via a trusted file transfer. On iPhone, install the downloaded profile in Settings, then enable full trust in **Settings > General > About > Certificate Trust Settings**. On Android, install it as a **CA certificate** from the device's security settings; menu names vary. Confirm the browser trusts the HTTPS page without a certificate warning before entering the code. Remove this development CA from the phone after the demo. If the laptop IP changes, reissue the leaf certificate for the new IP.

The phone must reach laptop TCP 8443 for signaling and the Python process's local ICE UDP ports for media. If Windows Firewall blocks it, allow the Python process on the **Private** network only. For example, in an elevated PowerShell from the project root:

```powershell
$python = .\.venv\Scripts\python.exe -c "import sys; print(sys._base_executable)"
New-NetFirewallRule -DisplayName 'Visual Assistant HTTPS demo' -Direction Inbound -Action Allow -Program $python -Protocol TCP -LocalPort 8443 -RemoteAddress LocalSubnet -Profile Private
New-NetFirewallRule -DisplayName 'Visual Assistant WebRTC demo' -Direction Inbound -Action Allow -Program $python -Protocol UDP -RemoteAddress LocalSubnet -Profile Private
```

Remove the rules after testing with `Remove-NetFirewallRule -DisplayName 'Visual Assistant HTTPS demo','Visual Assistant WebRTC demo'`. Guest Wi-Fi isolation, VPN routing, blocked UDP, or blocked local mDNS can prevent ICE even when the web page loads. This host-candidate setup is for the same reachable LAN, not cross-network streaming; that would require separate ICE/STUN/TURN design and configuration.

### Phone Walkthrough

1. Open `https://localhost:8443/` on the laptop and enter the pairing code. Open `https://<LAPTOP-IP>:8443/phone` in Android Chrome or iPhone Safari and enter the same code.
2. On the phone, tap **Start Camera**, grant permission, and verify the local preview. Tap **Start Streaming** and keep the page foregrounded. The laptop dashboard should show the incoming video and device state.
3. Aim at a well-lit, detailed scene with large text. The dashboard shows sampled versus saved counts, the latest filter result, and a timeline of archived thumbnails. Select a frame to see its sharpness, recognized text, and objects.
4. Tap **Stop Streaming**. The phone preview and camera should turn off, and the dashboard should return to idle. The saved frames remain available.
5. Optional installation: choose **Install app** or **Add to Home screen** in Android Chrome; on iPhone Safari use **Share > Add to Home Screen**. Camera and background behavior vary by browser and OS. The PWA can cache its interface, but a live server connection is required for pairing and streaming.

To confirm the archive on the laptop after viewing it in the dashboard, query the existing local API or SQLite directly. The database path below assumes the default Phase 3 storage directory:

```powershell
curl.exe -fsS "https://localhost:8443/frames?limit=5"
.\.venv\Scripts\python.exe -c "import sqlite3; db=sqlite3.connect('file:data/frames/frames.sqlite3?mode=ro', uri=True); print(db.execute('SELECT id, timestamp, image_path, sharpness_score FROM frames ORDER BY id DESC LIMIT 5').fetchall()); db.close()"
```

Set `LIVE_SAMPLE_SECONDS` before launch to change the analysis interval (default 2 seconds, minimum 0.2). At most one video sample is in inference at a time; excess frames are dropped from analysis, not queued. The dashboard relay is independent of inference. An in-flight sample may finish after stopping. Live samples are bounded to a 1280-pixel long edge and use the Phase 3 sharpness, deduplication, interval, and retention settings. A clear preview does not guarantee the frame will be archived. In particular, inference errors or repeated pixels are rejected by the archive.

The local prototype allows one camera and up to three dashboard viewers. Pairing uses an eight-hour session cookie; there is no account system. When live streaming is configured, non-loopback access to `/frames` and `/analyze-frame` requires the HTTPS pairing session, while local tools keep their existing API access. Do not expose this service to the public internet.

## Phase 5: Ask AI and Semantic Search

Both the laptop dashboard and phone page now include **Visual assistant**. They retain camera pairing, streaming, and archive behavior. The vision endpoint sends images to OpenAI only after an explicit question and per-question consent. **Semantic search is local and does not use OpenAI.** The existing keyword search remains at `/frames/search`.

### Install And Index

From the workspace root, use the existing Python 3.11 environment. Do not reinstall the earlier inference packages:

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: --timeout 90 "openai>=2,<3"
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: --timeout 90 -r requirements-semantic.txt
.\.venv\Scripts\python.exe -m backend.download_semantic
.\.venv\Scripts\python.exe -m backend.download_semantic --check .\models\phase2-smoke-bus.jpg
```

Use another local JPEG/PNG for `--check` if the bus image is unavailable. The SDK is for image-based questions; ONNX Runtime and the tokenizer are optional if you only need current-frame questions. The downloader fetches only the pinned vision encoder, text encoder, and tokenizer from the [SigLIP 2 ONNX artifact repository](https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/tree/ba1f3b0843f24bc5417d38e19c37b287d719b2f4), about **412 MB** total. It verifies the ONNX checksums and never sends any project image. A connection failure leaves a `.part` file for a later retry. A corrupt complete artifact may need to be removed and downloaded again. Normal startup never downloads models.

Google's [multilingual SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224) encodes English/French text and JPEG/PNG pixels into the same space. The local application L2-normalizes the vectors and ranks saved images by cosine similarity. Scores are not confidence or proof of a match. Two quantized CPU encoders load lazily; plan for roughly 1-2 GB of additional working RAM, varying by machine and runtime. A single background worker indexes new and preexisting frames in small steps with two CPU inference threads. Index entries live in a new SQLite table linked to the original frame IDs and disappear with frame deletion or retention. Missing images are skipped; **Index archive** retries unreadable entries after you restore them. Set `SEMANTIC_ENABLED=0` to pause local embeddings or `SEMANTIC_MODEL_DIR` to change the artifact directory.

### Configure The Server

Use the [existing HTTPS certificate setup above](#trusted-https-on-windows). Stop the previous server before starting this one. In PowerShell, enter your own key at the masked prompt; do not paste it into JavaScript, a command argument, or the README.

```powershell
Set-Location 'C:\Users\khate\Documents\AIVisualAssistant'
$secret = Read-Host 'OpenAI API key' -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
try {
    $env:OPENAI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    Remove-Variable secret,keyPointer
}
$env:AI_VISION_MODEL = 'gpt-4o-mini'
$env:AI_BUDGET_USD = '0.25'
$env:LIVE_SESSION_CODE = .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(24))"
Write-Host "Pairing code: $env:LIVE_SESSION_CODE"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8443 --workers 1 --no-proxy-headers --no-access-log --ssl-certfile .\certs\lan.pem --ssl-keyfile .\certs\lan-key.pem
Remove-Item Env:OPENAI_API_KEY
```

`.env.example` documents the variables; it is not loaded automatically. Do not commit a real `.env`, archived data, or certificate keys. `gpt-4o-mini` and its pinned snapshot use [Chat Completions with image input](https://developers.openai.com/api/docs/guides/images-vision?api-mode=chat). [OpenAI lists the model as deprecated](https://developers.openai.com/api/docs/models/gpt-4o-mini); access is project-dependent. To use a different vision-capable model on the Responses API, set `AI_VISION_MODEL`, `AI_INPUT_USD_PER_MILLION`, `AI_OUTPUT_USD_PER_MILLION`, and `AI_ESTIMATED_IMAGE_TOKENS` using that model's published pricing and low-detail image costs.

### Android And Laptop Test Plan

1. Open `https://localhost:8443/` on the laptop and `https://<LAPTOP-IP>:8443/phone` on Android Chrome. Pair both using the code printed in the server terminal. Start the phone camera and stream, and verify the dashboard still shows live video and archived images.
2. On either device, select **Ask AI**, choose **Current camera**, type “What can you see in front of me?”, check the image-sharing consent box, and press **Ask**. The answer should show the live snapshot and its capture time. If the live camera has stopped, the result explicitly says it used the latest archived frame instead.
3. Open **Semantic search**, allow backfill to finish, and search for a concrete scene phrase such as “a television in a living room” or “une télévision dans le salon”. Check the ranked thumbnails and dates, then try restricting the date range. **Ask about frame** selects exactly that archived image for your next consented question.
4. Ask “When did I last see a couch?” with **Auto select**, or explicitly choose **Archive search**. Historical answers use only retrieved evidence; a match is not proof that it was the last occurrence anywhere in the archive.
5. Select **Français**, press the microphone button, and say “Qu'est-ce que tu vois devant moi ?”. On-device recognition is preferred when an installed language pack is available. Otherwise explicitly allow browser-managed speech or type the question. Review the transcript, press **Ask** with fresh image consent, then try **Speak**, **Replay**, and **Stop speaking**. Repeat in English. Test denied microphone permission and the typed fallback.

Speech recognition is [browser-dependent](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition). Some Android/Chrome combinations use a remote browser speech service; the separate speech checkbox discloses and enables that fallback. For playback the app prefers local matching voices. The app never records or archives raw audio, and WebRTC stays video-only.

### API And Cost Checks

From a second PowerShell in the workspace, the following commands pair and test the local index without spending API budget. Only the final, commented line would make a paid request.

```powershell
$base = 'https://localhost:8443'
$pair = @{ code = (Read-Host 'Pairing code from server terminal') } | ConvertTo-Json
Invoke-RestMethod "$base/live/session" -Method Post -ContentType 'application/json' -Body $pair -SessionVariable paired
Invoke-RestMethod "$base/ai/index" -WebSession $paired
Invoke-RestMethod "$base/ai/index/backfill" -Method Post -WebSession $paired
$q = [uri]::EscapeDataString('a television in a living room')
Invoke-RestMethod "$base/ai/semantic-search?q=$q&limit=5" -WebSession $paired
$question = @{ question = 'Describe this room.'; scope = 'current'; language = 'en'; consent = $true } | ConvertTo-Json
# Invoke-RestMethod "$base/ai/ask" -Method Post -ContentType 'application/json' -Body $question -WebSession $paired
```

Endpoints require the same HTTPS pairing session as live video. `/ai/semantic-search` accepts inclusive `start` and `end` as Beirut civil dates (`YYYY-MM-DD`) or ISO timestamps with an explicit offset. Pass an explicit `Z` timestamp for callers that need UTC boundaries. `POST /ai/ask` accepts `auto`, `current`, `history`, or `frame` scope; `frame` requires an integer `frame_id`. Optional dates apply to history or auto scope. Live evidence previews are session-bound, held in memory for up to five minutes, and not added to the archive by asking.

OpenAI receives the question, selected resized JPEGs, and their capture times. The app requests `store=false`; that is not a promise of zero provider retention. No model request is triggered by streaming, indexing, or recording speech. By default at most three JPEGs (768-pixel longest side, `detail=low`) are submitted with a 300-token output limit and a 25-second timeout. Only one question may be in flight and paid calls are not automatically retried. Set `AI_MAX_IMAGES` (1-3), `AI_MAX_OUTPUT_TOKENS`, or `AI_TIMEOUT_SECONDS` before launch.

The SQLite `vision_usage` ledger records token counts and estimated USD, not the question or image. Use `--no-access-log` as shown so Uvicorn does not log semantic-search query strings. A conservative reservation is checked against `AI_BUDGET_USD` (default **$0.25 total for this archive directory**, across restarts) before submission and reconciled when usage is returned. Timeouts and uncertain charges keep their reservation. Set the budget to `0` to prevent paid requests. This is **not a guaranteed billing cap**: prices, missing usage data, other clients, and provider billing can differ. Check your OpenAI dashboard for actual spending.

If a question returns no evidence, check camera status, archive filters, or index progress. For small printed text, move the phone closer or use the existing local OCR. For an OpenAI credential/model error, check the masked server configuration and project access. If the provider sends HTTP 400, Uvicorn logs the endpoint, status, known error type, code, field, a fixed explanation, response format, and a validated request ID when available. Share the single `OpenAI vision request rejected:` line, not the raw response or a key. An HTML error in place of API JSON is a reason to check the network path, including a VPN or proxy; it is not proof of which server rejected the request. A structured error for `service_tier` means you should check the API project's allowed tiers. A 429 can mean local budget exhaustion or provider quota; the response explains which. For speech failures, check trusted HTTPS, browser microphone permission, and installed voices. Type a question whenever recognition is unavailable.

For repeated HTTP 400 errors, run `.\.venv\Scripts\python.exe -m backend.openai_check` from the PowerShell terminal with the key set. It makes one authenticated model metadata GET, with no generation, image, question, or key output. Success confirms the project can read that model's metadata; it does not confirm vision generation. To look up an alternate model without changing the server configuration, pass its model ID with `--model`. The dashboard also shows the safe response format and request ID for a rejected vision request after restarting Uvicorn.

If `gpt-4o-mini` returns HTTP 404, check a supported alternative before retrying a question:

```powershell
.\.venv\Scripts\python.exe -m backend.openai_check --model gpt-4.1-mini
```

On success, set these variables in the same PowerShell terminal before running the HTTPS Uvicorn command above. [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini) accepts image input through the Responses API. The image-token value is a conservative local estimate for the app's resized JPEGs, not a provider limit.

```powershell
$env:AI_VISION_MODEL = 'gpt-4.1-mini'
$env:AI_INPUT_USD_PER_MILLION = '0.40'
$env:AI_OUTPUT_USD_PER_MILLION = '1.60'
$env:AI_ESTIMATED_IMAGE_TOKENS = '1200'
```

If the alternative lookup also fails, run `.\.venv\Scripts\python.exe -m backend.openai_check --list` to see public model IDs visible to that key. The listing does not test image support, generation permissions, or billing.

If the lookup itself returns HTTP 400, run `.\.venv\Scripts\python.exe -m backend.openai_check --model gpt-4o-mini --compare`. This adds two metadata GETs to the official API, one using the configured key and one using a deliberately invalid token. They omit SDK-specific headers and report only status and whether a body was present. If the minimal lookup succeeds with the configured key but the SDK fails, check whether the SDK is including incorrect `OPENAI_ORG_ID` or `OPENAI_PROJECT_ID` settings. If the dummy token gets HTTP 401 while the configured key still gets an empty HTTP 400, the failure is specific to the authenticated path; check project routing or a network intermediary before trying another vision request.

## Phase 6: Android Voice Assistant

Start the existing HTTPS server from the project terminal with the working OpenAI key, vision model and budget settings (see [Configure The Server](#configure-the-server)). There is no new backend, APK, dependency install or archive migration. Stop any previous server using port 8443 first.

```powershell
Set-Location 'C:\Users\khate\Documents\AIVisualAssistant'
if (-not $env:LIVE_SESSION_CODE) {
    $env:LIVE_SESSION_CODE = .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(24))"
}
Write-Host "Pairing code: $env:LIVE_SESSION_CODE"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8443 --workers 1 --no-proxy-headers --no-access-log --ssl-certfile .\certs\lan.pem --ssl-keyfile .\certs\lan-key.pem
```

### Android And TalkBack Checklist

1. Put the unlocked phone and laptop on the same Wi-Fi. Use the Wi-Fi IPv4 address from `ipconfig` and open `https://<LAPTOP-IP>:8443/phone` in Android Chrome. It must open without a certificate warning. Use **Install app** on the page when offered, or Chrome's **Add to Home screen** / **Install** menu; launch **AI Visual Assistant** from Android's home screen. It should open the phone assistant, not the dashboard.
2. With TalkBack on, navigate from **Start Assistant** to the pairing form and consent controls. Pair once using the terminal code. Check the OpenAI **Session consent** box. If on-device recognition is unavailable, allow **browser-managed speech** separately or use the typed fallback. In **Settings**, choose English or French; the language is saved on the device. A speech-pack download is optional when the browser offers it.
3. Select **Start Assistant**, grant the camera and microphone permissions when prompted, and wait for the spoken ready message. Ask, **“What is in front of me?”** Let the answer finish. Once listening resumes, ask **“Read the text on this paper.”** Neither turn should need a second press of Ask or Speak. Test **Talk Again** after permission or silence errors, and **Repeat Answer** without sending another AI request. On French, try **“Que vois-tu devant moi ?”**
4. Make sure a laptop is visible in an archived frame first. Ask **“When did I last see a laptop?”** or **“Quand ai-je vu un ordinateur portable pour la dernière fois ?”**. The answer should use archive evidence and Beirut capture times, or explain that no match was found. Try **“What did I see yesterday?”** to exercise the Beirut date filter.
5. At [https://localhost:8443/](https://localhost:8443/) on the laptop, verify the live preview, connection status, stored frames, detections, OCR, semantic search and manual **Ask AI**. Close the dashboard and ask another question from the phone: only the server needs to stay open.
6. Tap **Stop Assistant** while listening, processing, and speaking. The mic, camera, speech and connection should stop, and session image consent should clear. Also try denying a permission, turning off Wi-Fi, and relaunching after the server restarts. A stale camera should not be described as live. Re-pair only if the server session has expired or restarted.

The controller transitions through `IDLE`, `CONNECTING`, `READY`, `LISTENING`, `PROCESSING`, `SPEAKING`, `ERROR`, and `STOPPED`. It accepts final transcripts only, waits for the recognition session to end before sending a question, and resumes listening after playback and a short pause. In silence it makes at most one automatic retry. **Talk Again** and **Type a question** work when automatic recognition is blocked; **Manual camera and Ask AI** preserves the earlier controls outside a voice session. Ordinary or ambiguous questions default to live camera. Common past-tense or archive cues in English/French route to the existing semantic archive search without another paid classification request. The phone's live requests require a fresh frame; the dashboard keeps its labeled archive fallback.

This is an assistive demo, not a navigation or obstacle-avoidance system. Keep the app foregrounded and the screen unlocked. Android Chrome can restrict speech restart or playback, including in an installed PWA; use **Talk Again** when needed. The static shell can open offline, but pairing, the camera stream, archive and AI need the laptop server. No speech audio is sent over WebRTC. If you opt in to browser-managed speech, the browser may use its own service. Stopping aborts the browser request where possible, but does not reverse an image submission or charge that already reached the provider.

## Tests

    ```powershell
    .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
    .\.venv\Scripts\python.exe -m pytest -q
    ```

    The tests use the real API and OpenCV decoding, with model loading and inference mocked. Frame archive tests use temporary SQLite databases and directories. No tests download weights or require OCR language data.

The WebRTC loopback test uses two real peers and a synthetic video track to verify signaling, relay, bounded sampling, archive writes, and disconnection; it skips if the WebRTC runtime is not installed. This does not replace a phone-over-Wi-Fi test of camera permissions, certificate trust, or ICE connectivity.

Phase 5 tests use temporary SQLite archives, a deterministic fixture encoder, and the real OpenAI SDK with a mocked HTTP transport. They never read a real OpenAI key, upload images, or make paid requests.

With Node.js 22+, run `node --test tests/*.test.mjs` for the voice controller, camera lifecycle and routing. The optional `node tests/mobile-browser-smoke.mjs` starts an isolated local server and desktop Chrome (set `CHROME_EXE` if needed) with a synthetic camera, mocked speech/AI and a temporary archive. It verifies the phone page, real browser-to-server WebRTC, English/French switching, the manifest and laptop dashboard. It does not verify Android installation, native speech, permissions or TalkBack; use the checklist above for those.

## Final Enhancements

- **Speech:** Change the language under Settings while the assistant is running. The previous turn is cancelled; recognition restarts with `fr-FR` or `en-US`. Speech synthesis sets both the chosen voice and its locale. Same-language regional voices can be used with a notice. If no compatible voice exists, there is no default-language fallback. Install or enable a voice for the selected language in Android's text-to-speech settings, then reopen the PWA. The optional browser-managed speech consent is still required for voices marked as remote.
- **History:** `Asia/Beirut` governs spoken times and `today`, `yesterday`, and `this morning`. “Two hours ago” accepts a 30-minute window on either side of the requested instant. Explicit filter bounds intersect the inferred time window. UTC remains canonical. New live frames use server-receipt time; old archive entries keep their original timestamps. Responses include UTC and Beirut timestamps for selected evidence. No GPS or named locations are stored.
- **Retrieval:** First/last/presence questions about common object classes use an indexed match at `HISTORY_OBJECT_CONFIDENCE` (default `0.45`) and a timestamp sort. Historical OCR uses saved text and, when specified, an exact case-insensitive phrase filter. Descriptive object questions use up to 48 label-filtered records before SigLIP reranking; other descriptions use semantic candidates. Ranking is not a confirmed sighting. Local answers return `model: null` and no API cost. A matching class cannot identify the same physical object across frames.
- **Migration:** SQLite `user_version=1` adds `frame_objects` and its label index. It backfills from existing JSON in one transaction, preserves timestamps and images, and cascades when frames are deleted or expire. `tzdata` supplies the IANA database for Python on Windows. It is the only new dependency.
- **Important changes:** Enable **Describe important changes** under phone Settings. New observations require `SCENE_CONFIRMATIONS` consecutive samples (default `2`), `SCENE_CONFIDENCE` (default `0.60`), and `SCENE_COOLDOWN_SECONDS` between announcements (default `18`). It reuses local YOLO/OCR output and never calls OpenAI. It describes visibility and image-relative position, not depth, passability, or safety. YOLO11n's installed class set does not include stairs, steps or doorways; those can be asked about with the on-demand vision model, not relied on as alerts. **Talk Again** or a typed question can interrupt an announcement; the microphone is off during playback to prevent feedback.
- **Timings:** Phone Settings > **Development timings** shows numeric-only stage timings. Recognition measures the session up to the final transcript; TTS startup uses the browser's `onstart` event, not an acoustic measurement. Live evidence can include frame age, encoding time and timing from a recent sample. Cancelling a pending question aborts the browser request and the server cancels on disconnect; a request already sent to the provider may still be charged. There are no automatic paid retries.

See [the final evaluation and seven-scenario demo](evaluation/REPORT.md) and the [20-question fixture](evaluation/questions.json).
