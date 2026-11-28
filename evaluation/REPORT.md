# Final Evaluation

Measured on 28 November 2026 on Windows 11, Intel Core i5-12450H, Python 3.11.9, CPU. No paid API calls, downloads, or scans of personal archive data were needed. Reproduce from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
node --test tests/*.test.mjs
node tests/mobile-browser-smoke.mjs
.\.venv\Scripts\python.exe -m evaluation.run
```

Final automated verification: **113 Python tests passed**, **24 JavaScript tests passed**, and the desktop Chrome/WebRTC smoke test passed. A migration rehearsal on a temporary copy of the existing archive preserved **180 of 180** frame rows and indexed 439 frame/label pairs with no foreign-key errors. The original archive was opened read-only.

## Fixture and Outcomes

- [20 questions](questions.json) cover live awareness, reading, safety limits, earliest/latest sightings, missing evidence, object identity, time windows, and EN/FR routing. The scripted checks exercise routing and deterministic evidence selection; the `expected` statements for cloud descriptions are manual evaluation criteria, not scored model outputs.
- The offline benchmark reuses `models/phase2-smoke-bus.jpg` (810 x 1080) and `models/phase2-smoke-sign.jpg` (600 x 200, text `HELLO WORLD`). The runner requires existing model files and will not download missing assets. The SQLite test suite also uses synthetic laptop observations and a `ROOM 204` OCR record. No personal images are included in the fixture.

| Check | Observed | Scope |
|---|---|---|
| SigLIP 2 ranking | Precision@1 **4/4**, Recall@1 **4/4** | Two indexed images; one labeled relevant image per query; one English and one French query for each scene. Not a representative accuracy estimate. |
| YOLO11n | `bus` and four `person` boxes | One image; no detection mAP or bounding-box annotation evaluation. |
| Tesseract | `HELLO WORLD`; exact match; 0 character errors over 11 reference characters | One English sign. Does not measure arbitrary OCR accuracy. |
| Local historical answers | First/last frame IDs, missing labels, OCR quote, and no identity/location claim asserted | Synthetic archived evidence; no cloud inference. |
| Live description grounding | Fresh-frame checks, provider request grounding, and deterministic distance/safety limits tested | Provider mocked; real AI answer quality not scored. |
| Language switching | Chrome requested French recognition and a French utterance/voice, then returned to English | Voice objects and audio mocked; audible accent and TalkBack still require Android testing. |

## Local Timing

Milliseconds. Five warm measurements per stage. Excludes weights loading, archive backfill, physical phone transport, OpenAI and speech. Search includes local text encoding and rank over two images. Local history includes query handling and SQLite retrieval, not HTTPS round trip.

| Stage | Mean | Median | Maximum |
|---|---:|---:|---:|
| YOLO11n bus image | 52.84 | 46.95 | 77.03 |
| Tesseract sign image | 14.04 | 14.24 | 14.48 |
| SigLIP 2 semantic search | 32.96 | 31.64 | 36.43 |
| Local historical OCR answer | 2.11 | 1.88 | 2.75 |

**Not measured:** real Android question-to-audio latency, on-device pronunciation, paid-provider latency/accuracy, larger-archive retrieval quality, and formal user accessibility feedback. For live evaluation, collect five consenting runs per scenario; record phone model, installed voices, network, and vision model. Compare each answer against the actual selected evidence, including object, text, time, identity and unsupported claims. Use the development timings to compare components and an external stopwatch or recording for audible start. Do not present a timed Chrome stub as an Android result.

## Android Demo

Start FastAPI with the existing HTTPS and OpenAI settings. Open the phone page on trusted LAN HTTPS and the laptop dashboard at `https://localhost:8443/`. Pair and consent. Keep Android Chrome foregrounded. Install compatible EN/FR voices first; if a voice is missing, verify the accessible warning rather than accepting the wrong accent. Test with TalkBack enabled and keep **Talk Again** available.

1. **Live English:** Point the camera at a laptop on a table. Ask “What is in front of me?” Expect one or two evidence-based sentences, not a distance or safety assessment.
2. **Live French:** Switch languages during a turn. Ask “Qu'est-ce qu'il y a devant moi ?” Verify that prior playback stops and the next answer is pronounced in French.
3. **First sighting:** Archive two views of a laptop. Ask “When did you first see a laptop?” Compare the spoken Beirut time with the earliest matching dashboard record.
4. **Archived OCR:** Archive a clear `ROOM 204` sign, then move away. Ask “What was written on the sign ‘ROOM 204’ earlier?” Verify the stored OCR and time; a weak reading can yield no match.
5. **Missing record:** Choose a known label absent from the demo archive, such as a bicycle. Ask whether it was seen yesterday. Expect no recorded sighting, not a claim that you never saw one.
6. **Important changes:** Enable the setting and place a chair near the center. Keep the view steady for two sampled detections. Expect at most one relevant notice in the cooldown interval, with no safety claim. Ask a question or use **Talk Again** to interrupt the optional turn.
7. **Switch back:** Select English while speaking or processing, verify previous playback/results are discarded, and ask again. Tap **Stop Assistant** and verify that camera, mic and speech stop.

The prototype describes a camera view, not safe passage. It does not estimate metric distance, perform physical identity tracking, or record named locations. The coarse sample rate and small detector class set can miss changes, and browser voices, permissions and speech restart vary by device.
