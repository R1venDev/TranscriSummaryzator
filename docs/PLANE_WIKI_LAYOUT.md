# Wiki layout v2

A newly exported meeting uses Plane's page title once, followed by the actual
workspace member mentions, one browser-compatible video, one chapter block,
a closed native disclosure with the full immutable source transcript, then the
summary. The eight summary sections remain; Timecodes is represented by the
chapter block instead of a repeated list in the body.

`summary/plane_wiki.py` projects original text, speakers and U-ID/timecode links.
`plane_projection.py` receives the same source revision as the effective summary.
Member matching is unique, exact, case-insensitive `display_name` matching of
structured source names. Unknown/ambiguous people remain visible plain text.
The backend resolves native `mention-component` UUIDs with the existing API key.
No model calls or additional speaker attribution are involved.

Only the playable MP4 is exported. A supported H.264/AAC MKV is remuxed with
stream copy; the uploaded original is retained locally. Previously uploaded
original assets are not deleted, but their duplicate Wiki node is removed by an
explicit layout migration. An unsupported codec is reported rather than silently
transcoded. Audio-only pages contain the full source disclosure and retain their Timecodes
section; video insertion replaces that list with one native chapter block.

`summary.plane_media.refresh_layout(store, job_id, source_index, transcript_url)`
is an explicit migration for app-owned existing video pages. It preserves remote
summary text and manual sections, reuses the confirmed asset and verifies source
identity. Exact before/target are saved privately with mode 0600 before PUT.
Round-trip checks cover every transcript row, link and native block order. A lost
PUT response is recovered by the layout marker **and complete content checks**.
Edited/incomplete migrated content is treated as a conflict, not overwritten.
Plane v1 provides no conditional PUT; the pre-write re-read catches observed
concurrent changes but cannot guarantee preservation during the network race.

The optional installed Plane player can be updated using
`python3 scripts/plane_video_single_timecodes.py /path/to/plane-video-attachments-player.js`.
This bounded patch hides the redundant source callout only after the player binds
its chapters. “Редактировать таймкоды” reveals the native editor. Without a working
player, the native callout stays visible. The document itself is not changed.
The installer saves `.before-single-timecodes`; restore it to roll back presentation.

Release: build the existing Docker `summary` target with the actual commit SHA;
replace only app/scheduler compose images. Keep the prior compose/image as rollback.
Do not restart the speech service. Rollback does not delete source, generations,
tasks, user edits or remote uploads; saved Wiki before HTML can restore layout.
