# Optional self-hosted Plane 3.1.0 compatibility overlay, built locally.
# No Plane source, secrets or private documents are distributed here.
ARG PLANE_LIVE_IMAGE=makeplane/live-commercial:v3.1.0
FROM ${PLANE_LIVE_IMAGE}
USER root
RUN node -e 'const fs=require("fs");const p="/app/apps/live/dist/start.mjs";const old="express.json({ verify:";let s=fs.readFileSync(p,"utf8");if(s.split(old).length!==2)throw new Error("Plane parser contract changed; review overlay");s=s.replace(old,"express.json({ limit: process.env.PLANE_JSON_BODY_LIMIT || \"10mb\", verify:");fs.writeFileSync(p,s);'
USER node
