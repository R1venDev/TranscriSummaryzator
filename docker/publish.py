#!/usr/bin/env python3
"""Publish aliases without overwriting commit or release image identities."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def manifest(reference, *, missing=False):
    process = subprocess.run(["docker", "buildx", "imagetools", "inspect", reference,
                              "--format", "{{json .Manifest}}"], text=True, capture_output=True)
    if process.returncode:
        message = process.stderr.lower()
        if missing and any(value in message for value in ("not found", "manifest unknown", "404")):
            return None
        raise RuntimeError("Cannot inspect registry image: " + reference)
    return json.loads(process.stdout)["digest"]


def main():
    action = sys.argv[1]
    repository = os.environ["GITHUB_REPOSITORY"].lower()
    revision = os.environ["GITHUB_SHA"]
    variant = os.environ["IMAGE_VARIANT"]
    if variant not in {"summary", "speech"} or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Invalid image identity")
    if not re.fullmatch(r"[a-z0-9_.-]+/[a-z0-9_.-]+", repository):
        raise ValueError("Invalid repository")
    image = "ghcr.io/" + repository
    commit_image = f"{image}:sha-{revision}-{variant}"
    reference = os.environ["GITHUB_REF"]
    version = (reference.removeprefix("refs/tags/v") if reference.startswith("refs/tags/v")
               else os.environ.get("RELEASE_VERSION", "").strip())
    if version and not re.fullmatch(r"[0-9][A-Za-z0-9_.-]{0,90}", version):
        raise ValueError("Invalid release version")
    if action == "prepare":
        exists = manifest(commit_image, missing=True) is not None
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            output.write(f"image={image}\ncommit_image={commit_image}\nexists={str(exists).lower()}\n")
        return
    if action not in {"promote", "validate"}:
        raise ValueError("Unknown publishing action")
    digest = manifest(commit_image)
    targets = []
    if version:
        release_image = f"{image}:{version}-{variant}"
        existing = manifest(release_image, missing=True)
        if existing is not None and existing != digest:
            raise RuntimeError("Release tag already names another image; choose a new version")
        targets.append(release_image)
    if reference == "refs/heads/main":
        targets.append(f"{image}:{variant}")
    if action == "validate":
        return
    for target in targets:
        subprocess.run(["docker", "buildx", "imagetools", "create", "--tag", target,
                        f"{image}@{digest}"], check=True)
        if manifest(target) != digest:
            raise RuntimeError("Published alias does not match the verified image")
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as output:
        output.write(f"## {variant}\n\nDigest: `{image}@{digest}`\n\n")
        output.write("\n".join(f"- `{target}`" for target in [commit_image, *targets]) + "\n")


if __name__ == "__main__":
    main()
