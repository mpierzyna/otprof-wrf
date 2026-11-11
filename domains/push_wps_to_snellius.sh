#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

# EXECUTE ON DELFTBLUE to push WPS data to Snellius.
#
# Usage: ./push_wps_to_snellius.sh <domain> [CONFIRM]
#   Without CONFIRM, rclone runs in --dry-run mode.

set -euo pipefail

REMOTE_DOMAINS_DIR="/home/mpierzyna/prjs1785/otprof-wrf/domains"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ] || [ -z "$1" ]; then
    echo "Usage: $0 <domain> [CONFIRM]" >&2
    exit 1
fi

DOMAIN="${1%/}"
LOCAL_DIR="${SCRIPT_DIR}/${DOMAIN}"

if [ ! -d "${LOCAL_DIR}" ]; then
    echo "Error: domain directory '${LOCAL_DIR}' does not exist." >&2
    exit 1
fi

# Pass 'CONFIRM' to actually run
if [ "${2:-}" = "CONFIRM" ]; then
    DRYRUN=()
else
    DRYRUN=(--dry-run)
    echo "Dry run. Pass CONFIRM as second argument to actually copy." >&2
fi

rclone copy \
    "${DRYRUN[@]}" \
    --progress \
    --transfers=8 \
    --multi-thread-streams=1 \
    --include '**/met_em*.nc' \
    "${LOCAL_DIR}" "snellius:${REMOTE_DOMAINS_DIR}/${DOMAIN}"
