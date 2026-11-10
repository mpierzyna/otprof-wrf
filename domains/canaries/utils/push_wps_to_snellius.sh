#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

# EXECUTE ON DELFTBLUE to push WPS data to Snellius.
# Execute from workspace root (ie one level above)

# Pass 'CONFIRM' to actually run
if [ "$1" = "CONFIRM" ]; then
    DRYRUN=''
else
    DRYRUN='--dry-run'
fi

rclone copy \
    ${DRYRUN} \
    --progress \
    --transfers=8 \
    --multi-thread-streams=1 \
    --include 'met_em*.nc' \
    . snellius:/scratch-shared/mpierzyna/