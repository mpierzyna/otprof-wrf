#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Use rclone to copy CERRA files defined in `include.txt` from remote to current directory
rclone copy --include-from includes.txt --transfers=4 --progress tudelft:staff-umbrella/HBaki/CERRA .