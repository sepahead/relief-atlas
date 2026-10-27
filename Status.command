#!/bin/sh
set -eu
atlas_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$atlas_root/atlas" status
