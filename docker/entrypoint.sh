#!/bin/bash
# Keep the container alive so the harness can dispatch commands via docker exec
exec tail -f /dev/null
