#!/bin/sh
# Entry point of the SageMaker monitoring jobs.
#
# SageMaker Processing mounts its input and output folders at run time, owned by
# root, so a process that starts as a normal user cannot write its report. This
# script starts as root only to hand those folders to the "app" user, then drops
# root for good and runs the job as that user.
set -eu
for dir in /opt/ml/processing/input /opt/ml/processing/output; do
    if [ -d "$dir" ]; then
        chown -R app:app "$dir"
    fi
done
exec setpriv --reuid=app --regid=app --init-groups "$@"
