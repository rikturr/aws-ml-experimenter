#!/usr/bin/env bash

echo 'datetime,mem_used,mem_free,disk_used,disk_free,cpu'
while true
do
    sleep 1
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$(free -m | awk 'NR==2{printf "%s,%s\n", $3,$4 }'),$(df / -h | grep -v Filesystem | awk -F' ' '{printf "%s,%s\n", $3,$4}'),\"$(uptime)\""
done