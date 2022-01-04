#!/bin/sh
# shellcheck disable=SC2039
# shellcheck disable=SC2034
count=800
for i in $(seq $count)
do
   python3 server_main.py
   sleep 0.3
done
