#!/bin/sh
# shellcheck disable=SC2039
# shellcheck disable=SC2034
count=800
for i in $(seq $count)
do
   python3 client_mehrdad_agent.py
   sleep 0.4
done
