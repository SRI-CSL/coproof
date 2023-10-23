#!/bin/sh
curl --header "Content-Type: application/json" \
     --request POST \
     --data @demo_compare.json \
     http://localhost:7111/compare
