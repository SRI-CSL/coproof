#!/bin/sh
curl --header "Content-Type: application/json" \
     --request POST \
     --data @demo_state3.json \
     http://localhost:7001/query
