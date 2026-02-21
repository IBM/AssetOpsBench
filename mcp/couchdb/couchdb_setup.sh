#!/bin/sh -xe


COUCHDB_USERNAME=${COUCHDB_USERNAME}
COUCHDB_PASSWORD=${COUCHDB_PASSWORD}
COUCHDB_DBNAME=${COUCHDB_DBNAME}
COUCHDB_URL="http://${COUCHDB_USERNAME}:${COUCHDB_PASSWORD}@127.0.0.1:5984"
INPUT_FILE="/sample_data/chiller6_june2020_sensordata_couchdb.json"
OUTPUT_FILE="/sample_data/bulk_docs.json"

# Convert the JSON file into a coudb bulk insertable JSON file
if [ ! -f "$INPUT_FILE" ]; then
  echo "❌ Error: $INPUT_FILE not found."
fi

# Read the array from file (single line) and wrap it
ARRAY_CONTENT=$(cat "$INPUT_FILE")
echo "{\"docs\": $ARRAY_CONTENT}" > "$OUTPUT_FILE"

echo "✅ Wrapped $INPUT_FILE into $OUTPUT_FILE"

cat >/opt/couchdb/etc/local.ini <<EOF
[couchdb]
single_node=true

[admins]
${COUCHDB_USERNAME} = ${COUCHDB_PASSWORD}
EOF

echo "Starting CouchDB..."

/opt/couchdb/bin/couchdb &
sleep 10

echo "Connecting to CouchDB..."
curl -u ${COUCHDB_USERNAME}:${COUCHDB_PASSWORD} GET http://localhost:5984/
curl -u ${COUCHDB_USERNAME}:${COUCHDB_PASSWORD} GET http://localhost:5984/_all_dbs

# Check if database exists
response=$(curl -s -o /dev/null -w "%{http_code}" "$COUCHDB_URL/$COUCHDB_DBNAME")

if [ "$response" -eq 200 ]; then
  echo "⚠️ Database $COUCHDB_DBNAME already exists. Skipping creation."
else
  echo "⚠️ Database $COUCHDB_DBNAME does not exist. Creating..."
  curl -s -X PUT "$COUCHDB_URL/$COUCHDB_DBNAME"

  echo "Uploading documents from $JSON_FILE..."
  curl -s -X POST "$COUCHDB_URL/$COUCHDB_DBNAME/_bulk_docs" \
    -H "Content-Type: application/json" \
    -d @"$OUTPUT_FILE"

  curl -s -X POST "$COUCHDB_URL/$COUCHDB_DBNAME/_index" -H "Content-Type: application/json" -d '{ "type" : "json", "index": { "fields": [ "asset_id", "timestamp" ] } }'

  echo "✅ Database created and populated."
fi

tail -f /dev/null
