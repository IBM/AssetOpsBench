#!/bin/sh -xe

cat >/opt/couchdb/etc/local.ini <<EOF
[couchdb]
single_node=true

[admins]
${COUCHDB_USERNAME} = ${COUCHDB_PASSWORD}
EOF

echo "Starting CouchDB..."
/opt/couchdb/bin/couchdb &

echo "Waiting for CouchDB to be ready..."
until curl -sf -u "${COUCHDB_USERNAME}:${COUCHDB_PASSWORD}" http://localhost:5984/ >/dev/null; do
  sleep 2
done
echo "CouchDB is ready."

echo "Installing Python dependencies..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip
pip3 install -q --break-system-packages requests pandas python-dotenv

echo "Loading IoT asset data..."
COUCHDB_URL="http://localhost:5984" \
  python3 /couchdb/init_asset_data.py \
    --data-file /sample_data/chiller6_june2020_sensordata_couchdb.json \
    --db "${IOT_DBNAME:-chiller}"

echo "Loading work order data..."
COUCHDB_URL="http://localhost:5984" \
  python3 /couchdb/init_wo.py \
    --data-dir /sample_data/work_order \
    --db "${WO_DBNAME:-workorder}"

echo "✅ All databases initialised."
tail -f /dev/null
