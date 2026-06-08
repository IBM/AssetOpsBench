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

echo "Loading default data (work_order + iot + vibration)..."
COUCHDB_URL="http://localhost:5984" \
  python3 /couchdb/init_data.py

echo "✅ All databases initialised."
tail -f /dev/null
