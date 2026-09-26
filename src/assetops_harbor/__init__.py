"""Harbor integration for AssetOpsBench.

Harbor (https://github.com/harbor-framework/harbor) runs one Compose project
per trial, which gives every scenario its own CouchDB and makes concurrent
scenario runs safe by construction. See harbor/README.md.
"""
