import os

import json
import tempfile
import base64

import psutil


# https://www.mongodb.com/docs/drivers/pymongo/#stable-api
from pymongo.mongo_client import MongoClient

# from pymongo.server_api import ServerApi


def delete_file(name):
    if name is not None and len(name) and os.path.isfile(name):
        print(f"[delete_file] Removing temporary cert file: {name}")
        os.remove(name)


def get_mongo_connection_string(create_ca_file=True):
    result = ("", "")
    tmp = None

    try:
        mongo_db = json.loads(
            os.environ["mongoDB"] if "mongoDB" in os.environ else "{}"
        )
        url = mongo_db["connection"]["mongodb"]["composed"][0]
        if create_ca_file is True:
            cert = mongo_db["connection"]["mongodb"]["certificate"][
                "certificate_base64"
            ]
            tmp = tempfile.NamedTemporaryFile(delete=False)
            with open(tmp.name, "wb") as file:
                b64 = base64.b64decode(cert)
                file.write(b64)
                print(f"[get_mongo_connection_string] Cert file created: {file.name}")
            result = (url, file.name)
        else:
            result = (url, None)
    except KeyError as exc:
        print(f"[get_mongo_connection_string] Error in env variable 'mongoDB': [{exc}]")
        if tmp is not None and "name" in tmp.keys():
            delete_file(tmp.name)
    return result


def get_mongo_client(tls=True):
    TLS_CA_FILE = ""
    try:
        MONGODB_URL, TLS_CA_FILE = get_mongo_connection_string(tls)
        kwargs = {
            # "server_api": ServerApi("1"),
            # compressors='zstd,zlib,snappy',
            "compressors": "zstd,zlib",
            "zlibCompressionLevel": 9,
            "tls": tls,
            "tlsCAFile": TLS_CA_FILE,
        }
        if tls is False:
            if "tls" in kwargs:
                del kwargs["tls"]
            if "tlsCAFile" in kwargs:
                del kwargs["tlsCAFile"]

        client = MongoClient(
            MONGODB_URL,
            **kwargs
            # server_api=ServerApi("1"),
            # # compressors='zstd,zlib,snappy',
            # compressors="zstd,zlib",
            # zlibCompressionLevel=9,
            # tls=tls,
            # tlsCAFile=TLS_CA_FILE,
        )
        return client
    except Exception as exc:
        print("[get_mongo_client] Exception", exc)
    finally:
        delete_file(TLS_CA_FILE)


def print_meminfo():
    suffix = ["", "KB", "MB", "GB", "TB"]
    mem = psutil.Process().memory_info().rss
    index = 0
    while mem > 1024 and index < len(suffix):
        mem = mem / 1024
        index = index + 1
    print(f"[rss memory] {mem:.4}{suffix[index]}")
