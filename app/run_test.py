import argparse
from tests.test_db_connectors import (
    test_milvus_connector, 
    test_single_store, 
    test_elasticsearch
)


def run_test():
    parser = argparse.ArgumentParser(description="Choose which test script to run.")
    parser.add_argument(
         "--test_script", 
         help="name of the test function to run.",
         default='elasticsearch'
                        )
    
    args = parser.parse_args()

    if args.test_script in ['test_milvus_connector.py', 'milvus', 'test_milvus', 'test_milvus_connector']:
        # run the milvus connection test.
        test_milvus_connector()
    if args.test_script in ['singlestore', 'single_store', 'test_single_store']:
        # run the single store connection test.
        test_single_store()
    if args.test_script in ['elasticsearch', 'test_elasticsearch']:
         # run the elasticsearch connection test.
         test_elasticsearch()


if __name__ == '__main__':
        run_test()
    