from scripts.article_classifier import run_article_classifier

from sys import exit
import argparse

# Create ArgumentParser object
parser = argparse.ArgumentParser(description='Argparser to decide which script to run')

# Add an argument
parser.add_argument('--script',
                    choices=['article_classifier', 'incident_classifier'],
                    help=f"Select script to run: choose one of: 'article_classifier', 'incident_classifier'",
                    default='article_classifier')
# Parse the arguments
args = parser.parse_args()

# Access the argument value using args.mode
if args.script == 'article_classifier':
    print(f"Running {args.script}")
    run_article_classifier()
    # Add code specific to option 1
elif args.script == 'incident_classifier':
    print(f"Running {args.script}")
    print("This script is not yet ready to run")
    exit()
    # Add code specific to option 2
else:
    print("Invalid script selected")
