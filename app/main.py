import yaml
from scripts.article_classifier import run_article_classifier


def main():

    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    task = config['TASK'].lower()
    if task == 'article_classifier':
        run_article_classifier()


if __name__ == '__main__':
    main()
