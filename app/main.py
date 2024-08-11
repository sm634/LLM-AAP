from src.classes.data_preparation import ElasticDataPrep
from utils.files_handler import FileHandler
from src.create_knowledge_base import create_knowledge_base

file_handler = FileHandler()


def main():
    # get models config
    file_handler.get_config('models_config.yaml')
    config = file_handler.config

    task = config['TASK'].lower()
    print("Running task: ", task)
    if task == 'preprocess_pipeline':        
        elastic_pipeline_class = ElasticDataPrep()
        elastic_pipeline_class.run_data_prep_pipeline()
    elif task == 'create_knowledge_base':
        create_knowledge_base(
            collection_name='test_collection',
            subject='test',
            websites_file='model_cards.txt',
            custom_schema=True
        )
 
    print("Task Complete")


if __name__ == '__main__':
    main()
