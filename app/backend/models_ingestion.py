import os
import re
import time
import json
import random
from enum import Enum
from typing import List, Optional
from contextlib import asynccontextmanager
from elasticsearch import Elasticsearch
from watsonXdiscovery import watsonXdiscovery_properties as wXdProp
from watsonXdiscovery import ingest


es_connection = Elasticsearch(
            wXdProp.URL,
            ca_certs=wXdProp.CERT_PATH,
            basic_auth=(wXdProp.USERNAME, wXdProp.PASSWORD),
            max_retries=wXdProp.MAX_RETRIES, retry_on_timeout=wXdProp.RETRY_ON_TIMEOUT,
            request_timeout=wXdProp.REQUEST_TIMEOUT)

sample_json = [
    {
        "model_id": "bigscience/mt0-xxl",
        "label": "bigscience/mt0-xxl",
        "provider": "Hugging Face",
        "source": "Hugging Face",
        "functions": [
            {
                "id": "text_generation"
            }
        ],
        "short_description": "An instruction-tuned iteration on mT5.",
        "long_description": "mt0-xxl (13B) is an instruction-tuned iteration on mT5. Like BLOOMZ, it was fine-tuned on a cross-lingual task mixture dataset (xP3) using multitask prompted finetuning (MTF).",
        "input_tier": "class_2",
        "output_tier": "class_2",
        "number_params": "13.9B",
        "min_shot_size": 0,
        "task_ids": [
            "question_answering",
            "summarization",
            "classification",
            "generation"
        ],
        "tasks": [
            {
                "id": "question_answering",
                "ratings": {
                    "quality": 3
                }
            },
            {
                "id": "summarization",
                "ratings": {
                    "quality": 3
                }
            },
            {
                "id": "retrieval_augmented_generation",
                "ratings": {
                    "quality": 2
                }
            },
            {
                "id": "classification",
                "ratings": {
                    "quality": 3
                }
            },
            {
                "id": "generation"
            },
            {
                "id": "extraction",
                "ratings": {
                    "quality": 2
                }
            }
        ],
        "model_limits": {
            "max_sequence_length": 4096
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 700
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 4095
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 4095
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2023-07-07"
            }
        ],
        "tuned_by": "none",
        "ratings": "none",
        "training_parameters": "none",
        "versions": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "Decoder-only",
        "ModelOverview": "The bigscience/mt0-xxl model is a text-to-text generation transformer model that is capable of following human instructions in dozens of languages zero-shot. It was fine-tuned on the xP3 task mixture and has demonstrated cross-lingual generalization to unseen tasks and languages. The model can be used to perform tasks expressed in natural language, such as translation, and can understand both pre-training and fine-tuning languages. It has a massive size of 13.9B parameters and uses the F32 tensor type.",
        "languages": [
            "English",
            "French",
            "Spanish",
            "Telugu",
            "Chinese",
            "Japanese",
            "Portuguese",
            "Russian"
        ],
        "TuningInformation": "Instruction-tuned",
        "TrainingData": "Code and code-related data",
        "UsesSupported": [
            "Text generation",
            "Translation",
            "Question answering"
        ],
        "OptimisedFor": "Low latency, real-time text generation, long inputs",
        "PromptingAdvice": "Provide the model with natural language prompts, and make it clear when the input stops to avoid the model trying to continue it.",
        "Output": "Text",
        "PromptTuningAvailability": "true",
        "RegionalAvailability": "",
        "License": "Apache-2.0"
    },
    {
        "model_id": "codellama/codellama-34b-instruct-hf",
        "label": "CodeLlama-34b-Instruct-hf",
        "provider": "Meta",
        "source": "Hugging Face",
        "functions": [
            {
                "id": "text_generation"
            }
        ],
        "short_description": "Code Llama is an AI model built on top of Llama 2, fine-tuned for generating and discussing code.",
        "long_description": "Code Llama is a pretrained and fine-tuned generative text models with 34 billion parameters. This model is designed for general code synthesis and understanding.",
        "input_tier": "class_2",
        "output_tier": "class_2",
        "number_params": "34B",
        "min_shot_size": 0,
        "task_ids": [
            "code"
        ],
        "tasks": [
            {
                "id": "code"
            }
        ],
        "model_limits": {
            "max_sequence_length": 16384
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 8192
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 8192
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 8192
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2024-03-14"
            }
        ],
        "tuned_by": "none",
        "ratings": "none",
        "training_parameters": "none",
        "versions": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "Decoder-only",
        "ModelOverview": "Code Llama is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 34 billion parameters. This model is designed for general code synthesis and understanding. It can be used for code completion, infilling, instructions, and chat. The model is an auto-regressive language model that uses an optimized transformer architecture. It was trained between January 2023 and July 2023 and has been fine-tuned for safer deployment. The model is intended for commercial and research use in English and relevant programming languages.",
        "languages": [
            "Python",
            "English"
        ],
        "TuningInformation": "Instruct-tuned",
        "TrainingData": "Code and code-related data",
        "UsesSupported": [
            "Code completion",
            "infilling",
            "instructions",
            "chat"
        ],
        "OptimisedFor": "Safer deployment",
        "PromptingAdvice": "",
        "Output": "Text",
        "PromptTuningAvailability": "true",
        "RegionalAvailability": "",
        "License": "Custom commercial license"
    },
    {
        "model_id": "google/flan-t5-xl",
        "label": "FLAN-T5-XL",
        "provider": "Hugging Face",
        "source": "Hugging Face",
        "functions": [
            {
                "id": "prompt_tune_inferable"
            },
            {
                "id": "prompt_tune_trainable"
            },
            {
                "id": "text_generation"
            }
        ],
        "short_description": "A pretrained T5 - an encoder-decoder model pre-trained on a mixture of supervised / unsupervised tasks converted into a text-to-text format.",
        "long_description": "flan-t5-xl (3B) is a 3 billion parameter model based on the Flan-T5 family. It is a pretrained T5 - an encoder-decoder model pre-trained on a mixture of supervised / unsupervised tasks converted into a text-to-text format, and fine-tuned on the Fine-tuned Language Net (FLAN) with instructions for better zero-shot and few-shot performance.",
        "input_tier": "class_1",
        "output_tier": "class_1",
        "number_params": "2.85B",
        "min_shot_size": 0,
        "task_ids": [
            "question_answering",
            "summarization",
            "retrieval_augmented_generation",
            "classification",
            "generation",
            "extraction"
        ],
        "tasks": [
            {
                "id": "question_answering"
            },
            {
                "id": "summarization",
                "tags": [
                    "function_prompt_tune_trainable"
                ]
            },
            {
                "id": "retrieval_augmented_generation"
            },
            {
                "id": "classification",
                "tags": [
                    "function_prompt_tune_trainable"
                ]
            },
            {
                "id": "generation",
                "tags": [
                    "function_prompt_tune_trainable"
                ]
            },
            {
                "id": "extraction"
            }
        ],
        "model_limits": {
            "max_sequence_length": 4096,
            "training_data_max_records": 10000
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 4095
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 4095
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 4095
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2023-12-07"
            }
        ],
        "training_parameters": {
            "init_method": {
                "supported": [
                    "random",
                    "text"
                ],
                "default": "random"
            },
            "init_text": {
                "default": "text"
            },
            "num_virtual_tokens": {
                "supported": [
                    20,
                    50,
                    100
                ],
                "default": 100
            },
            "num_epochs": {
                "default": 20,
                "min": 1,
                "max": 50
            },
            "verbalizer": {
                "default": "Input: {{input}} Output:"
            },
            "batch_size": {
                "default": 16,
                "min": 1,
                "max": 16
            },
            "max_input_tokens": {
                "default": 256,
                "min": 1,
                "max": 256
            },
            "max_output_tokens": {
                "default": 128,
                "min": 1,
                "max": 128
            },
            "torch_dtype": {
                "default": "bfloat16"
            },
            "accumulate_steps": {
                "default": 16,
                "min": 1,
                "max": 128
            },
            "learning_rate": {
                "default": 0.3,
                "min": 1e-05,
                "max": 0.5
            }
        },
        "tuned_by": "none",
        "ratings": "none",
        "versions": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "",
        "ModelOverview": "FLAN-T5-XL is a language model that has been fine-tuned on more than 1000 additional tasks covering multiple languages. It achieves state-of-the-art performance on several benchmarks, including 75.2% on five-shot MMLU. The model is designed for research on language models, including zero-shot NLP tasks and in-context few-shot learning NLP tasks, such as reasoning, and question answering. It can be used for advancing fairness and safety research, and understanding limitations of current large language models.",
        "languages": [
            "English",
            "Spanish",
            "Japanese",
            "Persian",
            "Hindi",
            "French",
            "Chinese",
            "Bengali",
            "Gujarati",
            "German",
            "Telugu",
            "Italian",
            "Arabic",
            "Polish",
            "Tamil",
            "Marathi",
            "Malayalam",
            "Oriya",
            "Panjabi",
            "Portuguese",
            "Urdu",
            "Galician",
            "Hebrew",
            "Korean",
            "Catalan",
            "Thai",
            "Dutch",
            "Indonesian",
            "Vietnamese",
            "Bulgarian",
            "Filipino",
            "Central Khmer",
            "Lao",
            "Turkish",
            "Russian",
            "Croatian",
            "Swedish",
            "Yoruba",
            "Kurdish",
            "Burmese",
            "Malay",
            "Czech",
            "Finnish",
            "Somali",
            "Tagalog",
            "Swahili",
            "Sinhala",
            "Kannada",
            "Zhuang",
            "Igbo",
            "Xhosa",
            "Romanian",
            "Haitian",
            "Estonian",
            "Slovak",
            "Lithuanian",
            "Greek",
            "Nepali",
            "Assamese",
            "Norwegian"
        ],
        "TuningInformation": "instruction-finetuned",
        "TrainingData": "a mixture of tasks",
        "UsesSupported": [
            "research on language models",
            "zero-shot NLP tasks",
            "in-context few-shot learning NLP tasks",
            "reasoning",
            "question answering",
            "advancing fairness and safety research",
            "understanding limitations of current large language models"
        ],
        "OptimisedFor": "low latency, real-time code completion, long inputs",
        "PromptingAdvice": "",
        "Output": "code and natural language about code",
        "PromptTuningAvailability": "true",
        "RegionalAvailability": "",
        "License": "Apache 2.0"
    },
    {
        "model_id": "google/flan-t5-xxl",
        "label": "FLAN-T5 XXL",
        "provider": "Hugging Face",
        "source": "Hugging Face",
        "functions": [
            {
                "id": "text_generation"
            }
        ],
        "short_description": "flan-t5-xxl is an 11 billion parameter model based on the Flan-T5 family.",
        "long_description": "flan-t5-xxl (11B) is an 11 billion parameter model based on the Flan-T5 family. It is a pretrained T5 - an encoder-decoder model pre-trained on a mixture of supervised / unsupervised tasks converted into a text-to-text format, and fine-tuned on the Fine-tuned Language Net (FLAN) with instructions for better zero-shot and few-shot performance.",
        "input_tier": "class_2",
        "output_tier": "class_2",
        "number_params": "11.3B",
        "min_shot_size": 0,
        "task_ids": [
            "question_answering",
            "summarization",
            "retrieval_augmented_generation",
            "classification",
            "generation",
            "extraction"
        ],
        "tasks": [
            {
                "id": "question_answering",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "summarization",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "retrieval_augmented_generation",
                "ratings": {
                    "quality": 3
                }
            },
            {
                "id": "classification",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "generation"
            },
            {
                "id": "extraction",
                "ratings": {
                    "quality": 4
                }
            }
        ],
        "model_limits": {
            "max_sequence_length": 4096
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 700
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 4095
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 4095
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2023-07-07"
            }
        ],
        "tuned_by": "none",
        "ratings": "none",
        "training_parameters": "none",
        "versions": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "Decoder-only",
        "ModelOverview": "FLAN-T5 XXL is a language model that has been fine-tuned on more than 1000 additional tasks covering multiple languages. It achieves state-of-the-art performance on several benchmarks, including 75.2% on five-shot MMLU. The model is designed for research on language models, including zero-shot NLP tasks and in-context few-shot learning NLP tasks, such as reasoning, and question answering. It can be used for advancing fairness and safety research, and understanding limitations of current large language models.",
        "languages": [
            "English",
            "German",
            "French"
        ],
        "TuningInformation": "Instruction-tuned",
        "TrainingData": "A mixture of tasks, including the tasks described in the table below (from the original paper, figure 2)",
        "UsesSupported": [
            "research on language models",
            "zero-shot NLP tasks",
            "in-context few-shot learning NLP tasks",
            "advancing fairness and safety research",
            "understanding limitations of current large language models"
        ],
        "OptimisedFor": "low latency, real-time code completion, long inputs",
        "PromptingAdvice": "Provide the model with code or natural language prompts",
        "Output": "Code and natural language about code",
        "PromptTuningAvailability": "true",
        "RegionalAvailability": "",
        "License": "Apache 2.0"
    },
    {
        "model_id": "google/flan-ul2",
        "label": "Flan-UL2",
        "provider": "Google",
        "source": "Hugging Face",
        "functions": [
            {
                "id": "text_generation"
            }
        ],
        "short_description": "flan-ul2 is an encoder decoder model based on the T5 architecture and instruction-tuned using the Fine-tuned Language Net.",
        "long_description": "flan-ul2 (20B) is an encoder decoder model based on the T5 architecture and instruction-tuned using the Fine-tuned Language Net (FLAN). Compared to the original UL2 model, flan-ul2 (20B) is more usable for few-shot in-context learning because it was trained with a three times larger receptive field. flan-ul2 (20B) outperforms flan-t5 (11B) by an overall relative improvement of +3.2%.",
        "input_tier": "class_3",
        "output_tier": "class_3",
        "number_params": "20B",
        "min_shot_size": 0,
        "task_ids": [
            "question_answering",
            "summarization",
            "retrieval_augmented_generation",
            "classification",
            "generation",
            "extraction"
        ],
        "tasks": [
            {
                "id": "question_answering",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "summarization",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "retrieval_augmented_generation",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "classification",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "generation"
            },
            {
                "id": "extraction",
                "ratings": {
                    "quality": 4
                }
            }
        ],
        "model_limits": {
            "max_sequence_length": 4096
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 700
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 4095
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 4095
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2023-07-07"
            }
        ],
        "tuned_by": "none",
        "ratings": "none",
        "training_parameters": "none",
        "versions": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "Encoder-Decoder",
        "ModelOverview": "Flan-UL2 is an encoder-decoder model based on the T5 architecture. It uses the same configuration as the UL2 model released earlier last year. It was fine-tuned using the \"Flan\" prompt tuning and dataset collection. According to the original blog, there are several notable improvements: The original UL2 model was only trained with a receptive field of 512, which made it non-ideal for N-shot prompting where N is large. The Flan-UL2 checkpoint uses a receptive field of 2048 which makes it more usable for few-shot in-context learning. The original UL2 model also had mode switch tokens that were rather mandatory to get good performance. However, they were a little cumbersome as this requires often some changes during inference or fine-tuning. In this update/change, we continue training UL2 20B for an additional 100k steps (with small batch) to forget \u201cmode tokens\u201d before applying Flan instruction tuning. This Flan-UL2 checkpoint does not require mode tokens anymore.",
        "languages": [
            "English"
        ],
        "TuningInformation": "Fine-tuned using Flan prompt tuning and dataset collection",
        "TrainingData": "C4 corpus",
        "UsesSupported": [
            "Text generation",
            "Natural language understanding",
            "Debugging"
        ],
        "OptimisedFor": "Low latency, real-time code completion, long inputs",
        "PromptingAdvice": "Provide the model with code or natural language prompts",
        "Output": "Code and natural language about code",
        "PromptTuningAvailability": "true",
        "RegionalAvailability": "",
        "License": "Apache-2.0"
    },
    {
        "model_id": "ibm/granite-13b-chat-v2",
        "label": "Granite 13 Billion chat V2",
        "provider": "IBM",
        "source": "IBM",
        "functions": [
            {
                "id": "text_generation"
            }
        ],
        "short_description": "The Granite model series is a family of IBM-trained, dense decoder-only models, which are particularly well-suited for generative tasks.",
        "long_description": "Granite models are designed to be used for a wide range of generative and non-generative tasks with appropriate prompt engineering. They employ a GPT-style decoder-only architecture, with additional innovations from IBM Research and the open community.",
        "input_tier": "class_1",
        "output_tier": "class_1",
        "number_params": "",
        "min_shot_size": 0,
        "task_ids": [
            "question_answering",
            "summarization",
            "classification",
            "generation",
            "extraction"
        ],
        "tasks": [
            {
                "id": "question_answering",
                "ratings": {
                    "quality": 3
                }
            },
            {
                "id": "summarization",
                "ratings": {
                    "quality": 2
                }
            },
            {
                "id": "retrieval_augmented_generation",
                "ratings": {
                    "quality": 2
                }
            },
            {
                "id": "classification",
                "ratings": {
                    "quality": 3
                }
            },
            {
                "id": "generation"
            },
            {
                "id": "extraction",
                "ratings": {
                    "quality": 2
                }
            }
        ],
        "model_limits": {
            "max_sequence_length": 8192
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 8191
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 8191
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 8191
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2023-12-01"
            }
        ],
        "versions": [
            {
                "version": "2.1.0",
                "available_date": "2024-02-15"
            },
            {
                "version": "2.0.0",
                "available_date": "2023-12-01"
            }
        ],
        "tuned_by": "none",
        "ratings": "none",
        "training_parameters": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "Decoder-only",
        "ModelOverview": "The Granite 13 Billion chat V2 model is a chat-focused variant initialized from the pre-trained Granite Base 13 Billion Base V2 model. It has been trained using over 2.5T tokens and is optimized for low-latency, real-time code completion, and handling long inputs. The model supports Retrieval Augmented Generation (RAG) use cases, summarization, and generation tasks. It has demonstrated the capability to support longer responses preferred in RAG-like use cases. The model is suitable for English-based closed-domain Question and Answering, summarization, and generation, extraction, and classification.",
        "languages": [
            "English"
        ],
        "TuningInformation": "Novel alignment technique for LLMs using large-scale targeted alignment for a generalist LLM",
        "TrainingData": "2.5 Trillion tokens of IBM's curated pre-training dataset",
        "UsesSupported": [
            "English-based closed-domain Question and Answering",
            "summarization",
            "generation",
            "extraction",
            "classification"
        ],
        "OptimisedFor": "Low-latency, real-time code completion, and handling long inputs",
        "PromptingAdvice": "",
        "Output": "",
        "PromptTuningAvailability": "",
        "RegionalAvailability": "",
        "License": "Available only through IBM products and offerings"
    },
    {
        "model_id": "ibm/granite-13b-instruct-v2",
        "label": "Granite 13B Instruct V2.0",
        "provider": "IBM",
        "source": "IBM",
        "functions": [
            {
                "id": "prompt_tune_inferable"
            },
            {
                "id": "prompt_tune_trainable"
            },
            {
                "id": "text_generation"
            }
        ],
        "short_description": "The Granite model series is a family of IBM-trained, dense decoder-only models, which are particularly well-suited for generative tasks.",
        "long_description": "Granite models are designed to be used for a wide range of generative and non-generative tasks with appropriate prompt engineering. They employ a GPT-style decoder-only architecture, with additional innovations from IBM Research and the open community.",
        "input_tier": "class_1",
        "output_tier": "class_1",
        "number_params": "",
        "min_shot_size": 0,
        "task_ids": [
            "question_answering",
            "summarization",
            "classification",
            "generation",
            "extraction"
        ],
        "tasks": [
            {
                "id": "question_answering",
                "ratings": {
                    "quality": 3
                }
            },
            {
                "id": "summarization",
                "ratings": {
                    "quality": 2
                },
                "tags": [
                    "function_prompt_tune_trainable"
                ],
                "training_parameters": {
                    "init_method": {
                        "supported": [
                            "random",
                            "text"
                        ],
                        "default": "text"
                    },
                    "init_text": {
                        "default": "Please write a summary highlighting the main points of the following text:"
                    },
                    "num_virtual_tokens": {
                        "supported": [
                            20,
                            50,
                            100
                        ],
                        "default": 100
                    },
                    "num_epochs": {
                        "default": 40,
                        "min": 1,
                        "max": 50
                    },
                    "verbalizer": {
                        "default": "Please write a summary highlighting the main points of the following text: {{input}}"
                    },
                    "batch_size": {
                        "default": 8,
                        "min": 1,
                        "max": 16
                    },
                    "max_input_tokens": {
                        "default": 256,
                        "min": 1,
                        "max": 1024
                    },
                    "max_output_tokens": {
                        "default": 128,
                        "min": 1,
                        "max": 512
                    },
                    "torch_dtype": {
                        "default": "bfloat16"
                    },
                    "accumulate_steps": {
                        "default": 1,
                        "min": 1,
                        "max": 128
                    },
                    "learning_rate": {
                        "default": 0.0002,
                        "min": 1e-05,
                        "max": 0.5
                    }
                }
            },
            {
                "id": "retrieval_augmented_generation",
                "ratings": {
                    "quality": 2
                }
            },
            {
                "id": "classification",
                "ratings": {
                    "quality": 3
                },
                "tags": [
                    "function_prompt_tune_trainable"
                ],
                "training_parameters": {
                    "init_method": {
                        "supported": [
                            "random",
                            "text"
                        ],
                        "default": "text"
                    },
                    "init_text": {
                        "default": "Classify the text:"
                    },
                    "num_virtual_tokens": {
                        "supported": [
                            20,
                            50,
                            100
                        ],
                        "default": 100
                    },
                    "num_epochs": {
                        "default": 20,
                        "min": 1,
                        "max": 50
                    },
                    "verbalizer": {
                        "default": "Input: {{input}} Output:"
                    },
                    "batch_size": {
                        "default": 8,
                        "min": 1,
                        "max": 16
                    },
                    "max_input_tokens": {
                        "default": 256,
                        "min": 1,
                        "max": 1024
                    },
                    "max_output_tokens": {
                        "default": 128,
                        "min": 1,
                        "max": 512
                    },
                    "torch_dtype": {
                        "default": "bfloat16"
                    },
                    "accumulate_steps": {
                        "default": 32,
                        "min": 1,
                        "max": 128
                    },
                    "learning_rate": {
                        "default": 0.0006,
                        "min": 1e-05,
                        "max": 0.5
                    }
                }
            },
            {
                "id": "generation",
                "tags": [
                    "function_prompt_tune_trainable"
                ],
                "training_parameters": {
                    "init_method": {
                        "supported": [
                            "random",
                            "text"
                        ],
                        "default": "text"
                    },
                    "init_text": {
                        "default": "text"
                    },
                    "num_virtual_tokens": {
                        "supported": [
                            20,
                            50,
                            100
                        ],
                        "default": 100
                    },
                    "num_epochs": {
                        "default": 20,
                        "min": 1,
                        "max": 50
                    },
                    "verbalizer": {
                        "default": "{{input}}"
                    },
                    "batch_size": {
                        "default": 16,
                        "min": 1,
                        "max": 16
                    },
                    "max_input_tokens": {
                        "default": 256,
                        "min": 1,
                        "max": 1024
                    },
                    "max_output_tokens": {
                        "default": 128,
                        "min": 1,
                        "max": 512
                    },
                    "torch_dtype": {
                        "default": "bfloat16"
                    },
                    "accumulate_steps": {
                        "default": 16,
                        "min": 1,
                        "max": 128
                    },
                    "learning_rate": {
                        "default": 0.0002,
                        "min": 1e-05,
                        "max": 0.5
                    }
                }
            },
            {
                "id": "extraction",
                "ratings": {
                    "quality": 2
                }
            }
        ],
        "model_limits": {
            "max_sequence_length": 8192,
            "training_data_max_records": 10000
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 8191
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 8191
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 8191
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2023-12-01"
            }
        ],
        "training_parameters": {
            "init_method": {
                "supported": [
                    "random",
                    "text"
                ],
                "default": "random"
            },
            "init_text": {
                "default": "text"
            },
            "num_virtual_tokens": {
                "supported": [
                    20,
                    50,
                    100
                ],
                "default": 100
            },
            "num_epochs": {
                "default": 20,
                "min": 1,
                "max": 50
            },
            "verbalizer": {
                "default": "{{input}}"
            },
            "batch_size": {
                "default": 16,
                "min": 1,
                "max": 16
            },
            "max_input_tokens": {
                "default": 256,
                "min": 1,
                "max": 1024
            },
            "max_output_tokens": {
                "default": 128,
                "min": 1,
                "max": 512
            },
            "torch_dtype": {
                "default": "bfloat16"
            },
            "accumulate_steps": {
                "default": 16,
                "min": 1,
                "max": 128
            },
            "learning_rate": {
                "default": 0.0002,
                "min": 1e-05,
                "max": 0.5
            }
        },
        "tuned_by": "none",
        "ratings": "none",
        "versions": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "Decoder-only",
        "ModelOverview": "The Granite 13B Instruct V2.0 model is a large decoder-only transformer model trained with a massive dataset of 2.5 trillion tokens. It is optimized for low-latency, real-time code completion, and handling long inputs. The model supports multiple programming languages, including Python, C++, Java, PHP, Typescript, C#, and Bash. It is designed for complex code-related tasks, offering state-of-the-art performance in various programming languages. The model is intended for English-based classification, extraction, and summarization tasks, and is primarily targeted towards IBM Enterprise clients.",
        "languages": [
            "English"
        ],
        "TuningInformation": "Instruction-tuned",
        "TrainingData": "IBM Data Pile Version 0.4 and 0.3",
        "UsesSupported": [
            "Classification",
            "Extraction",
            "Summarization"
        ],
        "OptimisedFor": "Low-latency, real-time code completion, and handling long inputs",
        "PromptingAdvice": "No special prompt is required for this model",
        "Output": "Code and natural language about code",
        "PromptTuningAvailability": "false",
        "RegionalAvailability": "",
        "License": "Available only through IBM products and offerings"
    },
    {
        "model_id": "ibm/granite-20b-code-instruct",
        "label": "Granite-20B-Code-Instruct",
        "provider": "IBM",
        "source": "IBM",
        "functions": [
            {
                "id": "text_generation"
            }
        ],
        "short_description": "The Granite model series is a family of IBM-trained, dense decoder-only models, which are particularly well-suited for generative tasks.",
        "long_description": "Granite models are designed to be used for a wide range of generative and non-generative tasks with appropriate prompt engineering. They employ a GPT-style decoder-only architecture, with additional innovations from IBM Research and the open community.",
        "input_tier": "class_1",
        "output_tier": "class_1",
        "number_params": "20B",
        "min_shot_size": 1,
        "task_ids": [
            "question_answering",
            "summarization",
            "classification",
            "generation",
            "extraction"
        ],
        "tasks": [
            {
                "id": "question_answering"
            },
            {
                "id": "summarization"
            },
            {
                "id": "classification"
            },
            {
                "id": "generation"
            },
            {
                "id": "extraction"
            }
        ],
        "model_limits": {
            "max_sequence_length": 8192
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 4096
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 4096
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 4096
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2024-05-06"
            }
        ],
        "tuned_by": "none",
        "ratings": "none",
        "training_parameters": "none",
        "versions": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "",
        "ModelOverview": "Granite-20B-Code-Instruct is a 20B parameter model fine-tuned from Granite-20B-Code-Base on a combination of permissively licensed instruction data to enhance instruction following capabilities including logical reasoning and problem-solving skills. The model is designed to respond to coding-related instructions and can be used to build coding assistants.",
        "languages": [
            "Python",
            "JavaScript",
            "Java",
            "Go",
            "C++",
            "Rust"
        ],
        "TuningInformation": "",
        "TrainingData": "Code Commits Datasets, Math Datasets, Code Instruction Datasets, Language Instruction Datasets",
        "UsesSupported": [
            "Generation",
            "logical reasoning",
            "problem-solving skills"
        ],
        "OptimisedFor": "",
        "PromptingAdvice": "",
        "Output": "",
        "PromptTuningAvailability": "",
        "RegionalAvailability": "",
        "License": "Apache 2.0"
    },
    {
        "model_id": "ibm/granite-20b-multilingual",
        "label": "Granite 20 Billion Multilingual Model",
        "provider": "IBM",
        "source": "IBM",
        "functions": [
            {
                "id": "text_generation"
            }
        ],
        "short_description": "The Granite model series is a family of IBM-trained, dense decoder-only models, which are particularly well-suited for generative tasks.",
        "long_description": "Granite models are designed to be used for a wide range of generative and non-generative tasks with appropriate prompt engineering. They employ a GPT-style decoder-only architecture, with additional innovations from IBM Research and the open community.",
        "input_tier": "class_1",
        "output_tier": "class_1",
        "number_params": "",
        "min_shot_size": 1,
        "task_ids": [
            "question_answering",
            "summarization",
            "retrieval_augmented_generation",
            "classification",
            "generation",
            "extraction"
        ],
        "tasks": [
            {
                "id": "question_answering",
                "ratings": {
                    "quality": 3
                }
            },
            {
                "id": "summarization",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "retrieval_augmented_generation",
                "ratings": {
                    "quality": 3
                }
            },
            {
                "id": "classification",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "generation"
            },
            {
                "id": "extraction",
                "ratings": {
                    "quality": 4
                }
            }
        ],
        "model_limits": {
            "max_sequence_length": 8192
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 4096
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 4096
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 4096
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2024-03-14"
            }
        ],
        "versions": [
            {
                "version": "1.1.0",
                "available_date": "2024-04-18"
            },
            {
                "version": "1.0.0",
                "available_date": "2024-03-14"
            }
        ],
        "tuned_by": "none",
        "ratings": "none",
        "training_parameters": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "Decoder-only",
        "ModelOverview": "The Granite 20 Billion Multilingual Model is a large decoder-only transformer model trained using over 2.6 trillion tokens and further fine-tuned using a collection of instruction-tuning datasets. The model underwent extended pre-training using multilingual common crawl data resulting in a model that works with English, German, Spanish, French, and Portuguese. It leverages a new training paradigm for Large Language Models (LLMs) where instead of large-scale pre-training and small-to-medium scale alignment, we do very large-scale targeted alignment for a generalist LLM. The model is designed for closed-domain Question and Answering, summarization, generation, extraction, and classification tasks.",
        "languages": [
            "English",
            "German",
            "Spanish",
            "French",
            "Portuguese"
        ],
        "TuningInformation": "Instruction-tuned",
        "TrainingData": "2.6 trillion tokens of IBM's curated pre-training dataset",
        "UsesSupported": [
            "closed-domain Question and Answering",
            "summarization",
            "generation",
            "extraction",
            "classification"
        ],
        "OptimisedFor": "Low latency, real-time code completion, long inputs",
        "PromptingAdvice": "",
        "Output": "",
        "PromptTuningAvailability": "false",
        "RegionalAvailability": "",
        "License": "Available only through IBM products and offerings"
    },
    {
        "model_id": "ibm/granite-34b-code-instruct",
        "label": "Granite-34B-Code-Instruct",
        "provider": "IBM Research",
        "source": "IBM",
        "functions": [
            {
                "id": "text_generation"
            }
        ],
        "short_description": "The Granite model series is a family of IBM-trained, dense decoder-only models, which are particularly well-suited for generative tasks.",
        "long_description": "Granite models are designed to be used for a wide range of generative and non-generative tasks with appropriate prompt engineering. They employ a GPT-style decoder-only architecture, with additional innovations from IBM Research and the open community.",
        "input_tier": "class_1",
        "output_tier": "class_1",
        "number_params": "34B",
        "min_shot_size": 1,
        "task_ids": [
            "question_answering",
            "summarization",
            "classification",
            "generation",
            "extraction"
        ],
        "tasks": [
            {
                "id": "question_answering"
            },
            {
                "id": "summarization"
            },
            {
                "id": "classification"
            },
            {
                "id": "generation"
            },
            {
                "id": "extraction"
            }
        ],
        "model_limits": {
            "max_sequence_length": 16384
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 8192
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 8192
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 8192
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2024-05-06"
            }
        ],
        "tuned_by": "none",
        "ratings": "none",
        "training_parameters": "none",
        "versions": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "Decoder-only",
        "ModelOverview": "Granite-34B-Code-Instruct is a 34B parameter model fine-tuned from Granite-34B-Code-Base on a combination of permissively licensed instruction data to enhance instruction following capabilities including logical reasoning and problem-solving skills. It is designed to respond to coding-related instructions and can be used to build coding assistants. The model is trained on a variety of datasets including code commits, math datasets, code instruction datasets, and language instruction datasets. It has a context length of 100000 tokens and is optimized for low latency, real-time code completion, and handling long inputs.",
        "languages": [
            "Python",
            "C++",
            "Java",
            "PHP",
            "Typescript (Javascript)",
            "C#"
        ],
        "TuningInformation": "Fine-tuned on a combination of permissively licensed instruction data",
        "TrainingData": "Code commits datasets, Math datasets, Code instruction datasets, Language instruction datasets",
        "UsesSupported": [
            "Code generation",
            "logical reasoning",
            "problem-solving skills"
        ],
        "OptimisedFor": "Low latency, real-time code completion, long inputs",
        "PromptingAdvice": "Provide the model with coding-related instructions",
        "Output": "Code and natural language about code",
        "PromptTuningAvailability": "true",
        "RegionalAvailability": "",
        "License": "Apache 2.0"
    },
    {
        "model_id": "ibm/granite-3b-code-instruct",
        "label": "Granite-3B-Code-Instruct",
        "provider": "IBM Research",
        "source": "IBM",
        "functions": [
            {
                "id": "text_generation"
            }
        ],
        "short_description": "The Granite model series is a family of IBM-trained, dense decoder-only models, which are particularly well-suited for generative tasks.",
        "long_description": "Granite models are designed to be used for a wide range of generative and non-generative tasks with appropriate prompt engineering. They employ a GPT-style decoder-only architecture, with additional innovations from IBM Research and the open community.",
        "input_tier": "class_1",
        "output_tier": "class_1",
        "number_params": "3B",
        "min_shot_size": 1,
        "task_ids": [
            "question_answering",
            "summarization",
            "classification",
            "generation",
            "extraction"
        ],
        "tasks": [
            {
                "id": "question_answering"
            },
            {
                "id": "summarization"
            },
            {
                "id": "classification"
            },
            {
                "id": "generation"
            },
            {
                "id": "extraction"
            }
        ],
        "model_limits": {
            "max_sequence_length": 8192
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 4096
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 4096
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 4096
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2024-05-09"
            }
        ],
        "tuned_by": "none",
        "ratings": "none",
        "training_parameters": "none",
        "versions": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "",
        "ModelOverview": "Granite-3B-Code-Instruct is a 3B parameter model fine-tuned from Granite-3B-Code-Base on a combination of permissively licensed instruction data to enhance instruction following capabilities including logical reasoning and problem-solving skills. It is designed to respond to coding-related instructions and can be used to build coding assistants. The model is trained on a variety of datasets including code commits, math datasets, code instruction datasets, and language instruction datasets. It has a context length of unknown tokens and is optimized for low-latency, real-time code completion, and handling long inputs.",
        "languages": [
            "Python",
            "C++",
            "Java",
            "PHP",
            "Typescript (Javascript)",
            "C#"
        ],
        "TuningInformation": "Fine-tuned from Granite-3B-Code-Base",
        "TrainingData": "Code commits datasets, Math datasets, Code instruction datasets, Language instruction datasets",
        "UsesSupported": [
            "Code generation",
            "logical reasoning",
            "problem-solving skills"
        ],
        "OptimisedFor": "Low-latency, real-time code completion, handling long inputs",
        "PromptingAdvice": "",
        "Output": "Code and natural language about code",
        "PromptTuningAvailability": "",
        "RegionalAvailability": "",
        "License": "Apache 2.0"
    },
    {
        "model_id": "ibm/granite-7b-lab",
        "label": "Granite-7b-lab",
        "provider": "IBM Research",
        "source": "IBM",
        "functions": [
            {
                "id": "text_generation"
            }
        ],
        "short_description": "The Granite model series is a family of IBM-trained, dense decoder-only models, which are particularly well-suited for generative tasks.",
        "long_description": "Granite models are designed to be used for a wide range of generative and non-generative tasks with appropriate prompt engineering. They employ a GPT-style decoder-only architecture, with additional innovations from IBM Research and the open community.",
        "input_tier": "class_1",
        "output_tier": "class_1",
        "number_params": "6.74B",
        "min_shot_size": 1,
        "task_ids": [
            "question_answering",
            "summarization",
            "retrieval_augmented_generation",
            "classification",
            "generation",
            "extraction"
        ],
        "tasks": [
            {
                "id": "question_answering"
            },
            {
                "id": "summarization"
            },
            {
                "id": "retrieval_augmented_generation"
            },
            {
                "id": "classification"
            },
            {
                "id": "generation"
            },
            {
                "id": "extraction"
            }
        ],
        "model_limits": {
            "max_sequence_length": 4096
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 4095
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 4095
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 4095
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2024-04-18"
            }
        ],
        "tuned_by": "none",
        "ratings": "none",
        "training_parameters": "none",
        "versions": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "",
        "ModelOverview": "Granite-7b-lab is a derivative model trained with the LAB methodology, using Mixtral-8x7B-Instruct as a teacher model. It is a cautious assistant that carefully follows instructions and is helpful and harmless. The model is designed to promote positive behavior and follows ethical guidelines. It is primarily trained on English language and has a model size of 6.74B parameters. The model is optimized for low latency, real-time code completion, and handling long inputs. It supports code generation, natural language understanding related to code, and debugging.",
        "languages": [
            "English"
        ],
        "TuningInformation": "Large-scale Alignment for chatBots (LAB)",
        "TrainingData": "Synthetic data generated by Mixtral-8x7B-Instruct",
        "UsesSupported": [
            "code generation",
            "natural language about code",
            "debugging"
        ],
        "OptimisedFor": "low latency, real-time code completion, long inputs",
        "PromptingAdvice": "Use the system prompt employed during the model's training for optimal inference performance.",
        "Output": "Code and natural language about code",
        "PromptTuningAvailability": "false",
        "RegionalAvailability": "",
        "License": "Apache 2.0"
    },
    {
        "model_id": "ibm/granite-8b-code-instruct",
        "label": "Granite-8B-Code-Instruct",
        "provider": "IBM Research",
        "source": "IBM",
        "functions": [
            {
                "id": "text_generation"
            }
        ],
        "short_description": "The Granite model series is a family of IBM-trained, dense decoder-only models, which are particularly well-suited for generative tasks.",
        "long_description": "Granite models are designed to be used for a wide range of generative and non-generative tasks with appropriate prompt engineering. They employ a GPT-style decoder-only architecture, with additional innovations from IBM Research and the open community.",
        "input_tier": "class_1",
        "output_tier": "class_1",
        "number_params": "8B",
        "min_shot_size": 1,
        "task_ids": [
            "question_answering",
            "summarization",
            "classification",
            "generation",
            "extraction"
        ],
        "tasks": [
            {
                "id": "question_answering"
            },
            {
                "id": "summarization"
            },
            {
                "id": "classification"
            },
            {
                "id": "generation"
            },
            {
                "id": "extraction"
            }
        ],
        "model_limits": {
            "max_sequence_length": 8192
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 4096
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 4096
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 4096
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2024-05-09"
            }
        ],
        "tuned_by": "none",
        "ratings": "none",
        "training_parameters": "none",
        "versions": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "",
        "ModelOverview": "Granite-8B-Code-Instruct is a 8B parameter model fine-tuned from Granite-8B-Code-Base on a combination of permissively licensed instruction data to enhance instruction following capabilities including logical reasoning and problem-solving skills. It is designed to respond to coding-related instructions and can be used to build coding assistants. The model is trained on a variety of datasets including code commits, math datasets, code instruction datasets, and language instruction datasets. It has a context length of 100000 tokens and is optimized for low latency, real-time code completion, and handling long inputs.",
        "languages": [
            "Python",
            "C++",
            "Java",
            "PHP",
            "Typescript (Javascript)",
            "C#"
        ],
        "TuningInformation": "Fine-tuned from Granite-8B-Code-Base",
        "TrainingData": "Code commits datasets, Math datasets, Code instruction datasets, Language instruction datasets",
        "UsesSupported": [
            "Code generation",
            "logical reasoning",
            "problem-solving skills"
        ],
        "OptimisedFor": "Low latency, real-time code completion, long inputs",
        "PromptingAdvice": "Provide the model with coding-related instructions",
        "Output": "Code and natural language about code",
        "PromptTuningAvailability": "true",
        "RegionalAvailability": "",
        "License": "Apache 2.0"
    },
    {
        "model_id": "meta-llama/llama-2-13b-chat",
        "label": "Llama 2",
        "provider": "Meta",
        "source": "Hugging Face",
        "functions": [
            {
                "id": "prompt_tune_inferable"
            },
            {
                "id": "prompt_tune_trainable"
            },
            {
                "id": "text_generation"
            }
        ],
        "short_description": "Llama-2-13b-chat is an auto-regressive language model that uses an optimized transformer architecture.",
        "long_description": "Llama-2-13b-chat is a pretrained and fine-tuned generative text model with 13 billion parameters, optimized for dialogue use cases.",
        "input_tier": "class_1",
        "output_tier": "class_1",
        "number_params": "13B",
        "min_shot_size": 1,
        "task_ids": [
            "question_answering",
            "summarization",
            "retrieval_augmented_generation",
            "classification",
            "generation",
            "code",
            "extraction"
        ],
        "tasks": [
            {
                "id": "question_answering",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "summarization",
                "ratings": {
                    "quality": 3
                },
                "tags": [
                    "function_prompt_tune_trainable"
                ]
            },
            {
                "id": "retrieval_augmented_generation",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "classification",
                "ratings": {
                    "quality": 4
                },
                "tags": [
                    "function_prompt_tune_trainable"
                ]
            },
            {
                "id": "generation",
                "tags": [
                    "function_prompt_tune_trainable"
                ]
            },
            {
                "id": "code"
            },
            {
                "id": "extraction",
                "ratings": {
                    "quality": 4
                }
            }
        ],
        "model_limits": {
            "max_sequence_length": 4096,
            "training_data_max_records": 10000
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 2048
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 4095
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 4095
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2023-11-09"
            }
        ],
        "training_parameters": {
            "init_method": {
                "supported": [
                    "random",
                    "text"
                ],
                "default": "random"
            },
            "init_text": {
                "default": "text"
            },
            "num_virtual_tokens": {
                "supported": [
                    20,
                    50,
                    100
                ],
                "default": 100
            },
            "num_epochs": {
                "default": 20,
                "min": 1,
                "max": 50
            },
            "verbalizer": {
                "default": "{{input}}"
            },
            "batch_size": {
                "default": 8,
                "min": 1,
                "max": 16
            },
            "max_input_tokens": {
                "default": 256,
                "min": 1,
                "max": 1024
            },
            "max_output_tokens": {
                "default": 128,
                "min": 1,
                "max": 512
            },
            "torch_dtype": {
                "default": "bfloat16"
            },
            "accumulate_steps": {
                "default": 16,
                "min": 1,
                "max": 128
            },
            "learning_rate": {
                "default": 0.002,
                "min": 1e-05,
                "max": 0.5
            }
        },
        "tuned_by": "none",
        "ratings": "none",
        "versions": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "Decoder-only",
        "ModelOverview": "Llama 2 is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. This is the repository for the 13B fine-tuned model, optimized for dialogue use cases. The tuned models are intended for assistant-like chat, whereas pretrained models can be adapted for a variety of natural language generation tasks.",
        "languages": [
            "English"
        ],
        "TuningInformation": "instruction-tuned",
        "TrainingData": "Code and code-related data",
        "UsesSupported": [
            "Code generation",
            "natural language about code",
            "debugging"
        ],
        "OptimisedFor": "Low latency, real-time code completion, long inputs",
        "PromptingAdvice": "Provide the model with code or natural language prompts",
        "Output": "Code and natural language about code",
        "PromptTuningAvailability": "true",
        "RegionalAvailability": "Not specified",
        "License": "Community license as Llama 2"
    },
    {
        "model_id": "meta-llama/llama-2-70b-chat",
        "label": "Code Llama",
        "provider": "Meta",
        "source": "Hugging Face",
        "functions": [
            {
                "id": "text_generation"
            }
        ],
        "short_description": "Llama-2-70b-chat is an auto-regressive language model that uses an optimized transformer architecture.",
        "long_description": "Llama-2-70b-chat is a pretrained and fine-tuned generative text model with 70 billion parameters, optimized for dialogue use cases.",
        "input_tier": "class_2",
        "output_tier": "class_2",
        "number_params": "70B",
        "min_shot_size": 1,
        "task_ids": [
            "question_answering",
            "summarization",
            "retrieval_augmented_generation",
            "classification",
            "generation",
            "code",
            "extraction"
        ],
        "tasks": [
            {
                "id": "question_answering",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "summarization",
                "ratings": {
                    "quality": 3
                }
            },
            {
                "id": "retrieval_augmented_generation",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "classification",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "generation"
            },
            {
                "id": "code"
            },
            {
                "id": "extraction",
                "ratings": {
                    "quality": 4
                }
            }
        ],
        "model_limits": {
            "max_sequence_length": 4096
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 900
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 4095
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 4095
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2023-09-07"
            }
        ],
        "tuned_by": "none",
        "ratings": "none",
        "training_parameters": "none",
        "versions": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "Decoder-only",
        "ModelOverview": "Code Llama is an advanced series of code language models designed by Meta. These models come in various sizes, with the largest being 70B parameters. Code Llama is optimized for real-time code completion, generation, and debugging, supporting multiple programming languages with a large context window.",
        "languages": [
            "Python",
            "C++",
            "Java",
            "PHP",
            "Typescript (Javascript)",
            "C#",
            "Bash"
        ],
        "TuningInformation": "Instruction-tuned",
        "TrainingData": "Code and code-related data",
        "UsesSupported": [
            "Code generation",
            "natural language about code",
            "debugging"
        ],
        "OptimisedFor": "Low latency, real-time code completion, long inputs",
        "PromptingAdvice": "Provide the model with code or natural language prompts",
        "Output": "Code and natural language about code",
        "PromptTuningAvailability": "true",
        "RegionalAvailability": "Not specified",
        "License": "Community license as Llama 2"
    },
    {
        "model_id": "meta-llama/llama-3-1-8b-instruct",
        "label": "Meta-Llama-3.1-8B-Instruct",
        "provider": "Meta",
        "source": "Hugging Face",
        "functions": [
            {
                "id": "text_generation"
            }
        ],
        "short_description": "Llama-3-1-8b-instruct is an auto-regressive language model that uses an optimized transformer architecture.",
        "long_description": "Llama-3-1-8b-instruct is a pretrained and fine-tuned generative text model with 8 billion parameters, optimized for multilingual dialogue use cases and code output.",
        "input_tier": "class_1",
        "output_tier": "class_1",
        "number_params": "8B",
        "min_shot_size": 1,
        "task_ids": [
            "question_answering",
            "summarization",
            "retrieval_augmented_generation",
            "classification",
            "generation",
            "code",
            "extraction"
        ],
        "tasks": [
            {
                "id": "question_answering",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "summarization",
                "ratings": {
                    "quality": 3
                }
            },
            {
                "id": "retrieval_augmented_generation",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "classification",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "generation"
            },
            {
                "id": "code"
            },
            {
                "id": "extraction",
                "ratings": {
                    "quality": 4
                }
            }
        ],
        "model_limits": {
            "max_sequence_length": 131072
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 4096
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 4096
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 4096
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2024-08-01"
            }
        ],
        "versions": [
            {
                "version": "3.1.0",
                "available_date": "2024-08-01"
            }
        ],
        "tuned_by": "none",
        "ratings": "none",
        "training_parameters": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "Decoder-only",
        "ModelOverview": "Meta-Llama-3.1-8B-Instruct is a large language model designed for real-time code completion, generation, and debugging. It is optimized for low latency, real-time code completion, and handling long inputs. The model is trained on a comprehensive dataset of code and code-related data and is available in multiple sizes, including 8B, 70B, and 405B parameters. It supports multiple programming languages, including Python, C++, Java, PHP, Typescript, C#, and Bash.",
        "languages": [
            "Python",
            "C++",
            "Java",
            "PHP",
            "Typescript (Javascript)",
            "C#",
            "Bash"
        ],
        "TuningInformation": "Instruction-tuned",
        "TrainingData": "Code and code-related data",
        "UsesSupported": [
            "Code generation",
            "natural language about code",
            "debugging"
        ],
        "OptimisedFor": "Low latency, real-time code completion, long inputs",
        "PromptingAdvice": "Provide the model with code or natural language prompts",
        "Output": "Code and natural language about code",
        "PromptTuningAvailability": "true",
        "RegionalAvailability": "Not specified",
        "License": "Community license as Llama 2"
    },
    {
        "model_id": "meta-llama/llama-3-405b-instruct",
        "label": "Meta-Llama-3.1-405B",
        "provider": "Meta",
        "source": "Meta",
        "functions": [
            {
                "id": "text_generation"
            }
        ],
        "short_description": "Llama-3-405b-instruct is Meta's largest open-sourced foundation model to date, with 405 billion parameters, optimized for dialogue use cases",
        "long_description": "Llama-3-405b-instruct is Meta's largest open-sourced foundation model to date, with 405 billion parameters, optimized for dialogue use cases. It's also the largest open-sourced model ever released. It can also be used as a synthetic data generator, post-training data ranking judge, or model teacher/supervisor that can improve specialized capabilities in derivative, more inference friendly models.",
        "input_tier": "class_3",
        "output_tier": "class_7",
        "number_params": "405B",
        "min_shot_size": 0,
        "task_ids": [
            "question_answering",
            "summarization",
            "retrieval_augmented_generation",
            "classification",
            "generation",
            "code",
            "extraction"
        ],
        "tasks": [
            {
                "id": "question_answering",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "summarization",
                "ratings": {
                    "quality": 3
                }
            },
            {
                "id": "retrieval_augmented_generation",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "classification",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "generation"
            },
            {
                "id": "code"
            },
            {
                "id": "extraction",
                "ratings": {
                    "quality": 4
                }
            }
        ],
        "limits": {
            "lite": {
                "call_time": "5m0s"
            },
            "v2-professional": {
                "call_time": "10m0s"
            },
            "v2-standard": {
                "call_time": "10m0s"
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2024-07-23"
            }
        ],
        "versions": [
            {
                "version": "3.1.0",
                "available_date": "2024-07-23"
            }
        ],
        "tuned_by": "none",
        "ratings": "none",
        "model_limits": "none",
        "training_parameters": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "Decoder-only",
        "ModelOverview": "Meta-Llama-3.1-405B is a foundational large language model designed by Meta. It is a collection of multilingual large language models (LLMs) that can be used for a variety of natural language generation tasks. The model is trained on a massive dataset of publicly available online data and fine-tuned on a range of instruction datasets. It is optimized for multilingual dialogue use cases and outperforms many available open-source and closed chat models on common industry benchmarks. The model has a size of 405B parameters and is designed to be used for commercial and research purposes.",
        "languages": [
            "English",
            "German",
            "French",
            "Italian",
            "Portuguese",
            "Hindi",
            "Spanish",
            "Thai"
        ],
        "TuningInformation": "Instruction-tuned",
        "TrainingData": "Code and code-related data",
        "UsesSupported": [
            "Code generation",
            "natural language about code",
            "debugging"
        ],
        "OptimisedFor": "Low latency, real-time code completion, long inputs",
        "PromptingAdvice": "Provide the model with code or natural language prompts",
        "Output": "Code and natural language about code",
        "PromptTuningAvailability": "true",
        "RegionalAvailability": "Not specified",
        "License": "Community license as Llama 2"
    },
    {
        "model_id": "meta-llama/llama-3-70b-instruct",
        "label": "Meta Llama 3",
        "provider": "Meta",
        "source": "Hugging Face",
        "functions": [
            {
                "id": "text_generation"
            }
        ],
        "short_description": "Llama-3-70b-instruct is an auto-regressive language model that uses an optimized transformer architecture.",
        "long_description": "Llama-3-70b-instruct is a pretrained and fine-tuned generative text model with 70 billion parameters, optimized for dialogue use cases.",
        "input_tier": "class_2",
        "output_tier": "class_2",
        "number_params": "70B",
        "min_shot_size": 1,
        "task_ids": [
            "question_answering",
            "summarization",
            "retrieval_augmented_generation",
            "classification",
            "generation",
            "code",
            "extraction"
        ],
        "tasks": [
            {
                "id": "question_answering",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "summarization",
                "ratings": {
                    "quality": 3
                }
            },
            {
                "id": "retrieval_augmented_generation",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "classification",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "generation"
            },
            {
                "id": "code"
            },
            {
                "id": "extraction",
                "ratings": {
                    "quality": 4
                }
            }
        ],
        "model_limits": {
            "max_sequence_length": 8192
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 4096
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 4096
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 4096
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2024-04-18"
            }
        ],
        "tuned_by": "none",
        "ratings": "none",
        "training_parameters": "none",
        "versions": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "Decoder-only",
        "ModelOverview": "Meta Llama 3 is a large language model developed by Meta AI. It is a collection of pre-trained instruction-tuned generative text models that can be used for a variety of natural language generation tasks. The model is designed to be highly flexible and can be fine-tuned for specific use cases. It has been trained on a massive dataset of 15 trillion tokens and has achieved state-of-the-art results on several benchmarks. The model is available in two sizes: 8B and 70B parameters. The 70B parameter model is the largest and most powerful model, capable of generating high-quality text in a variety of styles and formats.",
        "languages": [
            "English",
            "C++",
            "Java",
            "PHP",
            "Typescript (Javascript)",
            "C#",
            "Bash"
        ],
        "TuningInformation": "Instruction-tuned",
        "TrainingData": "Code and code-related data",
        "UsesSupported": [
            "Code generation",
            "natural language about code",
            "debugging"
        ],
        "OptimisedFor": "Low latency, real-time code completion, long inputs",
        "PromptingAdvice": "Provide the model with code or natural language prompts",
        "Output": "Code and natural language about code",
        "PromptTuningAvailability": "true",
        "RegionalAvailability": "Not specified",
        "License": "Community license as Llama 2"
    },
    {
        "model_id": "meta-llama/llama-3-8b-instruct",
        "label": "Meta-Llama-3-8B-Instruct",
        "provider": "Meta",
        "source": "Hugging Face",
        "functions": [
            {
                "id": "text_generation"
            }
        ],
        "short_description": "Llama-3-8b-instruct is an auto-regressive language model that uses an optimized transformer architecture.",
        "long_description": "Llama-3-8b-instruct is a pretrained and fine-tuned generative text model with 8 billion parameters, optimized for dialogue use cases.",
        "input_tier": "class_1",
        "output_tier": "class_1",
        "number_params": "8B",
        "min_shot_size": 1,
        "task_ids": [
            "question_answering",
            "summarization",
            "retrieval_augmented_generation",
            "classification",
            "generation",
            "code",
            "extraction"
        ],
        "tasks": [
            {
                "id": "question_answering",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "summarization",
                "ratings": {
                    "quality": 3
                }
            },
            {
                "id": "retrieval_augmented_generation",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "classification",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "generation"
            },
            {
                "id": "code"
            },
            {
                "id": "extraction",
                "ratings": {
                    "quality": 4
                }
            }
        ],
        "model_limits": {
            "max_sequence_length": 8192
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 4096
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 4096
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 4096
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2024-04-18"
            }
        ],
        "tuned_by": "none",
        "ratings": "none",
        "training_parameters": "none",
        "versions": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "Decoder-only",
        "ModelOverview": "Meta Llama 3 is a family of large language models developed by Meta. The 8B version is an instruction-tuned model optimized for dialogue use cases, outperforming many available open-source chat models on common industry benchmarks. It is designed to be accessible and helpful, with a focus on safety and responsibility. The model has been extensively tested and fine-tuned to reduce residual risks and ensure it is safe for use.",
        "languages": [
            "English"
        ],
        "TuningInformation": "Instruction-tuned",
        "TrainingData": "Code and code-related data",
        "UsesSupported": [
            "Code generation",
            "natural language about code",
            "debugging"
        ],
        "OptimisedFor": "Low latency, real-time code completion, long inputs",
        "PromptingAdvice": "Provide the model with code or natural language prompts",
        "Output": "Code and natural language about code",
        "PromptTuningAvailability": "true",
        "RegionalAvailability": "Not specified",
        "License": "Community license as Llama 2"
    },
    {
        "model_id": "mistralai/mistral-large",
        "label": "Mistral Large",
        "provider": "Mistral AI",
        "source": "Mistral",
        "functions": [
            {
                "id": "text_generation"
            }
        ],
        "short_description": "Mistral Large, the most advanced Large Language Model (LLM) developed by Mistral Al, is an exceptionally powerful model. Thanks to its state-of-the-art reasoning capabilities it can be applied to any language-based task, including the most sophisticated ones.",
        "long_description": "Mistral Large is ideal for complex tasks that require large reasoning capabilities or are highly specialized. For comprehensive information and examples about this model, please refer to the release blog post.",
        "input_tier": "class_6",
        "output_tier": "class_6",
        "number_params": "",
        "min_shot_size": 1,
        "task_ids": [
            "summarization",
            "retrieval_augmented_generation",
            "classification",
            "generation",
            "code",
            "extraction"
        ],
        "tasks": [
            {
                "id": "summarization"
            },
            {
                "id": "retrieval_augmented_generation"
            },
            {
                "id": "classification"
            },
            {
                "id": "generation"
            },
            {
                "id": "code"
            },
            {
                "id": "extraction"
            }
        ],
        "model_limits": {
            "max_sequence_length": 32768
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 16384
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 16384
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 16384
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2024-07-09"
            }
        ],
        "versions": [
            {
                "version": "2.0.0",
                "available_date": "2024-07-24"
            },
            {
                "version": "1.0.0",
                "available_date": "2024-07-09"
            }
        ],
        "tuned_by": "none",
        "ratings": "none",
        "training_parameters": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "",
        "ModelOverview": "Mistral Large is a cutting-edge text generation model that reaches top-tier reasoning capabilities. It can be used for complex multilingual reasoning tasks, including text understanding, transformation, and code generation. It is natively fluent in English, French, Spanish, German, and Italian, with a nuanced understanding of grammar and cultural context. Its 32K tokens context window allows precise information recall from large documents. Its precise instruction-following enables developers to design their moderation policies. It is natively capable of function calling, which, along with constrained output mode, implemented on la Plateforme, enables application development and tech stack modernisation at scale.",
        "languages": [
            "English",
            "French",
            "Spanish",
            "German",
            "Italian"
        ],
        "TuningInformation": "",
        "TrainingData": "",
        "UsesSupported": [
            "text understanding",
            "transformation",
            "code generation",
            "complex multilingual reasoning tasks"
        ],
        "OptimisedFor": "",
        "PromptingAdvice": "",
        "Output": "",
        "PromptTuningAvailability": "",
        "RegionalAvailability": "",
        "License": ""
    },
    {
        "model_id": "mistralai/mixtral-8x7b-instruct-v01",
        "label": "Mixtral-8x7B-Instruct-v0.1",
        "provider": "Hugging Face",
        "source": "Hugging Face",
        "functions": [
            {
                "id": "text_generation"
            }
        ],
        "short_description": "The Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts.",
        "long_description": "This model is made with AutoGPTQ, which mainly leverages the quantization technique to 'compress' the model weights from FP16 to 4-bit INT and performs 'decompression' on-the-fly before computation (in FP16). As a result, the GPU memory, and the data transferring between GPU memory and GPU compute engine, compared to the original FP16 model, is greatly reduced. The major quantization parameters used in the process are listed below.",
        "input_tier": "class_1",
        "output_tier": "class_1",
        "number_params": "46.7B",
        "min_shot_size": 1,
        "task_ids": [
            "summarization",
            "retrieval_augmented_generation",
            "classification",
            "generation",
            "code",
            "extraction"
        ],
        "tasks": [
            {
                "id": "summarization",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "retrieval_augmented_generation",
                "ratings": {
                    "quality": 3
                }
            },
            {
                "id": "classification",
                "ratings": {
                    "quality": 4
                }
            },
            {
                "id": "generation"
            },
            {
                "id": "code"
            },
            {
                "id": "extraction",
                "ratings": {
                    "quality": 4
                }
            }
        ],
        "model_limits": {
            "max_sequence_length": 32768
        },
        "limits": {
            "lite": {
                "call_time": "5m0s",
                "max_output_tokens": 16384
            },
            "v2-professional": {
                "call_time": "10m0s",
                "max_output_tokens": 16384
            },
            "v2-standard": {
                "call_time": "10m0s",
                "max_output_tokens": 16384
            }
        },
        "lifecycle": [
            {
                "id": "available",
                "start_date": "2024-04-17"
            }
        ],
        "tuned_by": "none",
        "ratings": "none",
        "training_parameters": "none",
        "versions": "none",
        "system": "none",
        "warnings": "none",
        "architecture": "Sparse Mixture of Experts",
        "ModelOverview": "The Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts. The Mixtral-8x7B outperforms Llama 2 70B on most benchmarks we tested. For full details of this model please read our release blog post.",
        "languages": [
            "English"
        ],
        "TuningInformation": "Instruction-tuned",
        "TrainingData": "Code and code-related data",
        "UsesSupported": [
            "Text Generation",
            "Conversational"
        ],
        "OptimisedFor": "Low latency, real-time text generation, long inputs",
        "PromptingAdvice": "Use the instruction format strictly, otherwise the model will generate sub-optimal outputs.",
        "Output": "Text",
        "PromptTuningAvailability": "true",
        "RegionalAvailability": "Not specified",
        "License": "Apache-2.0"
    }
]


ingest.ingest_models(sample_json, wXdProp.MODEL_INDEX_NAME, es_connection)
