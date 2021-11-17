import os
from celery import Celery

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from func import run_batch

celery = Celery(__name__)
celery.conf.broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379')
celery.conf.result_backend = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379')

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'There are {torch.cuda.device_count()} GPUs available')
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
else:
    print(f'Using CPU')
    device = torch.device('cpu')

model_URI = 'models/roberta-large/'

model = AutoModelForQuestionAnswering.from_pretrained(model_URI).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_URI, use_fast=False)

@celery.task(name='create_task')
def create_task(contract, questions, prompt_map, details_map):
    output = run_batch(contract, questions, prompt_map, details_map, model, tokenizer, device)
    return output