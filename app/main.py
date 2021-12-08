from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse

from celery.result import AsyncResult

from worker import create_task
import maps

app = FastAPI(title='Query Analysis')


def add_custom_query(questions, details, prompt_map, details_map):
    if len(questions) > 0:
        for i in range(len(details)):
            detail = details[i]
            question = questions[i]
            prompt = f"Highlight the parts (if any) of this contract related to '{question}' that should be reviewed by a lawyer. Details: {detail}"
            prompt_map[question] = prompt
            details_map[question] = detail
    return prompt_map, details_map


@app.get('/')
def home(request: Request):
    return {
        'status': True
    }


@app.post('/tasks', status_code=201)
def run_task(payload=Body(...)):
    contract = payload['contract']
    questions = payload.get('user_questions', [])
    details = payload.get('user_details', [])
    prompt_map, details_map = add_custom_query(
        questions, details, maps.prompt_map, maps.details_map)
    task = create_task.delay(contract, list(
        prompt_map.keys()), prompt_map, details_map)
    return JSONResponse({
        'task_id': task.id
    })


@app.get('/tasks/{task_id}')
def get_status(task_id):
    task_result = AsyncResult(task_id)
    result = {
        'task_id': task_id,
        'task_status': task_result.status,
        'task_result': task_result.result
    }
    return JSONResponse(result)
