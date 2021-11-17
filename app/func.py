import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample
from transformers.data.metrics.squad_metrics import compute_predictions_logits

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def run_batch(contract, questions, prompt_map, details_map, model, tokenizer, device):

    max_seq_length = 512
    doc_stride = 256
    n_best_size = 1
    max_query_length = 64
    max_answer_length = 512
    do_lower_case = False
    null_score_diff_threshold = 0.0

    prompts = [prompt_map[key] for key in prompt_map]
    
    processor = SquadV2Processor()

    examples = []
    for i, question_text in enumerate(prompts):
        example = SquadExample(
            qas_id = str(i),
            question_text = question_text,
            context_text = contract,
            answer_text = None,
            start_position_character = None,
            title = 'Predict',
            answers = None
        )
        examples.append(example)

    features, dataset = squad_convert_examples_to_features(
        examples = examples,
        tokenizer = tokenizer,
        max_seq_length = max_seq_length,
        doc_stride = doc_stride,
        max_query_length = max_query_length,
        is_training = False,
        return_dataset = 'pt',
        threads = 1
    )

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=16)

    all_results = []
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2]
            }
            example_indices = batch[3]
            outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                output = [to_list(output[i]) for output in outputs.to_tuple()]
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)
    
    final_predictions = compute_predictions_logits(
        all_examples = examples,
        all_features = features,
        all_results = all_results,
        n_best_size = n_best_size,
        max_answer_length = max_answer_length,
        do_lower_case = do_lower_case,
        output_prediction_file = None,
        output_nbest_file = None,
        output_null_log_odds_file = None,
        verbose_logging = False,
        version_2_with_negative = True,
        null_score_diff_threshold = null_score_diff_threshold,
        tokenizer = tokenizer
    )

    answers_list = []
    questions_list = []
    details_list = []

    for i, p in enumerate(final_predictions):
        if final_predictions[p] != '':
            questions_list.append(questions[int(p)])
            answers_list.append(final_predictions[p])
            details_list.append(details_map[questions[int(p)]])

    return {
        'answers': answers_list,
        'questions': questions_list,
        'details': details_list,
        'anomalous_topics': ['Not yet implemented.'],
        'anomalous_clauses': ['Not yet implemented.']
    }