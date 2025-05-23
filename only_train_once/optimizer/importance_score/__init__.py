from .magnitude import *
from .cosine_similarity import *
from .taylor import *
import torch

def calculate_importance_score(criteria, param_group):
    param_group['importance_scores'] = dict()
    with torch.no_grad():
        for cri_name in criteria:
            if 'magnitude' == cri_name:
                importance_score_by_magnitude(param_group)
            elif 'avg_magnitude' == cri_name:
                importance_score_by_avg_magnitude(param_group)
            elif 'cosine_similarity' == cri_name:
                importance_score_by_cosine_similarity(param_group)
            elif 'taylor_first_order' == cri_name:
                importance_score_by_first_order_taylor(param_group)
            elif 'taylor_second_order' == cri_name:
                importance_score_by_second_order_taylor(param_group)

def calculate_importance_score_lora(criteria, param_group, global_params):
    param_group['importance_scores'] = dict()
    with torch.no_grad():
        for cri_name in criteria:
            if 'magnitude' in cri_name:
                importance_score_by_magnitude_lora(param_group)
            elif 'avg_magnitude' == cri_name:
                importance_score_by_avg_magnitude_lora(param_group)
            elif 'cosine_similarity' in cri_name:
                importance_score_by_cosine_similarity_lora(param_group, global_params)
            elif 'taylor_first_order' in cri_name:
                importance_score_by_first_order_taylor_lora(param_group, global_params)
            elif 'taylor_second_order' in cri_name:
                importance_score_by_second_order_taylor_lora(param_group, global_params)