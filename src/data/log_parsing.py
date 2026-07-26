"""
Mining de templates de logs via drain3 — nécessaire pour les connecteurs
dont les logs bruts n'ont PAS de template_id pré-calculé (Jenkins, GitLab
CI), contrairement à LogHub (échantillons déjà templated) ou RCAEval
(cluster_id déjà présent dans logs.csv). drain3 est une dépendance du projet
depuis le début (requirements.txt) mais n'était utilisée nulle part avant
ce module.
"""

from typing import List

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


def mine_templates(lines: List[str]) -> List[int]:
    """
    Retourne un cluster_id (template_id) par ligne, dans le même ordre.

    Une nouvelle instance TemplateMiner par appel: pas d'état persisté ni
    partagé entre deux appels/sources différentes — chaque parse() constitue
    son propre corpus de templates.
    """
    miner = TemplateMiner(config=TemplateMinerConfig())
    return [miner.add_log_message(line.strip())["cluster_id"] for line in lines]
