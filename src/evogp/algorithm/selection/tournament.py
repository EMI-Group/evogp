import torch
from torch import Tensor

from .base import BaseSelection
from functools import partial


class TournamentSelection(BaseSelection):

    def __init__(
        self,
        tournament_size: int,
        best_probability: float,
        replace: bool = True,
    ):
        super().__init__()
        self.t_size = tournament_size
        self.best_p = best_probability
        self.replace = replace

    def __call__(self, fitness: Tensor, choose_num: int) -> Tensor:
        def generate_contenders():
            total_size = fitness.size(0)
            n_tournament = int(total_size / self.t_size)
            k_times = int((choose_num - 1) / n_tournament) + 1

            @partial(torch.vmap, randomness="different")
            def traverse_once(p):
                return torch.multinomial(
                    p, n_tournament * self.t_size, replacement=self.replace
                ).to(torch.int32)

            p = torch.ones((k_times, total_size)).cuda()
            return traverse_once(p).reshape(-1, self.t_size)[:choose_num]

        @torch.vmap
        def t_selection_without_p(contenders):
            contender_fitness = fitness[contenders]
            best_idx = torch.argmax(contender_fitness)[None]
            return contenders[best_idx]

        @partial(torch.vmap, randomness="different")
        def t_selection_with_p(contenders):
            contender_fitness = fitness[contenders]
            idx_rank = torch.argsort(
                contender_fitness, descending=True
            )  # the index of individual from high to low
            random = torch.rand(1).cuda()
            best_p = torch.tensor(self.best_p).cuda()
            nth_choosed = (torch.log(random) / torch.log(1 - best_p)).to(torch.int32)
            nth_choosed = torch.where(
                nth_choosed >= self.t_size, torch.tensor(0), nth_choosed
            )
            return contenders[idx_rank[nth_choosed]]

        contenders = generate_contenders()
        if self.t_size > 1000:
            choosed_indices = t_selection_without_p(contenders)
        else:
            choosed_indices = t_selection_with_p(contenders)
        return choosed_indices.reshape((-1))
