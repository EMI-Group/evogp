import torch
from torch import Tensor
from functools import partial


class BaseSelector:
    """
    Base class for selection operators. These selectors are designed to provide receptor
    and donor selection strategies for DiversityCrossover or LeafBiasedCrossover operations.
    """
    def __call__(self, fitness: Tensor, choosed_num: int) -> Tensor:
        raise NotImplementedError


class RankSelector(BaseSelector):
    """
    Rank-based selection operator. It assigns a probability to each individual based on their
    rank in the population, with higher-ranked individuals having a higher chance of being selected.
    Used for selecting receptors and donors in DiversityCrossover or LeafBiasedCrossover.
    """
    def __init__(self, selection_pressure: float = 0.5):
        self.sp = selection_pressure

    def __call__(self, fitness: Tensor, choosed_num: int) -> Tensor:
        sorted_fitness, sorted_indices = torch.sort(fitness, descending=True)
        rank = sorted_indices.cuda()
        n = fitness.size(0)
        random_indices = torch.multinomial(
            (1 / n) * (1 + self.sp * (1 - 2 * rank / (n - 1))),
            choosed_num,
            replacement=True,
        ).to(torch.int32)
        return sorted_indices[random_indices]


class RouletteSelector(BaseSelector):
    """
    Roulette wheel selection operator. This operator selects individuals based on their
    proportional fitness. Individuals with higher fitness values are more likely to be selected.
    Used for selecting receptors and donors in DiversityCrossover or LeafBiasedCrossover.
    """
    def __init__(self):
        pass

    def __call__(self, fitness: Tensor, choosed_num: int) -> Tensor:
        random_indices = torch.multinomial(
            (fitness / torch.sum(fitness)).cuda(),
            choosed_num,
            replacement=True,
        ).to(torch.int32)
        return random_indices


class TournamentSelector(BaseSelector):
    """
    Tournament selection operator. It selects a subset of individuals (contenders) randomly,
    and from each subset, the best individual is selected based on fitness. Can be used with 
    DiversityCrossover or LeafBiasedCrossover for receptor and donor selection.
    """
    def __init__(
        self,
        tournament_size: int,
        best_probability: float = 1,
        replace: bool = True,
    ):
        self.t_size = tournament_size
        self.best_p = best_probability
        self.replace = replace

    def __call__(self, fitness: Tensor, choosed_num: int) -> Tensor:
        def generate_contenders():
            total_size = fitness.size(0)
            n_tournament = int(total_size / self.t_size)
            k_times = int((choosed_num - 1) / n_tournament) + 1

            @partial(torch.vmap, randomness="different")
            def traverse_once(p):
                return torch.multinomial(
                    p, n_tournament * self.t_size, replacement=self.replace
                ).to(torch.int32)

            p = torch.ones((k_times, total_size)).cuda()
            return traverse_once(p).reshape(-1, self.t_size)[:choosed_num]

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
        return choosed_indices.reshape(-1)


class TruncationSelector(BaseSelector):
    """
    Truncation selection operator. This operator selects the top-performing individuals based 
    on their fitness and eliminates a fraction of the population. The remaining individuals 
    are selected for crossover, such as in DiversityCrossover or LeafBiasedCrossover.
    """
    def __init__(self, survivor_rate: float = 0.5):
        self.survivor_rate = survivor_rate

    def __call__(self, fitness: Tensor, choosed_num: int) -> Tensor:
        sorted_fitness, sorted_indices = torch.sort(fitness, descending=True)
        num_selectable = int(fitness.size(0) * self.survivor_rate)
        random_indices = torch.multinomial(
            (sorted_indices < num_selectable).to("cuda", torch.float),
            choosed_num,
            replacement=True,
        ).to(torch.int32)
        return sorted_indices[random_indices]
