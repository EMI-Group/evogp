from typing import Optional
import torch

from ...tree import Forest
from .base import BaseSelection
from functools import partial


class TournamentSelection(BaseSelection):
    """
    TournamentSelection implements a selection strategy where individuals compete in tournaments to be selected.

    A specified number of individuals are randomly selected from the population to form a tournament. 
    The winner of each tournament is selected based on a probability that favors the best-performing individual.

    Two sampling modes are available:
        - With replacement: Individuals can be selected multiple times for tournaments.
        - Without replacement: Individuals with fewer prior selections are favored in tournament formation.

    The selection probability for the best individual in each tournament is controlled by the `best_probability` parameter.
    """

    def __init__(
        self,
        tournament_size: int,
        best_probability: float = 1,
        replace: bool = True,
        survivor_rate: float = 0.5,
        elite_rate: float = 0,
        survivor_cnt: Optional[int] = None,
        elite_cnt: Optional[int] = None,
    ):
        """
        Args:
            tournament_size (int): The number of individuals in each tournament.
            best_probability (float): The probability of selecting the best individual from the tournament. Defaults to 1 (always select the best).
            replace (bool): Whether to allow replacement when selecting individuals for tournaments. Defaults to True (with replacement).
            survivor_rate (float): The proportion of individuals to retain in the next generation, based on their fitness. Should be between 0 and 1. Defaults to 0.5.
            elite_rate (float): The proportion of elite individuals to retain based on fitness. Should be between 0 and 1. Defaults to 0.
            survivor_cnt (Optional[int]): The exact number of individuals to retain as survivors (if provided). Defaults to None.
            elite_cnt (Optional[int]): The exact number of elite individuals to retain (if provided). Defaults to None.
        """
        super().__init__()

        # Ensure survivor_rate and elite_rate are within the valid range [0, 1].
        assert 0 <= survivor_rate <= 1, "survival_rate should be in [0, 1]"
        assert 0 <= elite_rate <= 1, "elite_rate should be in [0, 1]"

        # Initialize the parameters.
        self.t_size = tournament_size
        self.best_p = best_probability
        self.replace = replace
        self.survivor_rate = survivor_rate
        self.survivor_cnt = survivor_cnt
        self.elite_rate = elite_rate
        self.elite_cnt = elite_cnt


    def __call__(self, forest: Forest, fitness: torch.Tensor):
        """
        Perform tournament selection and return the indices of selected elite and survivor individuals.

        Args:
            forest (Forest): The population of individuals represented as a Forest object.
            fitness (torch.Tensor): A tensor containing the fitness values of individuals in the population.

        Returns:
            elite_indices (torch.Tensor): Indices of the individuals selected as elites.
            survivor_indices (torch.Tensor): Indices of the individuals selected as survivors based on tournament selection.
        """
        
        # Function to select individuals for tournament (with different randomness for each call).
        @partial(torch.vmap, randomness="different")
        def traverse_once(p):
            return torch.multinomial(
                p, n_tournament * self.t_size, replacement=self.replace
            ).to(torch.int32)

        # Function to select the winner without probability, always choosing the best individual.
        @torch.vmap
        def t_selection_without_p(contenders):
            contender_fitness = fitness[contenders]
            best_idx = torch.argmax(contender_fitness)[None]
            return contenders[best_idx]

        # Function to select the winner with probability, based on the best individual's fitness.
        @partial(torch.vmap, randomness="different")
        def t_selection_with_p(contenders):
            contender_fitness = fitness[contenders]
            idx_rank = torch.argsort(
                contender_fitness, descending=True
            )  # Sort individuals by fitness in descending order.
            random = torch.rand(1).cuda()
            best_p = torch.tensor(self.best_p).cuda()  # Probability of selecting the best individual.
            nth_choosed = (torch.log(random) / torch.log(1 - best_p)).to(torch.int32)
            nth_choosed = torch.where(
                nth_choosed >= self.t_size, torch.tensor(0), nth_choosed
            )
            return contenders[idx_rank[nth_choosed]]

        # Preprocess survivor and elite counts based on provided counts or rates.
        if self.survivor_cnt is not None:
            survivor_cnt = self.survivor_cnt
        else:
            survivor_cnt = int(forest.pop_size * self.survivor_rate)

        if self.elite_cnt is not None:
            elite_cnt = self.elite_cnt
        else:
            elite_cnt = int(forest.pop_size * self.elite_rate)

        # Survivor selection: run multiple tournaments and select winners.
        total_size = fitness.size(0)
        n_tournament = int(total_size / self.t_size)
        k_times = int((survivor_cnt - 1) / n_tournament) + 1
        p = torch.ones((k_times, total_size)).cuda()
        contenders = traverse_once(p).reshape(-1, self.t_size)[:survivor_cnt]

        # Choose survivor indices based on the tournament selection strategy.
        if self.t_size > 1000:
            survivor_indices = t_selection_without_p(contenders).reshape(-1)
        else:
            survivor_indices = t_selection_with_p(contenders).reshape(-1)

        # Elite selection: select the top individuals based on fitness.
        if elite_cnt == 0:
            elite_indices = torch.tensor([], device="cuda", dtype=torch.int64)
        else:
            # Sort individuals by fitness in descending order to get the top elites.
            sorted_fitness, sorted_indices = torch.sort(fitness, descending=True)
            elite_indices = sorted_indices[:elite_cnt]

        return elite_indices, survivor_indices
