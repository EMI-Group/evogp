from typing import Tuple
import jax
import jax.numpy as jnp
from .base import Selection
from jax import lax

class TournamentSelection(Selection):

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

    def __call__(self, key, trees, fitnesses, config) -> Tuple[jax.Array, jax.Array]:
        # returns: (trees, fitness)
        
        def generate_contenders(key):
            pop_size = config["pop_size"]
            t_size = self.t_size
            n_choose_once = int(pop_size / t_size)
            k_times = int(pop_size / n_choose_once) + 1
            keys = jax.random.split(key, k_times)

            @jax.vmap
            def choose_once(key):
                # traverse the entire population for one choice
                return jax.random.choice(key, jnp.arange(pop_size), shape=(n_choose_once, t_size), replace=self.replace)
            
            return choose_once(keys).reshape(-1, t_size)[:pop_size]

        @jax.vmap
        def t_selection_without_p(key, contenders):
            contender_fitness = fitnesses[contenders]
            best_idx = jnp.argmax(contender_fitness)
            return contenders[best_idx]
        
        @jax.vmap
        def t_selection_with_p(key, contenders):
            contender_fitness = fitnesses[contenders]
            idx_rank = jnp.argsort(contender_fitness)[::-1] # the index of individual from high to low
            random = jax.random.uniform(key)
            nth_choosed = (jnp.log(1 - random) / jnp.log(1 - self.best_p)).astype(int)
            nth_choosed = lax.cond(nth_choosed >= self.t_size, lambda: 0, lambda: nth_choosed)
            return contenders[idx_rank[nth_choosed]]

        k1, k2 = jax.random.split(key, 2)
        contenders = generate_contenders(k1)
        keys = jax.random.split(k2, config["pop_size"])
        winner_idx = lax.cond(self.t_size > 1000, t_selection_without_p, t_selection_with_p, keys, contenders)
        return trees[winner_idx], fitnesses[winner_idx]