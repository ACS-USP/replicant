"""The catalog registry: every entry, in catalog order (catalog.catalog() keys them by name and caches them).

Moved here from catalog.catalog() by K1 (the order of the 38 earlier entries is pinned by
tests/test_catalog.py test_registry_move_is_identity).  A new entry module registers by appending its
entries at the end of entries(); the earlier entries' order never changes.
"""


def entries() -> list:
    """[CatalogEntry] of every entry, built fresh (catalog() builds them once and caches them)."""
    from .catalog_bayes import bayes_entries
    from .catalog_bounds import bounds_entries
    from .catalog_csp import csp_entries
    from .catalog_dense_memory import dense_memory_entries
    from .catalog_entries import dense_entries
    from .catalog_factor import factor_entries
    from .catalog_inference import inference_entries
    from .catalog_lattice import lattice_entries
    from .catalog_mc import mc_entries
    from .catalog_meanfield import meanfield_entries
    from .catalog_learning import learning_entries
    from .catalog_online import online_entries
    from .catalog_percolation import percolation_entries
    from .catalog_percolation_kcore import kcore_entries
    from .catalog_percolation_lattice import percolation_lattice
    from .catalog_series import series_entries
    from .catalog_sparse import sparse_entries
    from .catalog_wishart import spiked_wishart
    return (dense_entries() + sparse_entries() + csp_entries() + inference_entries()
            + learning_entries() + series_entries() + lattice_entries() + bounds_entries()
            + online_entries() + percolation_entries() + [percolation_lattice()] + kcore_entries()
            + mc_entries() + dense_memory_entries() + meanfield_entries() + bayes_entries() + factor_entries()
            + [spiked_wishart()])


__all__ = ["entries"]
