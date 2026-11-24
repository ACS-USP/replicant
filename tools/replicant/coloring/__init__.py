"""Potts / graph coloring by BP on random graphs (K3; Krzakala & Zdeborova, lecture notes, Ch. 6).

  potts_bp.py   the paramagnetic fixed point (closed forms), the ferromagnetic ansatz on d-regular graphs (every
                fixed point, its stability and Bethe free entropy), the q >= 3 first-order structure (beta_stab,
                beta_s, beta_c), the annealed threshold c_ann and the Kesten-Stigum threshold c_KS
  first_moment.py  the exact first moment E Z of q-colorings of G(N, M) at finite N (two exact routes)
  planted.py    the planted coloring, its SBM link (Eq. (13.3)), c_l (Eq. (13.9)), quiet planting (K7)
  rsb.py        frozen SP closed form, colorability / clustering / condensation / rigidity thresholds (K7)
  rsb_pop.py    1RSB population dynamics (Parisi m) and the frozen-SP population (K7)
  sample.py     exhaustive counts, Bethe counts and BP from the planted start on sampled graphs (K7)
"""
