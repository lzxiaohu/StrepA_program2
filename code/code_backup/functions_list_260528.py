# file name: functions_list_260528.py

import numpy as np
from numba import njit
from typing import Any, Tuple
import matplotlib.pyplot as plt

from numpy.random import default_rng, Generator as _NpGen

from numpy.random import default_rng, SeedSequence

WEEKS_PER_YEAR = 52.14

def _get(p: Any, key_or_idx: Any):
    """
    Helper to read from either:
      - numpy array/list (0-based index), or
      - dict-like (by key), or
      - object with attributes (by name).
    """
    if isinstance(p, (list, tuple, np.ndarray)):
        return p[key_or_idx]
    # dict-like
    if isinstance(p, dict):
        return p[key_or_idx]
    # object with attributes
    return getattr(p, key_or_idx)

def parameters(param: Any) -> Tuple[
    int, int, int, float, int, int,
    float, float, float, float,
    float, float, float, int,
    float, float, float,
    np.ndarray, int, float
]:
    """
    Python port of MATLAB `parameters.m`.

    Input
    -----
    `param` in the same order as MATLAB:
      [ DurationSimulation(years), Nstrains, Dimmunity(weeks),
        sigma, omega, x,
        Cperweek, Nagents, alpha,
        AgeDeath(years), BasicReproductionNumber ]
    """

    # ---- Read raw inputs (supports vector, dict, or object) ----
    DurationSimulation      = float(_get(param, 0) if not isinstance(param, dict) and not hasattr(param, 'DurationSimulation')
                                    else _get(param, 'DurationSimulation'))
    Nstrains_in             = int  (_get(param, 1) if not isinstance(param, dict) and not hasattr(param, 'Nstrains')
                                    else _get(param, 'Nstrains'))
    Dimmunity               = float(_get(param, 2) if not isinstance(param, dict) and not hasattr(param, 'Dimmunity')
                                    else _get(param, 'Dimmunity'))
    StrengthImmunity_raw    = float(_get(param, 3) if not isinstance(param, dict) and not hasattr(param, 'sigma')
                                    else _get(param, 'sigma'))                   # 'sigma' in your top script
    StrengthCross_raw       = float(_get(param, 4) if not isinstance(param, dict) and not hasattr(param, 'omega')
                                    else _get(param, 'omega'))                   # 'omega' in your top script
    x                       = float(_get(param, 5) if not isinstance(param, dict) and not hasattr(param, 'x')
                                    else _get(param, 'x'))
    Cperweek                = float(_get(param, 6) if not isinstance(param, dict) and not hasattr(param, 'Cperweek')
                                    else _get(param, 'Cperweek'))
    Nagents                 = int  (_get(param, 7) if not isinstance(param, dict) and not hasattr(param, 'Nagents')
                                    else _get(param, 'Nagents'))
    MR_per_week             = float(_get(param, 8) if not isinstance(param, dict) and not hasattr(param, 'alpha')
                                    else _get(param, 'alpha'))                   # 'alpha' used as migration rate per week
    AgeDeath                = float(_get(param, 9) if not isinstance(param, dict) and not hasattr(param, 'AgeDeath')
                                    else _get(param, 'AgeDeath'))
    BasicReproductionNumber = float(_get(param,10) if not isinstance(param, dict) and not hasattr(param, 'BasicReproductionNumber')
                                    else _get(param, 'BasicReproductionNumber'))

    # ---- Multiplier (kept as in MATLAB; currently = 1) ----
    multiplier = 1

    # ---- Time grid ----
    Endtime_weeks = DurationSimulation * WEEKS_PER_YEAR
    dt_weeks = 1.0 / 7.0
    dt_years = dt_weeks / WEEKS_PER_YEAR
    time = np.arange(0.0, Endtime_weeks + 1e-12, dt_weeks)  # include endpoint like 0:dt:Endtime
    Ntimesteps = time.size

    # ---- Initial seeding ----
    NI0perstrain = 10
    NR0perstrain = 10

    # ---- Epidemiological params ----
    Nstrains = int(Nstrains_in * multiplier)  # number of initial strains (after multiplier)
    Nst = int(Nstrains_in)                    # number of strains modeled
    # Durations (weeks)
    Dinfection = 2.0
    # Co-infection carrying capacity
    CCC = float(Nstrains_in)
    # Migration controls
    MR = MR_per_week                           # number of migrations per week
    prevalence_in_migrants = 0.03               # original: 0.1

    # ---- Immunity strengths, clamped to [0,1] ----
    StrengthImmunity = float(np.clip(StrengthImmunity_raw, 0.0, 1.0))
    Immunity = int(StrengthImmunity > 0)       # 0 = none, 1 = waning immunity present
    StrengthCrossImmunity = float(np.clip(StrengthCross_raw, 0.0, 1.0))

    # ---- Rates per week ----
    Rrecovery = 1.0 / Dinfection
    Rimmunityloss = 1.0 / Dimmunity
    Rdeath = 1.0 / AgeDeath / WEEKS_PER_YEAR   # per week

    # ---- Probabilities per timestep (dt_weeks) ----
    Precovery = 1.0 - np.exp(-dt_weeks * Rrecovery)
    Pimmunityloss = 1.0 - np.exp(-dt_weeks * Rimmunityloss)

    # ---- Migration per timestep ----
    MRpertimestep = MR * dt_weeks

    # ---- Contacts per timestep ----
    Cpertimestep = Cperweek * dt_weeks

    # ---- Base probability of transmission per contact ----
    # Ptransmission = (Rdeath + Rrecovery + MR / Nagents) * R0 / Cperweek
    Ptransmission = (Rdeath + Rrecovery + MR / Nagents) * BasicReproductionNumber / Cperweek

    # ---- Return tuple (order exactly matches MATLAB) ----
    return (
        Nagents, Nstrains, Nst, AgeDeath, NI0perstrain, NR0perstrain,
        Cpertimestep, MRpertimestep, Precovery, Pimmunityloss,
        Ptransmission, x, StrengthImmunity, Immunity,
        StrengthCrossImmunity, prevalence_in_migrants, CCC,
        time, Ntimesteps, dt_years
    )


def initialise_agents(params):
    """
    Python port of MATLAB:
      [AgentCharacteristics, ImmuneStatus, time] = initialise_agents(params)

    Returns
    -------
    AgentCharacteristics : (Nagents, Nstrains+1) float
        cols 0..Nstrains-1: infection copies for each strain (0/1/2/…)
        last col: age (years)
    ImmuneStatus : (Nagents, Nstrains) int {0,1}
        strain-specific immunity flags
    time : np.ndarray
        time grid (weeks), as returned by parameters(params)
    """
    (Nagents, Nstrains, Nst, AgeDeath, NI0perstrain, NR0perstrain,
     _Cpertimestep, _MRpertimestep, _Precovery, _Pimmunityloss,
     _Ptransmission, _x, _StrengthImmunity, _Immunity,
     _StrengthCrossImmunity, _prevalence_in_migrants, _CCC,
     time, _Ntimesteps, _dt_years) = parameters(params)

    Nagents   = int(Nagents)
    Nstrains  = int(Nstrains)
    Nst       = int(Nst)
    AgeDeath  = float(AgeDeath)
    NI0       = int(NI0perstrain)
    NR0       = int(NR0perstrain)

    # Allocate
    AgentCharacteristics = np.zeros((Nagents, Nstrains + 1), dtype=float)
    ImmuneStatus         = np.zeros((Nagents, Nstrains), dtype=int)

    # Ages ~ Uniform(0, AgeDeath)
    AgentCharacteristics[:, -1] = np.random.rand(Nagents) * AgeDeath

    # Fallback like MATLAB if too few agents to seed NI0 per strain
    if Nagents < NI0 * Nst:
        NI0 = 4
        NR0 = 4

    # Pools for sampling without replacement across strains
    pool_inf = np.arange(Nagents)  # for infections seeding
    pool_imm = np.arange(Nagents)  # for immunity seeding

    for i in range(Nst):  # 0..Nst-1 strains to seed
        if pool_inf.size < NI0 or pool_imm.size < NR0:
            break

        # choose positions within the remaining pools
        pos_inf = np.random.choice(pool_inf.size, size=NI0, replace=False)
        pos_imm = np.random.choice(pool_imm.size, size=NR0, replace=False)

        infected_agents = pool_inf[pos_inf]
        immune_agents   = pool_imm[pos_imm]

        # set one copy of strain i
        AgentCharacteristics[infected_agents, i] = 1.0
        ImmuneStatus[immune_agents, i] = 1

        # remove those agents from further seeding (across strains)
        pool_inf = np.delete(pool_inf, pos_inf)
        pool_imm = np.delete(pool_imm, pos_imm)

    return AgentCharacteristics, ImmuneStatus, time


def initialise_agents_v5(params, rng):
    """
    Reproducible version of initialise_agents.
    Prefer passing an rng (numpy.random.Generator). If not provided, you can pass a seed.
    If neither is provided, a new RNG is created (non-deterministic).
    """
    
    rng = rng

    (Nagents, Nstrains, Nst, AgeDeath, NI0perstrain, NR0perstrain,
     _Cpertimestep, _MRpertimestep, _Precovery, _Pimmunityloss,
     _Ptransmission, _x, _StrengthImmunity, _Immunity,
     _StrengthCrossImmunity, _prevalence_in_migrants, _CCC,
     time, _Ntimesteps, _dt_years) = parameters(params)

    Nagents   = int(Nagents)
    Nstrains  = int(Nstrains)
    Nst       = int(Nst)
    AgeDeath  = float(AgeDeath)
    NI0       = int(NI0perstrain)
    NR0       = int(NR0perstrain)

    # Allocate
    AgentCharacteristics = np.zeros((Nagents, Nstrains + 1), dtype=float)
    ImmuneStatus         = np.zeros((Nagents, Nstrains), dtype=int)

    # Ages ~ Uniform(0, AgeDeath) using rng (not global np.random)
    AgentCharacteristics[:, -1] = rng.random(Nagents) * AgeDeath

    # Fallback if too few agents to seed NI0 per strain
    if Nagents < NI0 * Nst:
        NI0 = 4
        NR0 = 4

    # Pools for sampling without replacement across strains
    pool_inf = np.arange(Nagents)  # for infections seeding
    pool_imm = np.arange(Nagents)  # for immunity seeding

    for i in range(Nst):  # 0..Nst-1 strains to seed
        if pool_inf.size < NI0 or pool_imm.size < NR0:
            break

        # choose positions within the remaining pools (reproducible via rng)
        pos_inf = rng.choice(pool_inf.size, size=NI0, replace=False)
        pos_imm = rng.choice(pool_imm.size, size=NR0, replace=False)

        infected_agents = pool_inf[pos_inf]
        immune_agents   = pool_imm[pos_imm]

        # set one copy of strain i
        AgentCharacteristics[infected_agents, i] = 1.0
        ImmuneStatus[immune_agents, i] = 1

        # remove those agents from further seeding (across strains)
        pool_inf = np.delete(pool_inf, pos_inf)
        pool_imm = np.delete(pool_imm, pos_imm)

    return AgentCharacteristics, ImmuneStatus, time

def simulator(AgentCharacteristics, ImmuneStatus, params,
              specifyPtransmission: int = 0,
              cross_immunity_effect_on_coinfections: int = 1):
    """
    Python port of MATLAB simulator.m

    Inputs
    ------
    AgentCharacteristics : (Nagents, Nstrains+1) float
        cols 0..Nstrains-1: infection copies per strain
        last col: agent age (years)
    ImmuneStatus : (Nagents, Nstrains) int {0,1}
    params : same container you pass to parameters(params)
    specify Ptransmission : 1 to force Ptransmission=0.0301, else 0
    cross_immunity_effect_on_coinfections : 1 on, 0 off

    Returns
    -------
    SSPrev : (Nstrains, Ntimesteps)
    AgentsInfectedByKStrains : (Nstrains, Ntimesteps)
    """

    (Nagents, Nstrains, Nst, AgeDeath, _NI0, _NR0,
     Cpertimestep, MRpertimestep, Precovery, Pimmunityloss,
     Ptransmission, x, StrengthImmunity, Immunity,
     StrengthCrossImmunity, prevalence_in_migrants, CCC,
     time, Ntimesteps, dt_years) = parameters(params)

    Nagents   = int(Nagents)
    Nstrains  = int(Nstrains)
    Nst       = int(Nst)
    AgeDeath  = float(AgeDeath)
    CCC       = float(CCC)

    # Optionally override Ptransmission
    if specifyPtransmission == 1:
        Ptransmission = 0.0301

    # ---- Cross-immunity-accelerated recovery probability per step ----
    dt_weeks = 1.0/7.0  # from parameters.m
    Rrecovery = -np.log(1.0 - Precovery) / dt_weeks
    if StrengthCrossImmunity != 1:
        Rrecovery_cici = 1.0 / ((1.0 / Rrecovery) * (1.0 - StrengthCrossImmunity))
        Precovery_cici = 1.0 - np.exp(-dt_weeks * Rrecovery_cici)
    else:
        Precovery_cici = 1.0

    # ---- Pre-generated random streams (mirrors pregenerate_random_numbers.m) ----
    ContactRand = np.random.poisson(Cpertimestep, size=(1_000_000, 1)).astype(int)
    MRRand      = np.random.poisson(MRpertimestep, size=(1_000_000, 1)).astype(int)
    SamplingU   = np.random.rand(1_000_000, 1)
    countCR = 0  # contacts
    countMR = 0  # migrants
    countU  = 0  # generic uniforms

    def _takeU(n):
        nonlocal SamplingU, countU
        end = countU + n
        if end > len(SamplingU):
            SamplingU = np.random.rand(1_000_000, 1)
            countU = 0
            end = n
        out = SamplingU[countU:end, 0]
        countU = end
        return out

    def _takeCR():
        nonlocal ContactRand, countCR
        x = ContactRand[countCR, 0]
        countCR += 1
        if countCR >= len(ContactRand):
            ContactRand = np.random.poisson(Cpertimestep, size=(1_000_000, 1)).astype(int)
            countCR = 0
        return x

    def _takeMR():
        nonlocal MRRand, countMR
        x = MRRand[countMR, 0]
        countMR += 1
        if countMR >= len(MRRand):
            MRRand = np.random.poisson(MRpertimestep, size=(1_000_000, 1)).astype(int)
            countMR = 0
        return x

    # ---- Outputs ----
    SSPrev = np.zeros((Nstrains, Ntimesteps), dtype=float)
    AgentsInfectedByKStrains = np.zeros((Nstrains, Ntimesteps), dtype=float)

    # t = 0
    BB = AgentCharacteristics[:, :Nstrains]
    SSPrev[:, 0] = BB.sum(axis=0)

    tot0 = BB.sum()
    if tot0 > 1:
        kvec = BB.sum(axis=1).astype(int)
        kvec = kvec[kvec != 0]
        if kvec.size:
            K, counts = np.unique(kvec, return_counts=True)
            AgentsInfectedByKStrains[K - 1, 0] = counts
    elif tot0 == 1:
        AgentsInfectedByKStrains[0, 0] = 1

    # Tracks “fast recovery” flags (CICI) for each (agent, strain)
    CICI = np.zeros_like(BB)

    # ---- Main time loop ----
    for t in range(Ntimesteps - 1):
        CurrentAC  = AgentCharacteristics.copy()
        CurrentImm = ImmuneStatus.copy()
        DD = CurrentAC[:, :Nst]  # infections per strain at start of step

        # ===== RECOVERY =====
        inf_norm = (DD > 0) & (CICI == 0)
        inf_cici = (DD > 0) & (CICI > 0)

        r_n_rows, r_n_cols = np.where(inf_norm)
        if r_n_rows.size:
            rec = (np.random.rand(r_n_rows.size) < Precovery)
            AgentCharacteristics[r_n_rows[rec], r_n_cols[rec]] = 0
            # only normal recoveries gain ss-immunity
            ImmuneStatus[r_n_rows[rec], r_n_cols[rec]] = 1 * Immunity

        r_c_rows, r_c_cols = np.where(inf_cici)
        if r_c_rows.size:
            rec = (np.random.rand(r_c_rows.size) < Precovery_cici)
            AgentCharacteristics[r_c_rows[rec], r_c_cols[rec]] = 0
            CICI[r_c_rows[rec], r_c_cols[rec]] = 0  # no immunity granted here

        # ===== WANING IMMUNITY =====
        w_rows, w_cols = np.where(CurrentImm == 1)
        if w_rows.size:
            lose = (np.random.rand(w_rows.size) < Pimmunityloss)
            ImmuneStatus[w_rows[lose], w_cols[lose]] = 0

        # ===== TRANSMISSION =====
        G = DD.sum(axis=1)
        infected_agents = np.where(G > 0)[0]
        if infected_agents.size:
            # base per-contact susceptibility, with co-infection resistance
            TotalInf = DD.sum(axis=1)
            P1 = Ptransmission * np.power((1.0 - TotalInf / CCC), x)
            P1 = np.clip(P1, 0.0, 1.0)
            P1 = np.repeat(P1[:, None], Nstrains, axis=1)
            InfectionProb = P1.copy()

            # strain-specific immunity
            if StrengthImmunity > 0:
                mask_ss = (CurrentImm == 1)
                InfectionProb[mask_ss] = P1[mask_ss] * (1.0 - StrengthImmunity)

            # cross-strain immunity (any immunity to any strain)
            if StrengthCrossImmunity > 0:
                any_imm = (CurrentImm == 1).any(axis=1)[:, None]
                mask_cs = (CurrentImm == 0) & np.repeat(any_imm, Nstrains, axis=1)
                InfectionProb[mask_cs] = P1[mask_cs] * (1.0 - StrengthCrossImmunity)

            for a in infected_agents:
                infecting_strains = np.where(DD[a, :] > 0)[0]
                X = _takeCR()  # contacts for this source agent
                if X <= 0:
                    continue

                # sample contacts (with replacement), avoid self
                U = _takeU(X)
                contacts = np.ceil(Nagents * U).astype(int) - 1
                selfmask = (contacts == a)
                if np.any(selfmask):
                    others = np.arange(Nagents)
                    others = others[others != a]
                    contacts[selfmask] = np.random.choice(others, size=selfmask.sum(), replace=True)

                # choose one transmitting strain per contact among agent's current strains
                U2 = _takeU(X)
                if infecting_strains.size == 1:
                    chosen = np.full(X, infecting_strains[0], dtype=int)
                else:
                    idx = np.ceil(infecting_strains.size * U2).astype(int) - 1
                    chosen = infecting_strains[idx]

                # success Bernoulli
                susc = InfectionProb[contacts, chosen]
                U3 = _takeU(X)
                success = (U3 < susc)

                if np.any(success):
                    contacts = contacts[success]
                    chosen   = chosen[success]
                    # dedupe same contact (keep first)
                    order = np.argsort(contacts)
                    contacts = contacts[order]
                    chosen   = chosen[order]
                    keep = np.concatenate([[True], np.diff(contacts) > 0])
                    contacts = contacts[keep]
                    chosen   = chosen[keep]

                    # increment copies from the snapshot state
                    AgentCharacteristics[contacts, chosen] = CurrentAC[contacts, chosen] + 1

                    if cross_immunity_effect_on_coinfections == 1:
                        temp = AgentCharacteristics[:, :Nstrains].copy()
                        temp[contacts, chosen] = 0               # remove the newly acquired strains
                        temp = temp[contacts, :]
                        temp[temp > 0] = 1                       # mark other extant strains
                        add = np.zeros_like(CICI)
                        add[contacts, :] = temp
                        CICI = np.clip(CICI + add, 0, 1)

                    # those contacts cannot be infected again in this pass
                    InfectionProb[contacts, :] = 0.0

        # ===== AGE, DEATH, BIRTH =====
        AgentCharacteristics[:, Nstrains] = dt_years + CurrentAC[:, Nstrains]
        dead = np.where(AgentCharacteristics[:, Nstrains] > AgeDeath)[0]
        if dead.size:
            AgentCharacteristics[dead, :Nstrains] = 0
            ImmuneStatus[dead, :] = 0
            AgentCharacteristics[dead, Nstrains] = 0.001
            CICI[dead, :] = 0

        # ===== MIGRATION =====
        NumMig = _takeMR()
        if NumMig > 0:
            if NumMig >= Nagents:
                migrants = np.random.permutation(Nagents)
            else:
                migrants = np.random.choice(Nagents, size=NumMig, replace=False)

            infected_mig = (np.random.rand(NumMig) < prevalence_in_migrants)
            n_im = int(infected_mig.sum())
            if n_im > 0:
                mig_strains = np.random.randint(0, Nst, size=n_im)  # 0..Nst-1

            cm = ci = 0
            for m in range(NumMig):
                idx = migrants[m]
                ImmuneStatus[idx, :] = 0
                CICI[idx, :] = 0
                AgentCharacteristics[idx, Nstrains] = np.random.rand() * AgeDeath
                AgentCharacteristics[idx, :Nstrains] = 0
                if infected_mig[cm]:
                    AgentCharacteristics[idx, mig_strains[ci]] = 1
                    ci += 1
                cm += 1

        # ===== RECORDING =====
        BB = AgentCharacteristics[:, :Nstrains]
        SSPrev[:, t + 1] = BB.sum(axis=0)

        tot = BB.sum()
        if tot > 1:
            kvec = BB.sum(axis=1).astype(int)
            kvec = kvec[kvec != 0]
            if kvec.size:
                K, counts = np.unique(kvec, return_counts=True)
                AgentsInfectedByKStrains[K - 1, t + 1] = counts
        elif tot == 1:
            AgentsInfectedByKStrains[0, t + 1] = 1

    return SSPrev, AgentsInfectedByKStrains


def simulator_v2(AgentCharacteristics, ImmuneStatus, params,
              specifyPtransmission: int = 0,
              cross_immunity_effect_on_coinfections: int = 1):
    """

    Inputs
    ------
    AgentCharacteristics : (Nagents, Nstrains+1) float
        cols 0..Nstrains-1: infection copies per strain
        last col: agent age (years)
    ImmuneStatus : (Nagents, Nstrains) int {0,1}
    params : same container you pass to parameters(params)
    specifyPtransmission : 1 to force Ptransmission=0.0301, else 0
    cross_immunity_effect_on_coinfections : 1 on, 0 off

    Returns
    -------
    SSPrev : (Nstrains, Ntimesteps)
    AgentsInfectedByKStrains : (Nstrains, Ntimesteps)
    """

    (Nagents, Nstrains, Nst, AgeDeath, _NI0, _NR0,
     Cpertimestep, MRpertimestep, Precovery, Pimmunityloss,
     Ptransmission, x, StrengthImmunity, Immunity,
     StrengthCrossImmunity, prevalence_in_migrants, CCC,
     time, Ntimesteps, dt_years) = parameters(params)

    Nagents   = int(Nagents)
    Nstrains  = int(Nstrains)
    Nst       = int(Nst)
    AgeDeath  = float(AgeDeath)
    CCC       = float(CCC)

    # Optionally override Ptransmission
    if specifyPtransmission == 1:
        Ptransmission = 0.0301

    # ---- Cross-immunity-accelerated recovery probability per step ----
    dt_weeks = 1.0/7.0  # from parameters.m
    Rrecovery = -np.log(1.0 - Precovery) / dt_weeks
    if StrengthCrossImmunity != 1:
        Rrecovery_cici = 1.0 / ((1.0 / Rrecovery) * (1.0 - StrengthCrossImmunity))
        Precovery_cici = 1.0 - np.exp(-dt_weeks * Rrecovery_cici)
    else:
        Precovery_cici = 1.0

    # ---- Pre-generated random streams (mirrors pregenerate_random_numbers.m) ----
    ContactRand = np.random.poisson(Cpertimestep, size=(1_000_000, 1)).astype(int)
    MRRand      = np.random.poisson(MRpertimestep, size=(1_000_000, 1)).astype(int)
    SamplingU   = np.random.rand(1_000_000, 1)
    countCR = 0  # contacts
    countMR = 0  # migrants
    countU  = 0  # generic uniforms

    def _takeU(n):
        nonlocal SamplingU, countU
        end = countU + n
        if end > len(SamplingU):
            SamplingU = np.random.rand(1_000_000, 1)
            countU = 0
            end = n
        out = SamplingU[countU:end, 0]
        countU = end
        return out

    def _takeCR():
        nonlocal ContactRand, countCR
        x = ContactRand[countCR, 0]
        countCR += 1
        if countCR >= len(ContactRand):
            ContactRand = np.random.poisson(Cpertimestep, size=(1_000_000, 1)).astype(int)
            countCR = 0
        return x

    def _takeMR():
        nonlocal MRRand, countMR
        x = MRRand[countMR, 0]
        countMR += 1
        if countMR >= len(MRRand):
            MRRand = np.random.poisson(MRpertimestep, size=(1_000_000, 1)).astype(int)
            countMR = 0
        return x

    # ---- Outputs ----
    SSPrev = np.zeros((Nstrains, Ntimesteps), dtype=float)
    AgentsInfectedByKStrains = np.zeros((Nstrains, Ntimesteps), dtype=float)

    # t = 0
    BB = AgentCharacteristics[:, :Nstrains]
    SSPrev[:, 0] = BB.sum(axis=0)

    tot0 = BB.sum()
    if tot0 > 1:
        kvec = BB.sum(axis=1).astype(int)
        kvec = kvec[kvec != 0]
        if kvec.size:
            K, counts = np.unique(kvec, return_counts=True)
            AgentsInfectedByKStrains[K - 1, 0] = counts
    elif tot0 == 1:
        AgentsInfectedByKStrains[0, 0] = 1

    # Tracks “fast recovery” flags (CICI) for each (agent, strain)
    CICI = np.zeros_like(BB)

    # ---- Main time loop ----
    for t in range(Ntimesteps - 1):
        CurrentAC  = AgentCharacteristics.copy()
        CurrentImm = ImmuneStatus.copy()
        DD = CurrentAC[:, :Nst]  # infections per strain at start of step

        # ===== RECOVERY =====
        inf_norm = (DD > 0) & (CICI == 0)
        inf_cici = (DD > 0) & (CICI > 0)

        r_n_rows, r_n_cols = np.where(inf_norm)
        if r_n_rows.size:
            rec = (np.random.rand(r_n_rows.size) < Precovery)
            AgentCharacteristics[r_n_rows[rec], r_n_cols[rec]] = 0
            # only normal recoveries gain ss-immunity
            ImmuneStatus[r_n_rows[rec], r_n_cols[rec]] = 1 * Immunity

        r_c_rows, r_c_cols = np.where(inf_cici)
        if r_c_rows.size:
            rec = (np.random.rand(r_c_rows.size) < Precovery_cici)
            AgentCharacteristics[r_c_rows[rec], r_c_cols[rec]] = 0
            CICI[r_c_rows[rec], r_c_cols[rec]] = 0  # no immunity granted here

        # ===== WANING IMMUNITY =====
        w_rows, w_cols = np.where(CurrentImm == 1)
        if w_rows.size:
            lose = (np.random.rand(w_rows.size) < Pimmunityloss)
            ImmuneStatus[w_rows[lose], w_cols[lose]] = 0

        # ===== TRANSMISSION =====
        # Reuse TotalInf for both G and susceptibility calculation
        TotalInf = DD.sum(axis=1)
        infected_agents = np.where(TotalInf > 0)[0]

        if infected_agents.size:
            # base per-contact susceptibility, with co-infection resistance
            P1 = Ptransmission * np.power((1.0 - TotalInf / CCC), x)
            P1 = np.clip(P1, 0.0, 1.0)

            # expand to (Nagents, Nstrains)
            InfectionProb = np.repeat(P1[:, None], Nstrains, axis=1)

            # strain-specific immunity
            if StrengthImmunity > 0:
                mask_ss = (CurrentImm == 1)
                InfectionProb[mask_ss] *= (1.0 - StrengthImmunity)

            # cross-strain immunity (any immunity to any strain)
            if StrengthCrossImmunity > 0:
                any_imm = (CurrentImm == 1).any(axis=1)[:, None]
                mask_cs = (CurrentImm == 0) & np.repeat(any_imm, Nstrains, axis=1)
                InfectionProb[mask_cs] *= (1.0 - StrengthCrossImmunity)

            for a in infected_agents:
                infecting_strains = np.where(DD[a, :] > 0)[0]
                if infecting_strains.size == 0:
                    continue

                X = _takeCR()  # contacts for this source agent
                if X <= 0:
                    continue

                # sample contacts (with replacement), avoid self efficiently
                U = _takeU(X)
                # map U into 0..Nagents-2 and then "skip" a
                contacts = (U * (Nagents - 1)).astype(int)
                contacts[contacts >= a] += 1  # now in 0..Nagents-1, excluding a

                # choose one transmitting strain per contact among agent's strains
                U2 = _takeU(X)
                if infecting_strains.size == 1:
                    chosen = np.empty(X, dtype=int)
                    chosen.fill(infecting_strains[0])
                else:
                    idx = (U2 * infecting_strains.size).astype(int)
                    idx[idx == infecting_strains.size] = infecting_strains.size - 1
                    chosen = infecting_strains[idx]

                # success Bernoulli
                susc = InfectionProb[contacts, chosen]
                U3 = _takeU(X)
                success = (U3 < susc)

                if np.any(success):
                    contacts = contacts[success]
                    chosen   = chosen[success]
                    # dedupe same contact (keep first)
                    order = np.argsort(contacts)
                    contacts = contacts[order]
                    chosen   = chosen[order]
                    keep = np.concatenate([[True], np.diff(contacts) > 0])
                    contacts = contacts[keep]
                    chosen   = chosen[keep]

                    # increment copies from the snapshot state
                    AgentCharacteristics[contacts, chosen] = CurrentAC[contacts, chosen] + 1

                    if cross_immunity_effect_on_coinfections == 1:
                        # --- Optimised cross-immunity update: only touched rows ---
                        temp = AgentCharacteristics[contacts, :Nstrains].copy()
                        temp[np.arange(contacts.size), chosen] = 0   # remove newly acquired strain
                        temp[temp > 0] = 1                           # mark other extant strains
                        CICI[contacts, :] = np.clip(
                            CICI[contacts, :] + temp,
                            0,
                            1,
                        )

                    # those contacts cannot be infected again in this pass
                    InfectionProb[contacts, :] = 0.0

        # ===== AGE, DEATH, BIRTH =====
        AgentCharacteristics[:, Nstrains] = dt_years + CurrentAC[:, Nstrains]
        dead = np.where(AgentCharacteristics[:, Nstrains] > AgeDeath)[0]
        if dead.size:
            AgentCharacteristics[dead, :Nstrains] = 0
            ImmuneStatus[dead, :] = 0
            AgentCharacteristics[dead, Nstrains] = 0.001
            CICI[dead, :] = 0

        # ===== MIGRATION =====
        NumMig = _takeMR()
        if NumMig > 0:
            if NumMig >= Nagents:
                migrants = np.random.permutation(Nagents)
            else:
                migrants = np.random.choice(Nagents, size=NumMig, replace=False)

            infected_mig = (np.random.rand(NumMig) < prevalence_in_migrants)
            n_im = int(infected_mig.sum())
            if n_im > 0:
                mig_strains = np.random.randint(0, Nst, size=n_im)  # 0..Nst-1

            cm = ci = 0
            for m in range(NumMig):
                idx = migrants[m]
                ImmuneStatus[idx, :] = 0
                CICI[idx, :] = 0
                AgentCharacteristics[idx, Nstrains] = np.random.rand() * AgeDeath
                AgentCharacteristics[idx, :Nstrains] = 0
                if infected_mig[cm]:
                    AgentCharacteristics[idx, mig_strains[ci]] = 1
                    ci += 1
                cm += 1

        # ===== RECORDING =====
        BB = AgentCharacteristics[:, :Nstrains]
        SSPrev[:, t + 1] = BB.sum(axis=0)

        tot = BB.sum()
        if tot > 1:
            kvec = BB.sum(axis=1).astype(int)
            kvec = kvec[kvec != 0]
            if kvec.size:
                K, counts = np.unique(kvec, return_counts=True)
                AgentsInfectedByKStrains[K - 1, t + 1] = counts
        elif tot == 1:
            AgentsInfectedByKStrains[0, t + 1] = 1

    return SSPrev, AgentsInfectedByKStrains


def simulator_v3(AgentCharacteristics, ImmuneStatus, params,
                 specifyPtransmission: int = 0,
                 cross_immunity_effect_on_coinfections: int = 1):
    """
    Optimised version of simulator_v2 based on profiler feedback
    (still NumPy-only, no Numba).

    Inputs
    ------
    AgentCharacteristics : (Nagents, Nstrains+1) float
        cols 0..Nstrains-1: infection copies per strain
        last col: agent age (years)
    ImmuneStatus : (Nagents, Nstrains) int {0,1}
    params : same container you pass to parameters(params)
    specifyPtransmission : 1 to force Ptransmission=0.0301, else 0
    cross_immunity_effect_on_coinfections : 1 on, 0 off

    Returns
    -------
    SSPrev : (Nstrains, Ntimesteps)
    AgentsInfectedByKStrains : (Nstrains, Ntimesteps)
    """

    (Nagents, Nstrains, Nst, AgeDeath, _NI0, _NR0,
     Cpertimestep, MRpertimestep, Precovery, Pimmunityloss,
     Ptransmission, x, StrengthImmunity, Immunity,
     StrengthCrossImmunity, prevalence_in_migrants, CCC,
     time, Ntimesteps, dt_years) = parameters(params)

    Nagents   = int(Nagents)
    Nstrains  = int(Nstrains)
    Nst       = int(Nst)
    AgeDeath  = float(AgeDeath)
    CCC       = float(CCC)

    # Optionally override Ptransmission
    if specifyPtransmission == 1:
        Ptransmission = 0.0301

    # ---- Cross-immunity-accelerated recovery probability per step ----
    dt_weeks = 1.0 / 7.0  # from parameters.m
    Rrecovery = -np.log(1.0 - Precovery) / dt_weeks
    if StrengthCrossImmunity != 1:
        Rrecovery_cici = 1.0 / ((1.0 / Rrecovery) * (1.0 - StrengthCrossImmunity))
        Precovery_cici = 1.0 - np.exp(-dt_weeks * Rrecovery_cici)
    else:
        Precovery_cici = 1.0

    # ---- Pre-generated random streams (now 1D for less overhead) ----
    ContactRand = np.random.poisson(Cpertimestep, size=1_000_000).astype(int)
    MRRand      = np.random.poisson(MRpertimestep, size=1_000_000).astype(int)
    SamplingU   = np.random.rand(1_000_000)
    countCR = 0  # contacts
    countMR = 0  # migrants
    countU  = 0  # generic uniforms

    def _takeU(n: int) -> np.ndarray:
        nonlocal SamplingU, countU
        end = countU + n
        if end > SamplingU.size:
            SamplingU = np.random.rand(1_000_000)
            countU = 0
            end = n
        out = SamplingU[countU:end]
        countU = end
        return out

    def _takeCR() -> int:
        nonlocal ContactRand, countCR
        x = ContactRand[countCR]
        countCR += 1
        if countCR >= ContactRand.size:
            ContactRand = np.random.poisson(Cpertimestep, size=1_000_000).astype(int)
            countCR = 0
        return x

    def _takeMR() -> int:
        nonlocal MRRand, countMR
        x = MRRand[countMR]
        countMR += 1
        if countMR >= MRRand.size:
            MRRand = np.random.poisson(MRpertimestep, size=1_000_000).astype(int)
            countMR = 0
        return x

    # ---- Outputs ----
    SSPrev = np.zeros((Nstrains, Ntimesteps), dtype=float)
    AgentsInfectedByKStrains = np.zeros((Nstrains, Ntimesteps), dtype=float)

    # t = 0
    BB = AgentCharacteristics[:, :Nstrains]
    SSPrev[:, 0] = BB.sum(axis=0)

    tot0 = BB.sum()
    if tot0 > 1:
        kvec = BB.sum(axis=1).astype(int)
        kvec = kvec[kvec != 0]
        if kvec.size:
            K, counts = np.unique(kvec, return_counts=True)
            AgentsInfectedByKStrains[K - 1, 0] = counts
    elif tot0 == 1:
        AgentsInfectedByKStrains[0, 0] = 1

    # Tracks “fast recovery” flags (CICI) for each (agent, strain)
    CICI = np.zeros_like(BB)

    # ---- Main time loop ----
    for t in range(Ntimesteps - 1):
        CurrentAC  = AgentCharacteristics.copy()
        CurrentImm = ImmuneStatus.copy()
        DD = CurrentAC[:, :Nst]  # infections per strain at start of step

        # ===== RECOVERY =====
        inf_norm = (DD > 0) & (CICI == 0)
        inf_cici = (DD > 0) & (CICI > 0)

        r_n_rows, r_n_cols = np.where(inf_norm)
        if r_n_rows.size:
            rec = (np.random.rand(r_n_rows.size) < Precovery)
            AgentCharacteristics[r_n_rows[rec], r_n_cols[rec]] = 0
            # only normal recoveries gain ss-immunity
            ImmuneStatus[r_n_rows[rec], r_n_cols[rec]] = 1 * Immunity

        r_c_rows, r_c_cols = np.where(inf_cici)
        if r_c_rows.size:
            rec = (np.random.rand(r_c_rows.size) < Precovery_cici)
            AgentCharacteristics[r_c_rows[rec], r_c_cols[rec]] = 0
            CICI[r_c_rows[rec], r_c_cols[rec]] = 0  # no immunity granted here

        # ===== WANING IMMUNITY =====
        w_rows, w_cols = np.where(CurrentImm == 1)
        if w_rows.size:
            lose = (np.random.rand(w_rows.size) < Pimmunityloss)
            ImmuneStatus[w_rows[lose], w_cols[lose]] = 0

        # ===== TRANSMISSION =====
        # Reuse TotalInf for both infection presence and susceptibility
        TotalInf = DD.sum(axis=1)
        infected_agents = np.where(TotalInf > 0)[0]

        if infected_agents.size:
            # base per-contact susceptibility, with co-infection resistance
            P1 = Ptransmission * np.power((1.0 - TotalInf / CCC), x)
            P1 = np.clip(P1, 0.0, 1.0)

            # expand to (Nagents, Nstrains)
            InfectionProb = np.repeat(P1[:, None], Nstrains, axis=1)

            # strain-specific immunity
            if StrengthImmunity > 0:
                mask_ss = (CurrentImm == 1)
                InfectionProb[mask_ss] *= (1.0 - StrengthImmunity)

            # cross-strain immunity (any immunity to any strain)
            if StrengthCrossImmunity > 0:
                any_imm = (CurrentImm == 1).any(axis=1)[:, None]
                mask_cs = (CurrentImm == 0) & np.repeat(any_imm, Nstrains, axis=1)
                InfectionProb[mask_cs] *= (1.0 - StrengthCrossImmunity)

            for a in infected_agents:
                # strains infecting agent a
                infecting_strains = np.where(DD[a, :] > 0)[0]
                if infecting_strains.size == 0:
                    continue

                X = _takeCR()  # contacts for this source agent
                if X <= 0:
                    continue

                # sample contacts (with replacement), avoid self efficiently
                U = _takeU(X)
                # map U into 0..Nagents-2 and then "skip" a
                contacts = (U * (Nagents - 1)).astype(int)
                contacts[contacts >= a] += 1  # now in 0..Nagents-1, excluding a

                # choose one transmitting strain per contact among agent's strains
                U2 = _takeU(X)
                if infecting_strains.size == 1:
                    chosen = np.empty(X, dtype=int)
                    chosen.fill(infecting_strains[0])
                else:
                    idx = (U2 * infecting_strains.size).astype(int)
                    idx[idx == infecting_strains.size] = infecting_strains.size - 1
                    chosen = infecting_strains[idx]

                # success Bernoulli
                susc = InfectionProb[contacts, chosen]
                U3 = _takeU(X)

                # Instead of success mask + np.any(success), get indices directly
                success_idx = np.where(U3 < susc)[0]
                if success_idx.size:
                    contacts = contacts[success_idx]
                    chosen   = chosen[success_idx]

                    # dedupe same contact (keep first)
                    order = np.argsort(contacts)
                    contacts = contacts[order]
                    chosen   = chosen[order]
                    keep = np.concatenate([[True], np.diff(contacts) > 0])
                    contacts = contacts[keep]
                    chosen   = chosen[keep]

                    # increment copies from the snapshot state
                    AgentCharacteristics[contacts, chosen] = CurrentAC[contacts, chosen] + 1

                    if cross_immunity_effect_on_coinfections == 1:
                        # --- Cross-immunity update: only touched rows ---
                        temp = AgentCharacteristics[contacts, :Nstrains].copy()
                        temp[np.arange(contacts.size), chosen] = 0   # remove newly acquired strain
                        temp[temp > 0] = 1                           # mark other extant strains
                        CICI[contacts, :] = np.clip(
                            CICI[contacts, :] + temp,
                            0,
                            1,
                        )

                    # those contacts cannot be infected again in this pass
                    InfectionProb[contacts, :] = 0.0

        # ===== AGE, DEATH, BIRTH =====
        AgentCharacteristics[:, Nstrains] = dt_years + CurrentAC[:, Nstrains]
        dead = np.where(AgentCharacteristics[:, Nstrains] > AgeDeath)[0]
        if dead.size:
            AgentCharacteristics[dead, :Nstrains] = 0
            ImmuneStatus[dead, :] = 0
            AgentCharacteristics[dead, Nstrains] = 0.001
            CICI[dead, :] = 0

        # ===== MIGRATION =====
        NumMig = _takeMR()
        if NumMig > 0:
            if NumMig >= Nagents:
                migrants = np.random.permutation(Nagents)
            else:
                migrants = np.random.choice(Nagents, size=NumMig, replace=False)

            infected_mig = (np.random.rand(NumMig) < prevalence_in_migrants)
            n_im = int(infected_mig.sum())
            if n_im > 0:
                mig_strains = np.random.randint(0, Nst, size=n_im)  # 0..Nst-1

            cm = ci = 0
            for m in range(NumMig):
                idx = migrants[m]
                ImmuneStatus[idx, :] = 0
                CICI[idx, :] = 0
                AgentCharacteristics[idx, Nstrains] = np.random.rand() * AgeDeath
                AgentCharacteristics[idx, :Nstrains] = 0
                if infected_mig[cm]:
                    AgentCharacteristics[idx, mig_strains[ci]] = 1
                    ci += 1
                cm += 1

        # ===== RECORDING =====
        BB = AgentCharacteristics[:, :Nstrains]
        SSPrev[:, t + 1] = BB.sum(axis=0)

        tot = BB.sum()
        if tot > 1:
            kvec = BB.sum(axis=1).astype(int)
            kvec = kvec[kvec != 0]
            if kvec.size:
                K, counts = np.unique(kvec, return_counts=True)
                AgentsInfectedByKStrains[K - 1, t + 1] = counts
        elif tot == 1:
            AgentsInfectedByKStrains[0, t + 1] = 1

    return SSPrev, AgentsInfectedByKStrains

@njit
def _simulator_v3_core(
    AgentCharacteristics,
    ImmuneStatus,
    Nagents,
    Nstrains,
    Nst,
    AgeDeath,
    Cpertimestep,
    MRpertimestep,
    Precovery,
    Pimmunityloss,
    Ptransmission,
    x,
    StrengthImmunity,
    Immunity,
    StrengthCrossImmunity,
    prevalence_in_migrants,
    CCC,
    Ntimesteps,
    dt_years,
    cross_immunity_effect_on_coinfections,
):
    """
    Numba-compiled core version of simulator_v3.

    AgentCharacteristics : float64 (Nagents, Nstrains+1)
    ImmuneStatus         : int64   (Nagents, Nstrains)
    """

    # Outputs
    SSPrev = np.zeros((Nstrains, Ntimesteps), dtype=np.float64)
    AgentsInfectedByKStrains = np.zeros((Nstrains, Ntimesteps), dtype=np.float64)

    # Views for convenience
    infections = AgentCharacteristics[:, :Nstrains]   # (Nagents, Nstrains)
    ages       = AgentCharacteristics[:, Nstrains]    # (Nagents,)

    # t = 0: compute prevalence and K-strain counts
    # prevalence
    for s in range(Nstrains):
        total = 0.0
        for i in range(Nagents):
            total += infections[i, s]
        SSPrev[s, 0] = total

    # K-strain counts
    k_counts = np.zeros(Nstrains, dtype=np.float64)
    for i in range(Nagents):
        k = 0
        for s in range(Nstrains):
            if infections[i, s] > 0.0:
                k += 1
        if 1 <= k <= Nstrains:
            k_counts[k - 1] += 1.0
    for s in range(Nstrains):
        AgentsInfectedByKStrains[s, 0] = k_counts[s]

    # Fast recovery flags
    CICI = np.zeros((Nagents, Nstrains), dtype=np.float64)

    # ---- main time loop ----
    for t in range(Ntimesteps - 1):
        # Snapshot at start of step
        CurrentAC = AgentCharacteristics.copy()
        CurrentImm = ImmuneStatus.copy()
        DD = CurrentAC[:, :Nst]

        # ===== RECOVERY =====
        for i in range(Nagents):
            for s in range(Nst):
                if DD[i, s] > 0.0:
                    if CICI[i, s] == 0.0:
                        # normal recovery
                        if np.random.rand() < Precovery:
                            infections[i, s] = 0.0
                            if Immunity != 0:
                                ImmuneStatus[i, s] = 1
                    else:
                        # CICI recovery
                        if np.random.rand() < Precovery * (1.0 / (1.0 - StrengthCrossImmunity)):
                            infections[i, s] = 0.0
                            CICI[i, s] = 0.0

        # ===== WANING IMMUNITY =====
        for i in range(Nagents):
            for s in range(Nstrains):
                if CurrentImm[i, s] == 1:
                    if np.random.rand() < Pimmunityloss:
                        ImmuneStatus[i, s] = 0

        # ===== TRANSMISSION =====
        # 1) total infection per agent + list of infected agents
        TotalInf = np.zeros(Nagents, dtype=np.float64)
        infected_agents = np.empty(Nagents, dtype=np.int64)
        n_infected = 0
        for i in range(Nagents):
            ti = 0.0
            for s in range(Nst):
                ti += DD[i, s]
            TotalInf[i] = ti
            if ti > 0.0:
                infected_agents[n_infected] = i
                n_infected += 1

        if n_infected > 0:
            # 2) base susceptibility P1[i]
            P1 = np.empty(Nagents, dtype=np.float64)
            for i in range(Nagents):
                val = 1.0 - TotalInf[i] / CCC
                if val < 0.0:
                    val = 0.0
                p = Ptransmission * (val ** x)
                if p < 0.0:
                    p = 0.0
                if p > 1.0:
                    p = 1.0
                P1[i] = p

            # 3) InfectionProb[i, s]
            InfectionProb = np.empty((Nagents, Nstrains), dtype=np.float64)
            for i in range(Nagents):
                base = P1[i]
                for s in range(Nstrains):
                    InfectionProb[i, s] = base

            # strain-specific immunity
            if StrengthImmunity > 0.0:
                for i in range(Nagents):
                    for s in range(Nstrains):
                        if CurrentImm[i, s] == 1:
                            InfectionProb[i, s] = P1[i] * (1.0 - StrengthImmunity)

            # cross-strain immunity
            if StrengthCrossImmunity > 0.0:
                for i in range(Nagents):
                    any_imm = False
                    for s in range(Nstrains):
                        if CurrentImm[i, s] == 1:
                            any_imm = True
                            break
                    if any_imm:
                        for s in range(Nstrains):
                            if CurrentImm[i, s] == 0:
                                InfectionProb[i, s] = P1[i] * (1.0 - StrengthCrossImmunity)

            # 4) per infected agent
            contacts_buf = np.empty(Nagents * 10, dtype=np.int64)  # upper bound placeholder
            chosen_buf   = np.empty(Nagents * 10, dtype=np.int64)

            for idx_inf in range(n_infected):
                a = infected_agents[idx_inf]

                # strains infecting agent a
                # (equivalent to infecting_strains = np.where(DD[a, :] > 0)[0])
                infecting_strains = np.empty(Nst, dtype=np.int64)
                m = 0
                for s in range(Nst):
                    if DD[a, s] > 0.0:
                        infecting_strains[m] = s
                        m += 1
                if m == 0:
                    continue

                # contacts count
                X = np.random.poisson(Cpertimestep)
                if X <= 0:
                    continue

                if X > contacts_buf.shape[0]:
                    # grow buffers if needed
                    contacts_buf = np.empty(X, dtype=np.int64)
                    chosen_buf   = np.empty(X, dtype=np.int64)

                # sample contacts 0..Nagents-1, skipping a
                for j in range(X):
                    r = np.random.rand()
                    idx_c = int(r * (Nagents - 1))
                    if idx_c >= a:
                        idx_c += 1
                    contacts_buf[j] = idx_c

                # sample transmitting strain
                if m == 1:
                    s0 = infecting_strains[0]
                    for j in range(X):
                        chosen_buf[j] = s0
                else:
                    for j in range(X):
                        r2 = np.random.rand()
                        k = int(r2 * m)
                        if k == m:
                            k = m - 1
                        chosen_buf[j] = infecting_strains[k]

                # success trials, store successful indices into another list
                success_idx = np.empty(X, dtype=np.int64)
                n_success = 0
                for j in range(X):
                    c = contacts_buf[j]
                    s_inf = chosen_buf[j]
                    susc = InfectionProb[c, s_inf]
                    if np.random.rand() < susc:
                        success_idx[n_success] = j
                        n_success += 1

                if n_success > 0:
                    # dedupe by contact (keep first) – manual unique after sorting
                    # We’ll sort successes by contact index
                    # simple insertion sort (X per agent usually small)
                    for i_sort in range(1, n_success):
                        key_j = success_idx[i_sort]
                        key_c = contacts_buf[key_j]
                        k = i_sort - 1
                        while k >= 0 and contacts_buf[success_idx[k]] > key_c:
                            success_idx[k + 1] = success_idx[k]
                            k -= 1
                        success_idx[k + 1] = key_j

                    # compact unique contacts
                    unique_n = 1
                    last_j = success_idx[0]
                    last_c = contacts_buf[last_j]
                    unique_idx = np.empty(n_success, dtype=np.int64)
                    unique_idx[0] = last_j

                    for k in range(1, n_success):
                        j = success_idx[k]
                        c = contacts_buf[j]
                        if c != last_c:
                            unique_idx[unique_n] = j
                            unique_n += 1
                            last_c = c
                            last_j = j

                    # apply infections and cross-immunity
                    for k in range(unique_n):
                        j = unique_idx[k]
                        c = contacts_buf[j]
                        s_inf = chosen_buf[j]
                        infections[c, s_inf] = CurrentAC[c, s_inf] + 1.0

                    if cross_immunity_effect_on_coinfections == 1:
                        for k in range(unique_n):
                            j = unique_idx[k]
                            c = contacts_buf[j]
                            s_new = chosen_buf[j]
                            # mark other extant strains
                            for s in range(Nstrains):
                                if s != s_new and infections[c, s] > 0.0:
                                    CICI[c, s] = 1.0

                    # those contacts cannot be infected again: set InfectionProb[c,:]=0
                    for k in range(unique_n):
                        j = unique_idx[k]
                        c = contacts_buf[j]
                        for s in range(Nstrains):
                            InfectionProb[c, s] = 0.0

        # ===== AGE, DEATH, BIRTH =====
        for i in range(Nagents):
            ages[i] += dt_years

        for i in range(Nagents):
            if ages[i] > AgeDeath:
                for s in range(Nstrains):
                    infections[i, s] = 0.0
                    ImmuneStatus[i, s] = 0
                    CICI[i, s] = 0.0
                ages[i] = np.random.rand() * AgeDeath

        # ===== MIGRATION =====
        NumMig = np.random.poisson(MRpertimestep)
        if NumMig > 0:
            # sample migrants (without replacement – simple, but OK)
            if NumMig >= Nagents:
                migrants = np.arange(Nagents)
            else:
                migrants = np.random.choice(Nagents, size=NumMig, replace=False)

            infected_flags = np.zeros(NumMig, dtype=np.int64)
            n_im = 0
            for m in range(NumMig):
                if np.random.rand() < prevalence_in_migrants:
                    infected_flags[m] = 1
                    n_im += 1

            mig_strains = np.empty(n_im, dtype=np.int64)
            for k in range(n_im):
                mig_strains[k] = np.random.randint(0, Nst)

            ci = 0
            for m in range(NumMig):
                idx = migrants[m]
                # reset agent
                for s in range(Nstrains):
                    infections[idx, s] = 0.0
                    ImmuneStatus[idx, s] = 0
                    CICI[idx, s] = 0.0
                ages[idx] = np.random.rand() * AgeDeath

                if infected_flags[m] == 1 and ci < n_im:
                    s_new = mig_strains[ci]
                    infections[idx, s_new] = 1.0
                    ci += 1

        # ===== RECORDING =====
        # prevalence
        for s in range(Nstrains):
            total = 0.0
            for i in range(Nagents):
                total += infections[i, s]
            SSPrev[s, t + 1] = total

        # K-strain counts
        for s in range(Nstrains):
            k_counts[s] = 0.0
        for i in range(Nagents):
            k = 0
            for s in range(Nstrains):
                if infections[i, s] > 0.0:
                    k += 1
            if 1 <= k <= Nstrains:
                k_counts[k - 1] += 1.0
        for s in range(Nstrains):
            AgentsInfectedByKStrains[s, t + 1] = k_counts[s]

    return SSPrev, AgentsInfectedByKStrains


def simulator_v3_numba(AgentCharacteristics, ImmuneStatus, params,
                       specifyPtransmission: int = 0,
                       cross_immunity_effect_on_coinfections: int = 1):
    """
    Numba-accelerated version of simulator_v3.

    Same signature & outputs:
      SSPrev, AgentsInfectedByKStrains
    """

    (Nagents, Nstrains, Nst, AgeDeath, _NI0, _NR0,
     Cpertimestep, MRpertimestep, Precovery, Pimmunityloss,
     Ptransmission, x, StrengthImmunity, Immunity,
     StrengthCrossImmunity, prevalence_in_migrants, CCC,
     time, Ntimesteps, dt_years) = parameters(params)

    Nagents   = int(Nagents)
    Nstrains  = int(Nstrains)
    Nst       = int(Nst)
    AgeDeath  = float(AgeDeath)
    CCC       = float(CCC)

    if specifyPtransmission == 1:
        Ptransmission = 0.0301

    # Ensure dtypes are stable for Numba
    AgentCharacteristics = np.ascontiguousarray(AgentCharacteristics, dtype=np.float64)
    ImmuneStatus         = np.ascontiguousarray(ImmuneStatus, dtype=np.int64)

    SSPrev, AgentsInfectedByKStrains = _simulator_v3_core(
        AgentCharacteristics,
        ImmuneStatus,
        Nagents,
        Nstrains,
        Nst,
        AgeDeath,
        Cpertimestep,
        MRpertimestep,
        Precovery,
        Pimmunityloss,
        Ptransmission,
        x,
        StrengthImmunity,
        Immunity,
        StrengthCrossImmunity,
        prevalence_in_migrants,
        CCC,
        Ntimesteps,
        dt_years,
        cross_immunity_effect_on_coinfections,
    )

    return SSPrev, AgentsInfectedByKStrains

def simulator_v4(AgentCharacteristics, ImmuneStatus, params,
                 specifyPtransmission: int = 0,
                 cross_immunity_effect_on_coinfections: int = 1):
    """
    Inputs
    ------
    AgentCharacteristics : (Nagents, Nstrains+1) float
        cols 0..Nstrains-1: infection copies per strain
        last col: agent age (years)
    ImmuneStatus : (Nagents, Nstrains) int {0,1}
    params : same container you pass to parameters(params)
    specifyPtransmission : 1 to force Ptransmission=0.0301, else 0
    cross_immunity_effect_on_coinfections : 1 on, 0 off

    Returns
    -------
    SSPrev : (Nstrains, Ntimesteps)
    AgentsInfectedByKStrains : (Nstrains, Ntimesteps)
    SSPrev_selected: (Nstrains, Ntimesteps_selected) 
    """

    (Nagents, Nstrains, Nst, AgeDeath, _NI0, _NR0,
     Cpertimestep, MRpertimestep, Precovery, Pimmunityloss,
     Ptransmission, x, StrengthImmunity, Immunity,
     StrengthCrossImmunity, prevalence_in_migrants, CCC,
     time, Ntimesteps, dt_years) = parameters(params)

    Nagents   = int(Nagents)
    Nstrains  = int(Nstrains)
    Nst       = int(Nst)
    AgeDeath  = float(AgeDeath)
    CCC       = float(CCC)

    
    Ntimesteps_selected = int(23)
    enrolled = int(548)
    consultations = np.array([27,21,42,51,36,69, 
                              122,149,172,170,142,147, 
                              40,193,183,211,190,182, 
                              199,130,191,188,161], dtype=int)   # number of consultations at each time point
    time_obs = np.array([1, 32, 62, 93, 123, 154, 
                          185, 214, 245, 275, 306, 336, 
                          367, 398, 428, 459, 489, 520, 
                          551, 579, 610, 640, 671], dtype=int)   # days of consultations
    time_obs_idx = time_obs[time_obs > 0] + 365*18 - 1  # skip the former 18 years

    rng = np.random.default_rng(123)
    pool = rng.choice(Nagents, size=enrolled, replace=False)

    # Optionally override Ptransmission
    if specifyPtransmission == 1:
        Ptransmission = 0.0301

    # ---- Cross-immunity-accelerated recovery probability per step ----
    dt_weeks = 1.0 / 7.0  # from parameters.m
    Rrecovery = -np.log(1.0 - Precovery) / dt_weeks
    if StrengthCrossImmunity != 1:
        Rrecovery_cici = 1.0 / ((1.0 / Rrecovery) * (1.0 - StrengthCrossImmunity))
        Precovery_cici = 1.0 - np.exp(-dt_weeks * Rrecovery_cici)
    else:
        Precovery_cici = 1.0

    # ---- Pre-generated random streams (now 1D for less overhead) ----
    ContactRand = np.random.poisson(Cpertimestep, size=1_000_000).astype(int)
    MRRand      = np.random.poisson(MRpertimestep, size=1_000_000).astype(int)
    SamplingU   = np.random.rand(1_000_000)
    countCR = 0  # contacts
    countMR = 0  # migrants
    countU  = 0  # generic uniforms

    def _takeU(n: int) -> np.ndarray:
        nonlocal SamplingU, countU
        end = countU + n
        if end > SamplingU.size:
            SamplingU = np.random.rand(1_000_000)
            countU = 0
            end = n
        out = SamplingU[countU:end]
        countU = end
        return out

    def _takeCR() -> int:
        nonlocal ContactRand, countCR
        x = ContactRand[countCR]
        countCR += 1
        if countCR >= ContactRand.size:
            ContactRand = np.random.poisson(Cpertimestep, size=1_000_000).astype(int)
            countCR = 0
        return x

    def _takeMR() -> int:
        nonlocal MRRand, countMR
        x = MRRand[countMR]
        countMR += 1
        if countMR >= MRRand.size:
            MRRand = np.random.poisson(MRpertimestep, size=1_000_000).astype(int)
            countMR = 0
        return x

    # ---- Outputs ----
    SSPrev = np.zeros((Nstrains, Ntimesteps), dtype=float)
    SSPrev_selected = np.zeros((Nstrains, Ntimesteps_selected), dtype=float)
    AgentsInfectedByKStrains = np.zeros((Nstrains, Ntimesteps), dtype=float)

    # t = 0
    BB = AgentCharacteristics[:, :Nstrains]
    SSPrev[:, 0] = BB.sum(axis=0)

    tot0 = BB.sum()
    if tot0 > 1:
        kvec = BB.sum(axis=1).astype(int)
        kvec = kvec[kvec != 0]
        if kvec.size:
            K, counts = np.unique(kvec, return_counts=True)
            AgentsInfectedByKStrains[K - 1, 0] = counts
    elif tot0 == 1:
        AgentsInfectedByKStrains[0, 0] = 1

    # Tracks “fast recovery” flags (CICI) for each (agent, strain)
    CICI = np.zeros_like(BB)

    # ---- Main time loop ----
    ii = 0
    for t in range(Ntimesteps - 1):
        CurrentAC  = AgentCharacteristics.copy()
        CurrentImm = ImmuneStatus.copy()
        DD = CurrentAC[:, :Nst]  # infections per strain at start of step

        # ===== RECOVERY =====
        inf_norm = (DD > 0) & (CICI == 0)
        inf_cici = (DD > 0) & (CICI > 0)

        r_n_rows, r_n_cols = np.where(inf_norm)
        if r_n_rows.size:
            rec = (np.random.rand(r_n_rows.size) < Precovery)
            AgentCharacteristics[r_n_rows[rec], r_n_cols[rec]] = 0
            # only normal recoveries gain ss-immunity
            ImmuneStatus[r_n_rows[rec], r_n_cols[rec]] = 1 * Immunity

        r_c_rows, r_c_cols = np.where(inf_cici)
        if r_c_rows.size:
            rec = (np.random.rand(r_c_rows.size) < Precovery_cici)
            AgentCharacteristics[r_c_rows[rec], r_c_cols[rec]] = 0
            CICI[r_c_rows[rec], r_c_cols[rec]] = 0  # no immunity granted here

        # ===== WANING IMMUNITY =====
        w_rows, w_cols = np.where(CurrentImm == 1)
        if w_rows.size:
            lose = (np.random.rand(w_rows.size) < Pimmunityloss)
            ImmuneStatus[w_rows[lose], w_cols[lose]] = 0

        # ===== TRANSMISSION =====
        # Reuse TotalInf for both infection presence and susceptibility
        TotalInf = DD.sum(axis=1)
        infected_agents = np.where(TotalInf > 0)[0]

        if infected_agents.size:
            # base per-contact susceptibility, with co-infection resistance
            P1 = Ptransmission * np.power((1.0 - TotalInf / CCC), x)
            P1 = np.clip(P1, 0.0, 1.0)

            # expand to (Nagents, Nstrains)
            InfectionProb = np.repeat(P1[:, None], Nstrains, axis=1)

            # strain-specific immunity
            if StrengthImmunity > 0:
                mask_ss = (CurrentImm == 1)
                InfectionProb[mask_ss] *= (1.0 - StrengthImmunity)

            # cross-strain immunity (any immunity to any strain)
            if StrengthCrossImmunity > 0:
                any_imm = (CurrentImm == 1).any(axis=1)[:, None]
                mask_cs = (CurrentImm == 0) & np.repeat(any_imm, Nstrains, axis=1)
                InfectionProb[mask_cs] *= (1.0 - StrengthCrossImmunity)

            for a in infected_agents:
                # strains infecting agent a
                infecting_strains = np.where(DD[a, :] > 0)[0]
                if infecting_strains.size == 0:
                    continue

                X = _takeCR()  # contacts for this source agent
                if X <= 0:
                    continue

                # sample contacts (with replacement), avoid self efficiently
                U = _takeU(X)
                # map U into 0..Nagents-2 and then "skip" a
                contacts = (U * (Nagents - 1)).astype(int)
                contacts[contacts >= a] += 1  # now in 0..Nagents-1, excluding a

                # choose one transmitting strain per contact among agent's strains
                U2 = _takeU(X)
                if infecting_strains.size == 1:
                    chosen = np.empty(X, dtype=int)
                    chosen.fill(infecting_strains[0])
                else:
                    idx = (U2 * infecting_strains.size).astype(int)
                    idx[idx == infecting_strains.size] = infecting_strains.size - 1
                    chosen = infecting_strains[idx]

                # success Bernoulli
                susc = InfectionProb[contacts, chosen]
                U3 = _takeU(X)

                # Instead of success mask + np.any(success), get indices directly
                success_idx = np.where(U3 < susc)[0]
                if success_idx.size:
                    contacts = contacts[success_idx]
                    chosen   = chosen[success_idx]

                    # dedupe same contact (keep first)
                    order = np.argsort(contacts)
                    contacts = contacts[order]
                    chosen   = chosen[order]
                    keep = np.concatenate([[True], np.diff(contacts) > 0])
                    contacts = contacts[keep]
                    chosen   = chosen[keep]

                    # increment copies from the snapshot state
                    AgentCharacteristics[contacts, chosen] = CurrentAC[contacts, chosen] + 1

                    if cross_immunity_effect_on_coinfections == 1:
                        # --- Cross-immunity update: only touched rows ---
                        temp = AgentCharacteristics[contacts, :Nstrains].copy()
                        temp[np.arange(contacts.size), chosen] = 0   # remove newly acquired strain
                        temp[temp > 0] = 1                           # mark other extant strains
                        CICI[contacts, :] = np.clip(
                            CICI[contacts, :] + temp,
                            0,
                            1,
                        )

                    # those contacts cannot be infected again in this pass
                    InfectionProb[contacts, :] = 0.0

        # ===== AGE, DEATH, BIRTH =====
        AgentCharacteristics[:, Nstrains] = dt_years + CurrentAC[:, Nstrains]
        dead = np.where(AgentCharacteristics[:, Nstrains] > AgeDeath)[0]
        if dead.size:
            AgentCharacteristics[dead, :Nstrains] = 0
            ImmuneStatus[dead, :] = 0
            AgentCharacteristics[dead, Nstrains] = 0.001
            CICI[dead, :] = 0

        # ===== MIGRATION =====
        NumMig = _takeMR()
        if NumMig > 0:
            if NumMig >= Nagents:
                migrants = np.random.permutation(Nagents)
            else:
                migrants = np.random.choice(Nagents, size=NumMig, replace=False)

            infected_mig = (np.random.rand(NumMig) < prevalence_in_migrants)
            n_im = int(infected_mig.sum())
            if n_im > 0:
                mig_strains = np.random.randint(0, Nst, size=n_im)  # 0..Nst-1

            cm = ci = 0
            for m in range(NumMig):
                idx = migrants[m]
                ImmuneStatus[idx, :] = 0
                CICI[idx, :] = 0
                AgentCharacteristics[idx, Nstrains] = np.random.rand() * AgeDeath
                AgentCharacteristics[idx, :Nstrains] = 0
                if infected_mig[cm]:
                    AgentCharacteristics[idx, mig_strains[ci]] = 1
                    ci += 1
                cm += 1

        # ===== RECORDING =====
        
        BB = AgentCharacteristics[:, :Nstrains]
        SSPrev[:, t + 1] = BB.sum(axis=0)

        if ii < Ntimesteps_selected and t+1 == time_obs_idx[ii]:
            pool_selected = rng.choice(pool, size=consultations[ii], replace=False)
            AC_selected = BB[pool_selected]
            SSPrev_selected[:, ii] = AC_selected.sum(axis=0)
            ii = ii + 1
            # print(ii)

        tot = BB.sum()
        if tot > 1:
            kvec = BB.sum(axis=1).astype(int)
            kvec = kvec[kvec != 0]
            if kvec.size:
                K, counts = np.unique(kvec, return_counts=True)
                AgentsInfectedByKStrains[K - 1, t + 1] = counts
        elif tot == 1:
            AgentsInfectedByKStrains[0, t + 1] = 1

    return SSPrev_selected, SSPrev, AgentsInfectedByKStrains


@njit(cache=True)
def _take1d(arr, idx_ptr):
    """Return arr[idx_ptr[0]] and advance idx_ptr circularly."""
    x = arr[idx_ptr[0]]
    idx_ptr[0] += 1
    if idx_ptr[0] >= arr.size:
        idx_ptr[0] = 0
    return x

@njit(cache=True)
def _takeU_many(U, u_ptr, n, out):
    """Fill 'out' with next n uniforms from U, cycling if needed."""
    m = U.size
    for i in range(n):
        out[i] = U[u_ptr[0]]
        u_ptr[0] += 1
        if u_ptr[0] >= m:
            u_ptr[0] = 0

@njit(cache=True)
def _fisher_yates_first_k(pool, k, U, u_ptr):
    """
    Sample k items without replacement from 'pool' using Fisher–Yates.
    Uses U-stream to generate indices.
    Returns an array of indices into pool (values are pool entries).
    """
    n = pool.size
    tmp = pool.copy()
    for i in range(k):
        # draw j uniformly from [i, n-1]
        u = _take1d(U, u_ptr)
        j = i + int(u * (n - i))
        if j >= n:
            j = n - 1
        # swap
        ti = tmp[i]
        tmp[i] = tmp[j]
        tmp[j] = ti
    return tmp[:k]


@njit(cache=True)
def _simulator_v4_core(
    AgentCharacteristics,
    ImmuneStatus,
    # parameters() unpacked (all numeric, arrays are 1D/2D)
    Nagents, Nstrains, Nst, AgeDeath,
    Cpertimestep, MRpertimestep, Precovery, Pimmunityloss,
    Ptransmission, x, StrengthImmunity, Immunity,
    StrengthCrossImmunity, prevalence_in_migrants, CCC,
    time, Ntimesteps, dt_years,
    # study design arrays
    Ntimesteps_selected,
    consultations,          # (23,)
    time_obs_idx,           # (23,)
    pool,                   # (enrolled,)
    # random streams
    ContactRand, MRRand, U  # 1D pre-generated streams
):
    # pointer "indices" into random streams (mutable int arrays)
    cr_ptr = np.zeros(1, dtype=np.int64)
    mr_ptr = np.zeros(1, dtype=np.int64)
    u_ptr  = np.zeros(1, dtype=np.int64)

    # ---- Cross-immunity-accelerated recovery probability per step ----
    dt_weeks = 1.0 / 7.0
    # invert Precovery back to rate
    Rrecovery = -np.log(1.0 - Precovery) / dt_weeks
    if StrengthCrossImmunity != 1.0:
        Rrecovery_cici = 1.0 / ((1.0 / Rrecovery) * (1.0 - StrengthCrossImmunity))
        Precovery_cici = 1.0 - np.exp(-dt_weeks * Rrecovery_cici)
    else:
        Precovery_cici = 1.0

    # ---- Outputs ----
    SSPrev = np.zeros((Nstrains, Ntimesteps), dtype=np.float64)
    SSPrev_selected = np.zeros((Nstrains, Ntimesteps_selected), dtype=np.float64)
    AgentsInfectedByKStrains = np.zeros((Nstrains, Ntimesteps), dtype=np.float64)

    # t = 0
    BB = AgentCharacteristics[:, :Nstrains]
    SSPrev[:, 0] = BB.sum(axis=0)

    tot0 = BB.sum()
    if tot0 > 1:
        kvec = BB.sum(axis=1).astype(np.int64)
        # count histogram of k>0
        # (Numba doesn't like boolean indexing on new arrays; use loop)
        # build unique K and counts
        # K is 1..max(kvec)
        kmax = 0
        for i in range(kvec.size):
            if kvec[i] > kmax:
                kmax = kvec[i]
        if kmax > 0:
            counts = np.zeros(kmax, dtype=np.int64)
            for i in range(kvec.size):
                kv = kvec[i]
                if kv > 0:
                    counts[kv - 1] += 1
            # place counts back (truncate if > Nstrains)
            Klen = counts.size if counts.size < Nstrains else Nstrains
            for i in range(Klen):
                AgentsInfectedByKStrains[i, 0] = counts[i]
    elif tot0 == 1:
        AgentsInfectedByKStrains[0, 0] = 1

    # Tracks “fast recovery” flags (CICI) for each (agent, strain)
    CICI = np.zeros_like(BB)

    # prealloc tmp buffers
    # We'll reuse these to avoid reallocations in the loop
    tmpU = np.empty(4096, dtype=np.float64)  # will be resized by slicing
    tmpU2 = np.empty(4096, dtype=np.float64)
    tmpU3 = np.empty(4096, dtype=np.float64)

    ii = 0  # index into selected time points
    for t in range(Ntimesteps - 1):
        CurrentAC  = AgentCharacteristics.copy()
        CurrentImm = ImmuneStatus.copy()
        DD = CurrentAC[:, :Nst]  # infections per strain at start of step

        # ===== RECOVERY =====
        # iterate all cells; vectorized bool masks are not JIT-friendly with many ops
        for i in range(DD.shape[0]):
            for j in range(DD.shape[1]):
                if DD[i, j] > 0:
                    if CICI[i, j] == 0:
                        # normal recovery
                        if _take1d(U, u_ptr) < Precovery:
                            AgentCharacteristics[i, j] = 0.0
                            # only normal recoveries gain ss-immunity
                            if Immunity == 1:
                                ImmuneStatus[i, j] = 1
                    else:
                        # cici recovery
                        if _take1d(U, u_ptr) < Precovery_cici:
                            AgentCharacteristics[i, j] = 0.0
                            CICI[i, j] = 0

        # ===== WANING IMMUNITY =====
        for i in range(CurrentImm.shape[0]):
            for j in range(CurrentImm.shape[1]):
                if CurrentImm[i, j] == 1:
                    if _take1d(U, u_ptr) < Pimmunityloss:
                        ImmuneStatus[i, j] = 0

        # ===== TRANSMISSION =====
        # total infections per agent
        TotalInf = DD.sum(axis=1)
        # collect agents who are infectious
        # first pass to count
        na = 0
        for i in range(TotalInf.size):
            if TotalInf[i] > 0:
                na += 1
        if na > 0:
            infected_agents = np.empty(na, dtype=np.int64)
            k = 0
            for i in range(TotalInf.size):
                if TotalInf[i] > 0:
                    infected_agents[k] = i
                    k += 1

            # base per-contact susceptibility P1 (Nagents,)
            P1 = np.empty(Nagents, dtype=np.float64)
            for i in range(Nagents):
                s = 1.0 - TotalInf[i] / CCC
                if s < 0.0:
                    s = 0.0
                P1[i] = Ptransmission * (s ** x)
                if P1[i] < 0.0:
                    P1[i] = 0.0
                if P1[i] > 1.0:
                    P1[i] = 1.0

            # InfectionProb shape (Nagents, Nstrains)
            InfectionProb = np.empty((Nagents, Nstrains), dtype=np.float64)
            for i in range(Nagents):
                for j in range(Nstrains):
                    InfectionProb[i, j] = P1[i]

            # strain-specific immunity
            if StrengthImmunity > 0.0:
                for i in range(Nagents):
                    for j in range(Nstrains):
                        if CurrentImm[i, j] == 1:
                            InfectionProb[i, j] *= (1.0 - StrengthImmunity)

            # cross-strain immunity (any immunity to any strain)
            if StrengthCrossImmunity > 0.0:
                any_imm = np.zeros(Nagents, dtype=np.uint8)
                for i in range(Nagents):
                    ai = 0
                    for j in range(Nstrains):
                        if CurrentImm[i, j] == 1:
                            ai = 1
                            break
                    any_imm[i] = ai
                for i in range(Nagents):
                    if any_imm[i] == 1:
                        for j in range(Nstrains):
                            if CurrentImm[i, j] == 0:
                                InfectionProb[i, j] *= (1.0 - StrengthCrossImmunity)

            # iterate infectious sources
            for a in infected_agents:
                # which strains infecting agent a?
                # first count
                ns = 0
                for j in range(DD.shape[1]):
                    if DD[a, j] > 0:
                        ns += 1
                if ns == 0:
                    continue
                infecting_strains = np.empty(ns, dtype=np.int64)
                kk = 0
                for j in range(DD.shape[1]):
                    if DD[a, j] > 0:
                        infecting_strains[kk] = j
                        kk += 1

                X = _take1d(ContactRand, cr_ptr)
                if X <= 0:
                    continue

                # ensure tmp buffers large enough
                if X > tmpU.size:
                    tmpU = np.empty(X, dtype=np.float64)
                    tmpU2 = np.empty(X, dtype=np.float64)
                    tmpU3 = np.empty(X, dtype=np.float64)

                # contacts: sample with replacement from all agents except self
                _takeU_many(U, u_ptr, X, tmpU)
                contacts = (tmpU[:X] * (Nagents - 1)).astype(np.int64)
                for i in range(X):
                    if contacts[i] >= a:
                        contacts[i] += 1  # now in 0..Nagents-1 excluding a

                # choose transmitting strain per contact among infecting_strains
                _takeU_many(U, u_ptr, X, tmpU2)
                chosen = np.empty(X, dtype=np.int64)
                if infecting_strains.size == 1:
                    s0 = infecting_strains[0]
                    for i in range(X):
                        chosen[i] = s0
                else:
                    m = infecting_strains.size
                    for i in range(X):
                        idx = int(tmpU2[i] * m)
                        if idx >= m:
                            idx = m - 1
                        chosen[i] = infecting_strains[idx]

                # success Bernoulli
                _takeU_many(U, u_ptr, X, tmpU3)
                # collect successful indices
                succ_n = 0
                for i in range(X):
                    if tmpU3[i] < InfectionProb[contacts[i], chosen[i]]:
                        succ_n += 1
                if succ_n == 0:
                    continue

                succ_contacts = np.empty(succ_n, dtype=np.int64)
                succ_chosen   = np.empty(succ_n, dtype=np.int64)
                kk = 0
                for i in range(X):
                    if tmpU3[i] < InfectionProb[contacts[i], chosen[i]]:
                        succ_contacts[kk] = contacts[i]
                        succ_chosen[kk] = chosen[i]
                        kk += 1

                # dedupe: keep first by stable pass (O(n^2) is OK for small X)
                keep = np.ones(succ_n, dtype=np.uint8)
                for i in range(1, succ_n):
                    ci = succ_contacts[i]
                    for j in range(i):
                        if succ_contacts[j] == ci:
                            keep[i] = 0
                            break

                # apply infections
                for i in range(succ_n):
                    if keep[i] == 1:
                        c = succ_contacts[i]
                        s = succ_chosen[i]
                        AgentCharacteristics[c, s] = CurrentAC[c, s] + 1.0

                # update CICI for touched rows only
                # (cross-immunity effect on coinfections)
                for i in range(succ_n):
                    if keep[i] == 1:
                        c = succ_contacts[i]
                        s = succ_chosen[i]
                        # temp row: 1 where other strains > 0
                        for j in range(Nstrains):
                            if j == s:
                                continue
                            if AgentCharacteristics[c, j] > 0.0:
                                CICI[c, j] = 1

                # those contacts cannot be infected again in this pass
                for i in range(succ_n):
                    if keep[i] == 1:
                        c = succ_contacts[i]
                        for j in range(Nstrains):
                            InfectionProb[c, j] = 0.0

        # ===== AGE, DEATH, BIRTH =====
        for i in range(Nagents):
            AgentCharacteristics[i, Nstrains] = dt_years + CurrentAC[i, Nstrains]
            if AgentCharacteristics[i, Nstrains] > AgeDeath:
                for j in range(Nstrains):
                    AgentCharacteristics[i, j] = 0.0
                    ImmuneStatus[i, j] = 0
                    CICI[i, j] = 0
                AgentCharacteristics[i, Nstrains] = 0.001

        # ===== MIGRATION =====
        NumMig = _take1d(MRRand, mr_ptr)
        if NumMig > 0:
            # pick migrants without replacement by shuffling first NumMig positions
            # build a simple permutation of 0..Nagents-1 for first NumMig
            # (reuse fisher-yates trick)
            # Here we just sample indices (less strict than fully shuffling):
            mig_idx = _fisher_yates_first_k(np.arange(Nagents, dtype=np.int64), NumMig, U, u_ptr)

            # infected status for migrants
            infected_mig = np.zeros(NumMig, dtype=np.uint8)
            for i in range(NumMig):
                infected_mig[i] = 1 if _take1d(U, u_ptr) < prevalence_in_migrants else 0

            # random strains for infected migrants
            # choose in 0..Nst-1
            # we draw via U
            n_im = 0
            for i in range(NumMig):
                if infected_mig[i] == 1:
                    n_im += 1
            mig_strains = np.empty(n_im, dtype=np.int64)
            kk = 0
            for i in range(NumMig):
                if infected_mig[i] == 1:
                    u = _take1d(U, u_ptr)
                    s = int(u * Nst)
                    if s >= Nst:
                        s = Nst - 1
                    mig_strains[kk] = s
                    kk += 1

            ci = 0
            for m in range(NumMig):
                idxm = mig_idx[m]
                for j in range(Nstrains):
                    ImmuneStatus[idxm, j] = 0
                    CICI[idxm, j] = 0
                    AgentCharacteristics[idxm, j] = 0.0
                AgentCharacteristics[idxm, Nstrains] = _take1d(U, u_ptr) * AgeDeath
                if infected_mig[m] == 1:
                    AgentCharacteristics[idxm, mig_strains[ci]] = 1.0
                    ci += 1

        # ===== RECORDING =====
        BB = AgentCharacteristics[:, :Nstrains]
        for j in range(Nstrains):
            SSPrev[j, t + 1] = 0.0
        for j in range(Nstrains):
            SSPrev[j, t + 1] = SSPrev[j, t + 1] + BB[:, j].sum()

        # study sampling at specific times
        if ii < Ntimesteps_selected and (t + 1) == time_obs_idx[ii]:
            k = consultations[ii]
            sel = _fisher_yates_first_k(pool, k, U, u_ptr)
            # sum selected agents across strains
            for s in range(Nstrains):
                # sum over selected rows
                total = 0.0
                for r in range(sel.size):
                    total += BB[sel[r], s]
                SSPrev_selected[s, ii] = total
            ii += 1

        # AgentsInfectedByKStrains
        tot = 0.0
        for i in range(Nagents):
            for j in range(Nstrains):
                tot += BB[i, j]
        if tot > 1:
            # k per agent
            # get max k
            kmax = 0
            for i in range(Nagents):
                kv = 0
                for j in range(Nstrains):
                    if BB[i, j] > 0.0:
                        kv += 1
                if kv > kmax:
                    kmax = kv
            if kmax > 0:
                counts = np.zeros(kmax, dtype=np.int64)
                for i in range(Nagents):
                    kv = 0
                    for j in range(Nstrains):
                        if BB[i, j] > 0.0:
                            kv += 1
                    if kv > 0:
                        counts[kv - 1] += 1
                Klen = counts.size if counts.size < Nstrains else Nstrains
                for i in range(Klen):
                    AgentsInfectedByKStrains[i, t + 1] = counts[i]
        elif tot == 1:
            AgentsInfectedByKStrains[0, t + 1] = 1

    return SSPrev_selected, SSPrev, AgentsInfectedByKStrains


def simulator_v4_numba(AgentCharacteristics, ImmuneStatus, params,
                       specifyPtransmission: int = 0,
                       cross_immunity_effect_on_coinfections: int = 1,
                       seed: int = 123,
                       # random stream lengths (tune if needed)
                       stream_len: int = 2_000_000):
    """
    JIT-accelerated simulator.
    Returns:
      SSPrev_selected, SSPrev, AgentsInfectedByKStrains
    """
    # 1) unpack parameters once (Python)
    (Nagents, Nstrains, Nst, AgeDeath, _NI0, _NR0,
     Cpertimestep, MRpertimestep, Precovery, Pimmunityloss,
     Ptransmission, x, StrengthImmunity, Immunity,
     StrengthCrossImmunity, prevalence_in_migrants, CCC,
     time, Ntimesteps, dt_years) = parameters(params)

    # optional override
    if specifyPtransmission == 1:
        Ptransmission = 0.0301

    # 2) sample design
    Ntimesteps_selected = 23
    consultations = np.array([27, 21, 42, 51, 36, 69,
                              122,149,172,170,142,147,
                              40,193,183,211,190,182,
                              199,130,191,188,161], dtype=np.int64)
    time_obs = np.array([1, 32, 62, 93, 123, 154,
                         185, 214, 245, 275, 306, 336,
                         367, 398, 428, 459, 489, 520,
                         551, 579, 610, 640, 671], dtype=np.int64)
    time_obs_idx = time_obs[time_obs > 0] + 365*18 - 1
    time_obs_idx = time_obs_idx.astype(np.int64)

    # 3) cohort pool (chosen in Python)
    rng = np.random.default_rng(seed)
    enrolled = 548
    pool = rng.choice(int(Nagents), size=enrolled, replace=False).astype(np.int64)

    # 4) random streams (Python) → pass to numba
    ContactRand = rng.poisson(Cpertimestep, size=stream_len).astype(np.int64)
    MRRand      = rng.poisson(MRpertimestep, size=stream_len).astype(np.int64)
    U           = rng.random(stream_len).astype(np.float64)

    # 5) call JIT core
    return _simulator_v4_core(
        np.ascontiguousarray(AgentCharacteristics, dtype=np.float64),
        np.ascontiguousarray(ImmuneStatus, dtype=np.int64),
        int(Nagents), int(Nstrains), int(Nst), float(AgeDeath),
        float(Cpertimestep), float(MRpertimestep), float(Precovery), float(Pimmunityloss),
        float(Ptransmission), float(x), float(StrengthImmunity), int(Immunity),
        float(StrengthCrossImmunity), float(prevalence_in_migrants), float(CCC),
        np.ascontiguousarray(time, dtype=np.float64), int(Ntimesteps), float(dt_years),
        int(Ntimesteps_selected),
        np.ascontiguousarray(consultations, dtype=np.int64),
        np.ascontiguousarray(time_obs_idx, dtype=np.int64),
        np.ascontiguousarray(pool, dtype=np.int64),
        np.ascontiguousarray(ContactRand, dtype=np.int64),
        np.ascontiguousarray(MRRand, dtype=np.int64),
        np.ascontiguousarray(U, dtype=np.float64),
    )



def simulator_v5(AgentCharacteristics, ImmuneStatus, params, rng, 
                 specifyPtransmission: int = 0,
                 cross_immunity_effect_on_coinfections: int = 1, 
                 ):
    """
    Reproducible version: all randomness comes from a single numpy.random.Generator.

    Returns
    -------
    SSPrev : (Nstrains, Ntimesteps)
    AgentsInfectedByKStrains : (Nstrains, Ntimesteps)
    SSPrev_selected: (Nstrains, Ntimesteps_selected)
    """
    rng = rng  # <-- single RNG for the whole simulation

    (Nagents, Nstrains, Nst, AgeDeath, _NI0, _NR0,
     Cpertimestep, MRpertimestep, Precovery, Pimmunityloss,
     Ptransmission, x, StrengthImmunity, Immunity,
     StrengthCrossImmunity, prevalence_in_migrants, CCC,
     time, Ntimesteps, dt_years) = parameters(params)

    Nagents   = int(Nagents)
    Nstrains  = int(Nstrains)
    Nst       = int(Nst)
    AgeDeath  = float(AgeDeath)
    CCC       = float(CCC)

    # ---------- observation schedule ----------
    Ntimesteps_selected = int(23)
    enrolled = int(548)
    consultations = np.array([27,21,42,51,36,69,
                              122,149,172,170,142,147,
                              40,193,183,211,190,182,
                              199,130,191,188,161], dtype=int)
    time_obs = np.array([1, 32, 62, 93, 123, 154,
                         185, 214, 245, 275, 306, 336,
                         367, 398, 428, 459, 489, 520,
                         551, 579, 610, 640, 671], dtype=int)
    time_obs_idx = time_obs[time_obs > 0] + 365*18 - 1  # 0-based steps

    # enrolled pool is now reproducible
    pool = rng.choice(Nagents, size=enrolled, replace=False)

    # Optionally override Ptransmission
    if specifyPtransmission == 1:
        Ptransmission = 0.0301

    # ---- Cross-immunity-accelerated recovery probability per step ----
    dt_weeks = 1.0 / 7.0
    Rrecovery = -np.log(1.0 - Precovery) / dt_weeks
    if StrengthCrossImmunity != 1:
        Rrecovery_cici = 1.0 / ((1.0 / Rrecovery) * (1.0 - StrengthCrossImmunity))
        Precovery_cici = 1.0 - np.exp(-dt_weeks * Rrecovery_cici)
    else:
        Precovery_cici = 1.0

    # ---- Pre-generated random streams (refilled by *rng*, not global) ----
    ContactRand = rng.poisson(Cpertimestep, size=1_000_000).astype(np.int64)
    MRRand      = rng.poisson(MRpertimestep, size=1_000_000).astype(np.int64)
    SamplingU   = rng.random(1_000_000)
    countCR = 0
    countMR = 0
    countU  = 0

    def _takeU(n: int) -> np.ndarray:
        nonlocal SamplingU, countU
        end = countU + n
        if end > SamplingU.size:
            SamplingU = rng.random(1_000_000)      # refill with rng
            countU = 0
            end = n
        out = SamplingU[countU:end]
        countU = end
        return out

    def _takeCR() -> int:
        nonlocal ContactRand, countCR
        x = ContactRand[countCR]
        countCR += 1
        if countCR >= ContactRand.size:
            ContactRand = rng.poisson(Cpertimestep, size=1_000_000).astype(np.int64)
            countCR = 0
        return int(x)

    def _takeMR() -> int:
        nonlocal MRRand, countMR
        x = MRRand[countMR]
        countMR += 1
        if countMR >= MRRand.size:
            MRRand = rng.poisson(MRpertimestep, size=1_000_000).astype(np.int64)
            countMR = 0
        return int(x)

    # ---- Outputs ----
    SSPrev = np.zeros((Nstrains, Ntimesteps), dtype=float)
    SSPrev_selected = np.zeros((Nstrains, Ntimesteps_selected), dtype=float)
    AgentsInfectedByKStrains = np.zeros((Nstrains, Ntimesteps), dtype=float)

    # t = 0
    BB = AgentCharacteristics[:, :Nstrains]
    SSPrev[:, 0] = BB.sum(axis=0)

    tot0 = BB.sum()
    if tot0 > 1:
        kvec = BB.sum(axis=1).astype(np.int64)
        kvec = kvec[kvec != 0]
        if kvec.size:
            K, counts = np.unique(kvec, return_counts=True)
            AgentsInfectedByKStrains[K - 1, 0] = counts
    elif tot0 == 1:
        AgentsInfectedByKStrains[0, 0] = 1

    # Tracks “fast recovery” flags (CICI)
    CICI = np.zeros_like(BB)

    # ---- Main time loop ----
    ii = 0
    for t in range(Ntimesteps - 1):
        CurrentAC  = AgentCharacteristics.copy()
        CurrentImm = ImmuneStatus.copy()
        DD = CurrentAC[:, :Nst]

        # ===== RECOVERY =====
        inf_norm = (DD > 0) & (CICI == 0)
        inf_cici = (DD > 0) & (CICI > 0)

        r_n_rows, r_n_cols = np.where(inf_norm)
        if r_n_rows.size:
            rec = (rng.random(r_n_rows.size) < Precovery)  # rng, not np.random
            AgentCharacteristics[r_n_rows[rec], r_n_cols[rec]] = 0
            ImmuneStatus[r_n_rows[rec], r_n_cols[rec]] = 1 * Immunity

        r_c_rows, r_c_cols = np.where(inf_cici)
        if r_c_rows.size:
            rec = (rng.random(r_c_rows.size) < Precovery_cici)
            AgentCharacteristics[r_c_rows[rec], r_c_cols[rec]] = 0
            CICI[r_c_rows[rec], r_c_cols[rec]] = 0

        # ===== WANING IMMUNITY =====
        w_rows, w_cols = np.where(CurrentImm == 1)
        if w_rows.size:
            lose = (rng.random(w_rows.size) < Pimmunityloss)
            ImmuneStatus[w_rows[lose], w_cols[lose]] = 0

        # ===== TRANSMISSION =====
        TotalInf = DD.sum(axis=1)
        infected_agents = np.where(TotalInf > 0)[0]

        if infected_agents.size:
            P1 = Ptransmission * np.power((1.0 - TotalInf / CCC), x)
            P1 = np.clip(P1, 0.0, 1.0)

            InfectionProb = np.repeat(P1[:, None], Nstrains, axis=1)

            if StrengthImmunity > 0:
                mask_ss = (CurrentImm == 1)
                InfectionProb[mask_ss] *= (1.0 - StrengthImmunity)

            if StrengthCrossImmunity > 0:
                any_imm = (CurrentImm == 1).any(axis=1)[:, None]
                mask_cs = (CurrentImm == 0) & np.repeat(any_imm, Nstrains, axis=1)
                InfectionProb[mask_cs] *= (1.0 - StrengthCrossImmunity)

            for a in infected_agents:
                infecting_strains = np.where(DD[a, :] > 0)[0]
                if infecting_strains.size == 0:
                    continue

                X = _takeCR()
                if X <= 0:
                    continue

                # contacts (avoid self)
                U1 = _takeU(X)
                contacts = (U1 * (Nagents - 1)).astype(np.int64)
                contacts[contacts >= a] += 1

                # pick a strain among agent's strains
                U2 = _takeU(X)
                if infecting_strains.size == 1:
                    chosen = np.empty(X, dtype=np.int64)
                    chosen.fill(infecting_strains[0])
                else:
                    idx = (U2 * infecting_strains.size).astype(np.int64)
                    idx[idx == infecting_strains.size] = infecting_strains.size - 1
                    chosen = infecting_strains[idx]

                # Bernoulli success
                susc = InfectionProb[contacts, chosen]
                U3 = _takeU(X)

                success_idx = np.where(U3 < susc)[0]
                if success_idx.size:
                    contacts = contacts[success_idx]
                    chosen   = chosen[success_idx]

                    order = np.argsort(contacts)         # stable dedupe
                    contacts = contacts[order]
                    chosen   = chosen[order]
                    keep = np.concatenate(([True], np.diff(contacts) > 0))
                    contacts = contacts[keep]
                    chosen   = chosen[keep]

                    AgentCharacteristics[contacts, chosen] = CurrentAC[contacts, chosen] + 1

                    if cross_immunity_effect_on_coinfections == 1:
                        temp = AgentCharacteristics[contacts, :Nstrains].copy()
                        temp[np.arange(contacts.size), chosen] = 0
                        temp[temp > 0] = 1
                        CICI[contacts, :] = np.clip(CICI[contacts, :] + temp, 0, 1)

                    InfectionProb[contacts, :] = 0.0

        # ===== AGE, DEATH, BIRTH =====
        AgentCharacteristics[:, Nstrains] = dt_years + CurrentAC[:, Nstrains]
        dead = np.where(AgentCharacteristics[:, Nstrains] > AgeDeath)[0]
        if dead.size:
            AgentCharacteristics[dead, :Nstrains] = 0
            ImmuneStatus[dead, :] = 0
            AgentCharacteristics[dead, Nstrains] = 0.001
            CICI[dead, :] = 0

        # ===== MIGRATION =====
        NumMig = _takeMR()
        if NumMig > 0:
            if NumMig >= Nagents:
                migrants = rng.permutation(Nagents)      # rng, not np.random
            else:
                migrants = rng.choice(Nagents, size=NumMig, replace=False)

            infected_mig = (rng.random(NumMig) < prevalence_in_migrants)
            n_im = int(infected_mig.sum())
            if n_im > 0:
                mig_strains = rng.integers(0, Nst, size=n_im)  # 0..Nst-1

            cm = ci = 0
            for m in range(NumMig):
                idx = migrants[m]
                ImmuneStatus[idx, :] = 0
                CICI[idx, :] = 0
                AgentCharacteristics[idx, Nstrains] = rng.random() * AgeDeath
                AgentCharacteristics[idx, :Nstrains] = 0
                if infected_mig[cm]:
                    AgentCharacteristics[idx, mig_strains[ci]] = 1
                    ci += 1
                cm += 1

        # ===== RECORDING =====
        BB = AgentCharacteristics[:, :Nstrains]
        SSPrev[:, t + 1] = BB.sum(axis=0)

        if ii < Ntimesteps_selected and t + 1 == time_obs_idx[ii]:
            pool_selected = rng.choice(pool, size=consultations[ii], replace=False)
            AC_selected = BB[pool_selected]
            SSPrev_selected[:, ii] = AC_selected.sum(axis=0)
            ii += 1

        tot = BB.sum()
        if tot > 1:
            kvec = BB.sum(axis=1).astype(np.int64)
            kvec = kvec[kvec != 0]
            if kvec.size:
                K, counts = np.unique(kvec, return_counts=True)
                AgentsInfectedByKStrains[K - 1, t + 1] = counts
        elif tot == 1:
            AgentsInfectedByKStrains[0, t + 1] = 1

    return SSPrev_selected, SSPrev, AgentsInfectedByKStrains

def simulator_v5_numba(AgentCharacteristics, ImmuneStatus, params,
                       specifyPtransmission: int = 0,
                       cross_immunity_effect_on_coinfections: int = 1,
                       seed: int = 123,
                       stream_len: int = 2_000_000):
    """
    Reproducible wrapper:
      - All random draws (cohort pool, contact counts, migrant counts, uniforms)
        come from independent child RNGs derived from one master SeedSequence.
      - Core _simulator_v4_core is purely deterministic given these arrays.
    """
    # ----------------------------
    # 1) Unpack parameters (pure)
    # ----------------------------
    (Nagents, Nstrains, Nst, AgeDeath, _NI0, _NR0,
     Cpertimestep, MRpertimestep, Precovery, Pimmunityloss,
     Ptransmission, x, StrengthImmunity, Immunity,
     StrengthCrossImmunity, prevalence_in_migrants, CCC,
     time, Ntimesteps, dt_years) = parameters(params)

    if specifyPtransmission == 1:
        Ptransmission = 0.0301

    # ----------------------------
    # 2) Study schedule (pure)
    # ----------------------------
    Ntimesteps_selected = 23
    consultations = np.array([27, 21, 42, 51, 36, 69,
                              122,149,172,170,142,147,
                              40,193,183,211,190,182,
                              199,130,191,188,161], dtype=np.int64)
    time_obs = np.array([1, 32, 62, 93, 123, 154,
                         185, 214, 245, 275, 306, 336,
                         367, 398, 428, 459, 489, 520,
                         551, 579, 610, 640, 671], dtype=np.int64)
    time_obs_idx = (time_obs[time_obs > 0] + 365*18 - 1).astype(np.int64)

    # ----------------------------
    # 3) Master seed → independent child RNGs
    #    (stable even if you add/remove one stream later)
    # ----------------------------
    ss = SeedSequence(int(seed))
    ss_pool, ss_contact, ss_migrant, ss_uniform = ss.spawn(4)

    rng_pool    = default_rng(ss_pool)
    rng_contact = default_rng(ss_contact)
    rng_migrant = default_rng(ss_migrant)
    rng_uniform = default_rng(ss_uniform)

    # ----------------------------
    # 4) Random draws (reproducible)
    # ----------------------------
    enrolled = 548
    pool = rng_pool.choice(int(Nagents), size=enrolled, replace=False).astype(np.int64)

    ContactRand = rng_contact.poisson(float(Cpertimestep), size=stream_len).astype(np.int64)
    MRRand      = rng_migrant.poisson(float(MRpertimestep), size=stream_len).astype(np.int64)
    U           = rng_uniform.random(stream_len).astype(np.float64)

    # ----------------------------
    # 5) Call JIT core (pure given inputs)
    # ----------------------------
    return _simulator_v4_core(
        np.ascontiguousarray(AgentCharacteristics, dtype=np.float64),
        np.ascontiguousarray(ImmuneStatus, dtype=np.int64),
        int(Nagents), int(Nstrains), int(Nst), float(AgeDeath),
        float(Cpertimestep), float(MRpertimestep), float(Precovery), float(Pimmunityloss),
        float(Ptransmission), float(x), float(StrengthImmunity), int(Immunity),
        float(StrengthCrossImmunity), float(prevalence_in_migrants), float(CCC),
        np.ascontiguousarray(time, dtype=np.float64), int(Ntimesteps), float(dt_years),
        int(Ntimesteps_selected),
        np.ascontiguousarray(consultations, dtype=np.int64),
        np.ascontiguousarray(time_obs_idx, dtype=np.int64),
        np.ascontiguousarray(pool, dtype=np.int64),
        np.ascontiguousarray(ContactRand, dtype=np.int64),
        np.ascontiguousarray(MRRand, dtype=np.int64),
        np.ascontiguousarray(U, dtype=np.float64),
    )

def div(SSP: np.ndarray) -> np.ndarray:
    """
    Reciprocal Simpson diversity over time.
    SSP: (Nstrains, T) counts per strain per timestep
    returns: (T,)
    """
    SSP1 = SSP - 1
    N = SSP.sum(axis=0)                 # (T,)
    D = N * (N - 1)
    sumSSP = (SSP * SSP1).sum(axis=0)   # (T,)
    with np.errstate(divide='ignore', invalid='ignore'):
        D = D / sumSSP
        inf_mask = ~np.isfinite(D) & (sumSSP == 0)
        D[inf_mask] = N[inf_mask]
        D[np.isnan(D)] = 0.0
    return D

def plotheatmap(x, y, z, vmax=90, xylim=(-0.03, 10.005, 0.4, 42.6)):
    z = np.array(z, dtype=float).copy()
    z[z == 0] = np.nan  # mask zeros like MATLAB

    # pixel-centered edges
    dx = (x[1] - x[0]) / 2.0 if len(x) > 1 else 0.5
    dy = (y[1] - y[0]) / 2.0 if len(y) > 1 else 0.5
    x_edges = np.concatenate([x - dx, [x[-1] + dx]])     # length M+1
    y_edges = np.concatenate([y - dy, [y[-1] + dy]])     # length N+1

    # reversed grayscale like your MATLAB map
    cmap = plt.cm.get_cmap('gray_r')

    # NOTE: C must be (M,N) where M=len(x_edges)-1 and N=len(y_edges)-1
    # We swapped axes so: x-axis=time (y_edges), y-axis=strain (x_edges)
    mesh = plt.pcolormesh(y_edges, x_edges, z, shading='flat',
                          vmin=0, vmax=vmax, cmap=cmap)
    plt.colorbar(mesh)
    if xylim:
        plt.axis(xylim)


def rmsd(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    return np.sqrt(np.mean(diff**2))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return 100 * np.mean(np.abs((y_pred - y_true) / y_true))


def simulator_v6(AgentCharacteristics, ImmuneStatus, params, rng,
                 specifyPtransmission: int = 0,
                 cross_immunity_effect_on_coinfections: int = 1,
                 ):
    """
    Reproducible version: all randomness comes from a single numpy.random.Generator.

    Returns
    -------
    SSPrev : (Nstrains, Ntimesteps)
    AgentsInfectedByKStrains : (Nstrains, Ntimesteps)
    SSPrev_selected: (Nstrains, Ntimesteps_selected)
    """
    rng = rng  # <-- single RNG for the whole simulation

    (Nagents, Nstrains, Nst, AgeDeath, _NI0, _NR0,
     Cpertimestep, MRpertimestep, Precovery, Pimmunityloss,
     Ptransmission, x, StrengthImmunity, Immunity,
     StrengthCrossImmunity, prevalence_in_migrants, CCC,
     time, Ntimesteps, dt_years) = parameters(params)

    Nagents   = int(Nagents)
    Nstrains  = int(Nstrains)
    Nst       = int(Nst)
    AgeDeath  = float(AgeDeath)
    CCC       = float(CCC)

    # ---------- observation schedule ----------
    Ntimesteps_selected = int(23)
    enrolled = int(548)
    consultations = np.array([27,21,42,51,36,69,
                              122,149,172,170,142,147,
                              40,193,183,211,190,182,
                              199,130,191,188,161], dtype=int)
    time_obs = np.array([1, 32, 62, 93, 123, 154,
                         185, 214, 245, 275, 306, 336,
                         367, 398, 428, 459, 489, 520,
                         551, 579, 610, 640, 671], dtype=int)
    time_obs_idx = time_obs[time_obs > 0] + 365*18 - 1  # 0-based steps

    # enrolled pool is now reproducible
    pool = rng.choice(Nagents, size=enrolled, replace=False)

    # Optionally override Ptransmission
    if specifyPtransmission == 1:
        Ptransmission = 0.0301

    # ---- Cross-immunity-accelerated recovery probability per step ----
    dt_weeks = 1.0 / 7.0
    Rrecovery = -np.log(1.0 - Precovery) / dt_weeks
    if StrengthCrossImmunity != 1:
        Rrecovery_cici = 1.0 / ((1.0 / Rrecovery) * (1.0 - StrengthCrossImmunity))
        Precovery_cici = 1.0 - np.exp(-dt_weeks * Rrecovery_cici)
    else:
        Precovery_cici = 1.0

    # ---- Pre-generated random streams (refilled by *rng*, not global) ----
    ContactRand = rng.poisson(Cpertimestep, size=1_000_000).astype(np.int64)
    MRRand      = rng.poisson(MRpertimestep, size=1_000_000).astype(np.int64)
    SamplingU   = rng.random(1_000_000)
    countCR = 0
    countMR = 0
    countU  = 0

    def _takeU(n: int) -> np.ndarray:
        nonlocal SamplingU, countU
        end = countU + n
        if end > SamplingU.size:
            SamplingU = rng.random(1_000_000)      # refill with rng
            countU = 0
            end = n
        out = SamplingU[countU:end]
        countU = end
        return out

    def _takeCR() -> int:
        nonlocal ContactRand, countCR
        x = ContactRand[countCR]
        countCR += 1
        if countCR >= ContactRand.size:
            ContactRand = rng.poisson(Cpertimestep, size=1_000_000).astype(np.int64)
            countCR = 0
        return int(x)

    def _takeMR() -> int:
        nonlocal MRRand, countMR
        x = MRRand[countMR]
        countMR += 1
        if countMR >= MRRand.size:
            MRRand = rng.poisson(MRpertimestep, size=1_000_000).astype(np.int64)
            countMR = 0
        return int(x)

    # ---- Outputs ----
    SSPrev = np.zeros((Nstrains, Ntimesteps), dtype=float)
    SSPrev_selected = np.zeros((Nstrains, Ntimesteps_selected), dtype=float)
    AgentsInfectedByKStrains = np.zeros((Nstrains, Ntimesteps), dtype=float)

    # t = 0
    BB = AgentCharacteristics[:, :Nstrains]
    SSPrev[:, 0] = BB.sum(axis=0)

    tot0 = BB.sum()
    if tot0 > 1:
        kvec = BB.sum(axis=1).astype(np.int64)
        kvec = kvec[kvec != 0]
        if kvec.size:
            K, counts = np.unique(kvec, return_counts=True)
            AgentsInfectedByKStrains[K - 1, 0] = counts
    elif tot0 == 1:
        AgentsInfectedByKStrains[0, 0] = 1

    # Tracks “fast recovery” flags (CICI)
    CICI = np.zeros_like(BB)

    # ---- Main time loop ----
    ii = 0
    for t in range(Ntimesteps - 1):
        CurrentAC  = AgentCharacteristics.copy()
        CurrentImm = ImmuneStatus.copy()
        DD = CurrentAC[:, :Nst]

        # ===== RECOVERY =====
        inf_norm = (DD > 0) & (CICI == 0)
        inf_cici = (DD > 0) & (CICI > 0)

        r_n_rows, r_n_cols = np.where(inf_norm)
        if r_n_rows.size:
            rec = (rng.random(r_n_rows.size) < Precovery)  # rng, not np.random
            AgentCharacteristics[r_n_rows[rec], r_n_cols[rec]] = 0
            ImmuneStatus[r_n_rows[rec], r_n_cols[rec]] = 1 * Immunity

        r_c_rows, r_c_cols = np.where(inf_cici)
        if r_c_rows.size:
            rec = (rng.random(r_c_rows.size) < Precovery_cici)
            AgentCharacteristics[r_c_rows[rec], r_c_cols[rec]] = 0
            CICI[r_c_rows[rec], r_c_cols[rec]] = 0

        # ===== WANING IMMUNITY =====
        w_rows, w_cols = np.where(CurrentImm == 1)
        if w_rows.size:
            lose = (rng.random(w_rows.size) < Pimmunityloss)
            ImmuneStatus[w_rows[lose], w_cols[lose]] = 0

        # ===== TRANSMISSION =====
        TotalInf = DD.sum(axis=1)
        infected_agents = np.where(TotalInf > 0)[0]

        if infected_agents.size:
            P1 = Ptransmission * np.power((1.0 - TotalInf / CCC), x)
            P1 = np.clip(P1, 0.0, 1.0)

            InfectionProb = np.repeat(P1[:, None], Nstrains, axis=1)

            if StrengthImmunity > 0:
                mask_ss = (CurrentImm == 1)
                InfectionProb[mask_ss] *= (1.0 - StrengthImmunity)

            if StrengthCrossImmunity > 0:
                any_imm = (CurrentImm == 1).any(axis=1)[:, None]
                mask_cs = (CurrentImm == 0) & np.repeat(any_imm, Nstrains, axis=1)
                InfectionProb[mask_cs] *= (1.0 - StrengthCrossImmunity)

            for a in infected_agents:
                infecting_strains = np.where(DD[a, :] > 0)[0]
                if infecting_strains.size == 0:
                    continue

                X = _takeCR()
                if X <= 0:
                    continue

                # contacts (avoid self)
                U1 = _takeU(X)
                contacts = (U1 * (Nagents - 1)).astype(np.int64)
                contacts[contacts >= a] += 1

                # pick a strain among agent's strains
                U2 = _takeU(X)
                if infecting_strains.size == 1:
                    chosen = np.empty(X, dtype=np.int64)
                    chosen.fill(infecting_strains[0])
                else:
                    idx = (U2 * infecting_strains.size).astype(np.int64)
                    idx[idx == infecting_strains.size] = infecting_strains.size - 1
                    chosen = infecting_strains[idx]

                # Bernoulli success
                susc = InfectionProb[contacts, chosen]
                U3 = _takeU(X)

                success_idx = np.where(U3 < susc)[0]
                if success_idx.size:
                    contacts = contacts[success_idx]
                    chosen   = chosen[success_idx]

                    order = np.argsort(contacts)         # stable dedupe
                    contacts = contacts[order]
                    chosen   = chosen[order]
                    keep = np.concatenate(([True], np.diff(contacts) > 0))
                    contacts = contacts[keep]
                    chosen   = chosen[keep]

                    AgentCharacteristics[contacts, chosen] = CurrentAC[contacts, chosen] + 1

                    if cross_immunity_effect_on_coinfections == 1:
                        temp = AgentCharacteristics[contacts, :Nstrains].copy()
                        temp[np.arange(contacts.size), chosen] = 0
                        temp[temp > 0] = 1
                        CICI[contacts, :] = np.clip(CICI[contacts, :] + temp, 0, 1)

                    InfectionProb[contacts, :] = 0.0

        # ===== AGE, DEATH, BIRTH =====
        AgentCharacteristics[:, Nstrains] = dt_years + CurrentAC[:, Nstrains]
        dead = np.where(AgentCharacteristics[:, Nstrains] > AgeDeath)[0]
        if dead.size:
            AgentCharacteristics[dead, :Nstrains] = 0
            ImmuneStatus[dead, :] = 0
            AgentCharacteristics[dead, Nstrains] = 0.001
            CICI[dead, :] = 0

        # ===== MIGRATION =====
        NumMig = _takeMR()
        if NumMig > 0:
            if NumMig >= Nagents:
                migrants = rng.permutation(Nagents)      # rng, not np.random
            else:
                migrants = rng.choice(Nagents, size=NumMig, replace=False)

            infected_mig = (rng.random(NumMig) < prevalence_in_migrants)
            n_im = int(infected_mig.sum())
            if n_im > 0:
                mig_strains = rng.integers(0, Nst, size=n_im)  # 0..Nst-1

            cm = ci = 0
            for m in range(NumMig):
                idx = migrants[m]
                ImmuneStatus[idx, :] = 0
                CICI[idx, :] = 0
                AgentCharacteristics[idx, Nstrains] = rng.random() * AgeDeath
                AgentCharacteristics[idx, :Nstrains] = 0
                if infected_mig[cm]:
                    AgentCharacteristics[idx, mig_strains[ci]] = 1
                    ci += 1
                cm += 1

        # ===== RECORDING =====
        BB = AgentCharacteristics[:, :Nstrains]
        SSPrev[:, t + 1] = BB.sum(axis=0)

        # if ii < Ntimesteps_selected and t + 1 == time_obs_idx[ii]:
        #     pool_selected = rng.choice(pool, size=consultations[ii], replace=False)
        #     AC_selected = BB[pool_selected]
        #     SSPrev_selected[:, ii] = AC_selected.sum(axis=0)
        #     ii += 1

        if ii < Ntimesteps_selected and t + 1 == time_obs_idx[ii]:
            pool_selected = rng.choice(pool, size=consultations[ii], replace=False)
            AC_selected = BB[pool_selected]  # (consultations[ii] × 42)

            # For each agent, randomly select one strain if co-infected
            AC_single = np.zeros_like(AC_selected)  # (consultations[ii] × 42)

            for agent_idx in range(AC_selected.shape[0]):
                # Find all strains this agent is infected with
                infected_strains = np.where(AC_selected[agent_idx, :] > 0)[0]

                if infected_strains.size > 0:
                    # Randomly pick one strain
                    chosen_strain = rng.choice(infected_strains)
                    AC_single[agent_idx, chosen_strain] = 1

            SSPrev_selected[:, ii] = AC_single.sum(axis=0)
            ii += 1

        tot = BB.sum()
        if tot > 1:
            kvec = BB.sum(axis=1).astype(np.int64)
            kvec = kvec[kvec != 0]
            if kvec.size:
                K, counts = np.unique(kvec, return_counts=True)
                AgentsInfectedByKStrains[K - 1, t + 1] = counts
        elif tot == 1:
            AgentsInfectedByKStrains[0, t + 1] = 1

    return SSPrev_selected, SSPrev, AgentsInfectedByKStrains


# @njit(cache=True)
# def _simulator_v6_core(
#     AgentCharacteristics,
#     ImmuneStatus,
#     Nagents, Nstrains, Nst, AgeDeath,
#     Cpertimestep, MRpertimestep, Precovery, Pimmunityloss,
#     Ptransmission, x, StrengthImmunity, Immunity,
#     StrengthCrossImmunity, prevalence_in_migrants, CCC,
#     time, Ntimesteps, dt_years,
#     Ntimesteps_selected,
#     consultations,
#     time_obs_idx,
#     pool,
#     ContactRand, MRRand, U
# ):
#     # ---- Everything identical to _simulator_v4_core until RECORDING ----
#     cr_ptr = np.zeros(1, dtype=np.int64)
#     mr_ptr = np.zeros(1, dtype=np.int64)
#     u_ptr  = np.zeros(1, dtype=np.int64)

#     dt_weeks = 1.0 / 7.0
#     Rrecovery = -np.log(1.0 - Precovery) / dt_weeks
#     if StrengthCrossImmunity != 1.0:
#         Rrecovery_cici = 1.0 / ((1.0 / Rrecovery) * (1.0 - StrengthCrossImmunity))
#         Precovery_cici = 1.0 - np.exp(-dt_weeks * Rrecovery_cici)
#     else:
#         Precovery_cici = 1.0

#     SSPrev = np.zeros((Nstrains, Ntimesteps), dtype=np.float64)
#     SSPrev_selected = np.zeros((Nstrains, Ntimesteps_selected), dtype=np.float64)
#     AgentsInfectedByKStrains = np.zeros((Nstrains, Ntimesteps), dtype=np.float64)

#     BB = AgentCharacteristics[:, :Nstrains]
#     SSPrev[:, 0] = BB.sum(axis=0)

#     tot0 = BB.sum()
#     if tot0 > 1:
#         kvec = BB.sum(axis=1).astype(np.int64)
#         kmax = 0
#         for i in range(kvec.size):
#             if kvec[i] > kmax:
#                 kmax = kvec[i]
#         if kmax > 0:
#             counts = np.zeros(kmax, dtype=np.int64)
#             for i in range(kvec.size):
#                 kv = kvec[i]
#                 if kv > 0:
#                     counts[kv - 1] += 1
#             Klen = counts.size if counts.size < Nstrains else Nstrains
#             for i in range(Klen):
#                 AgentsInfectedByKStrains[i, 0] = counts[i]
#     elif tot0 == 1:
#         AgentsInfectedByKStrains[0, 0] = 1

#     CICI = np.zeros_like(BB)

#     tmpU  = np.empty(4096, dtype=np.float64)
#     tmpU2 = np.empty(4096, dtype=np.float64)
#     tmpU3 = np.empty(4096, dtype=np.float64)

#     ii = 0
#     for t in range(Ntimesteps - 1):

#         # ===== All sections identical to _simulator_v4_core =====
#         # (RECOVERY, WANING IMMUNITY, TRANSMISSION, AGE/DEATH/BIRTH, MIGRATION)
#         # ... [same code as _simulator_v4_core] ...

#         # ===== RECORDING =====
#         BB = AgentCharacteristics[:, :Nstrains]
#         for j in range(Nstrains):
#             SSPrev[j, t + 1] = BB[:, j].sum()

#         # ✅ MODIFIED: single-strain selection per agent
#         if ii < Ntimesteps_selected and (t + 1) == time_obs_idx[ii]:
#             k = consultations[ii]
#             sel = _fisher_yates_first_k(pool, k, U, u_ptr)

#             for r in range(sel.size):
#                 agent = sel[r]

#                 # count infected strains for this agent
#                 ns = 0
#                 for s in range(Nstrains):
#                     if BB[agent, s] > 0:
#                         ns += 1

#                 if ns > 0:
#                     # randomly pick one strain using U stream
#                     u = _take1d(U, u_ptr)
#                     idx = int(u * ns)
#                     if idx >= ns:
#                         idx = ns - 1

#                     # find the idx-th infected strain
#                     count = 0
#                     for s in range(Nstrains):
#                         if BB[agent, s] > 0:
#                             if count == idx:
#                                 SSPrev_selected[s, ii] += 1.0
#                                 break
#                             count += 1

#             ii += 1

#         # ===== AgentsInfectedByKStrains (identical to _simulator_v4_core) =====
#         tot = 0.0
#         for i in range(Nagents):
#             for j in range(Nstrains):
#                 tot += BB[i, j]
#         if tot > 1:
#             kmax = 0
#             for i in range(Nagents):
#                 kv = 0
#                 for j in range(Nstrains):
#                     if BB[i, j] > 0.0:
#                         kv += 1
#                 if kv > kmax:
#                     kmax = kv
#             if kmax > 0:
#                 counts = np.zeros(kmax, dtype=np.int64)
#                 for i in range(Nagents):
#                     kv = 0
#                     for j in range(Nstrains):
#                         if BB[i, j] > 0.0:
#                             kv += 1
#                     if kv > 0:
#                         counts[kv - 1] += 1
#                 Klen = counts.size if counts.size < Nstrains else Nstrains
#                 for i in range(Klen):
#                     AgentsInfectedByKStrains[i, t + 1] = counts[i]
#         elif tot == 1:
#             AgentsInfectedByKStrains[0, t + 1] = 1

#     return SSPrev_selected, SSPrev, AgentsInfectedByKStrains

@njit(cache=True)
def _simulator_v6_core(
    AgentCharacteristics,
    ImmuneStatus,
    Nagents, Nstrains, Nst, AgeDeath,
    Cpertimestep, MRpertimestep, Precovery, Pimmunityloss,
    Ptransmission, x, StrengthImmunity, Immunity,
    StrengthCrossImmunity, prevalence_in_migrants, CCC,
    time, Ntimesteps, dt_years,
    Ntimesteps_selected,
    consultations,
    time_obs_idx,
    pool,
    ContactRand, MRRand, U,
    U_obs  # ← separate obs stream
):
    # pointer "indices" into random streams
    cr_ptr = np.zeros(1, dtype=np.int64)
    mr_ptr = np.zeros(1, dtype=np.int64)
    u_ptr  = np.zeros(1, dtype=np.int64)
    u_obs_ptr = np.zeros(1, dtype=np.int64)  # ← separate pointer for obs

    # ---- Cross-immunity-accelerated recovery probability per step ----
    dt_weeks = 1.0 / 7.0
    Rrecovery = -np.log(1.0 - Precovery) / dt_weeks
    if StrengthCrossImmunity != 1.0:
        Rrecovery_cici = 1.0 / ((1.0 / Rrecovery) * (1.0 - StrengthCrossImmunity))
        Precovery_cici = 1.0 - np.exp(-dt_weeks * Rrecovery_cici)
    else:
        Precovery_cici = 1.0

    # ---- Outputs ----
    SSPrev = np.zeros((Nstrains, Ntimesteps), dtype=np.float64)
    SSPrev_selected = np.zeros((Nstrains, Ntimesteps_selected), dtype=np.float64)
    AgentsInfectedByKStrains = np.zeros((Nstrains, Ntimesteps), dtype=np.float64)

    # t = 0
    BB = AgentCharacteristics[:, :Nstrains]
    SSPrev[:, 0] = BB.sum(axis=0)

    tot0 = BB.sum()
    if tot0 > 1:
        kvec = BB.sum(axis=1).astype(np.int64)
        kmax = 0
        for i in range(kvec.size):
            if kvec[i] > kmax:
                kmax = kvec[i]
        if kmax > 0:
            counts = np.zeros(kmax, dtype=np.int64)
            for i in range(kvec.size):
                kv = kvec[i]
                if kv > 0:
                    counts[kv - 1] += 1
            Klen = counts.size if counts.size < Nstrains else Nstrains
            for i in range(Klen):
                AgentsInfectedByKStrains[i, 0] = counts[i]
    elif tot0 == 1:
        AgentsInfectedByKStrains[0, 0] = 1

    # Tracks "fast recovery" flags (CICI)
    CICI = np.zeros_like(BB)

    # prealloc tmp buffers
    tmpU  = np.empty(4096, dtype=np.float64)
    tmpU2 = np.empty(4096, dtype=np.float64)
    tmpU3 = np.empty(4096, dtype=np.float64)

    ii = 0
    for t in range(Ntimesteps - 1):
        CurrentAC  = AgentCharacteristics.copy()
        CurrentImm = ImmuneStatus.copy()
        DD = CurrentAC[:, :Nst]

        # ===== RECOVERY =====
        for i in range(DD.shape[0]):
            for j in range(DD.shape[1]):
                if DD[i, j] > 0:
                    if CICI[i, j] == 0:
                        if _take1d(U, u_ptr) < Precovery:
                            AgentCharacteristics[i, j] = 0.0
                            if Immunity == 1:
                                ImmuneStatus[i, j] = 1
                    else:
                        if _take1d(U, u_ptr) < Precovery_cici:
                            AgentCharacteristics[i, j] = 0.0
                            CICI[i, j] = 0

        # ===== WANING IMMUNITY =====
        for i in range(CurrentImm.shape[0]):
            for j in range(CurrentImm.shape[1]):
                if CurrentImm[i, j] == 1:
                    if _take1d(U, u_ptr) < Pimmunityloss:
                        ImmuneStatus[i, j] = 0

        # ===== TRANSMISSION =====
        TotalInf = DD.sum(axis=1)
        na = 0
        for i in range(TotalInf.size):
            if TotalInf[i] > 0:
                na += 1
        if na > 0:
            infected_agents = np.empty(na, dtype=np.int64)
            k = 0
            for i in range(TotalInf.size):
                if TotalInf[i] > 0:
                    infected_agents[k] = i
                    k += 1

            P1 = np.empty(Nagents, dtype=np.float64)
            for i in range(Nagents):
                s = 1.0 - TotalInf[i] / CCC
                if s < 0.0:
                    s = 0.0
                P1[i] = Ptransmission * (s ** x)
                if P1[i] < 0.0:
                    P1[i] = 0.0
                if P1[i] > 1.0:
                    P1[i] = 1.0

            InfectionProb = np.empty((Nagents, Nstrains), dtype=np.float64)
            for i in range(Nagents):
                for j in range(Nstrains):
                    InfectionProb[i, j] = P1[i]

            if StrengthImmunity > 0.0:
                for i in range(Nagents):
                    for j in range(Nstrains):
                        if CurrentImm[i, j] == 1:
                            InfectionProb[i, j] *= (1.0 - StrengthImmunity)

            if StrengthCrossImmunity > 0.0:
                any_imm = np.zeros(Nagents, dtype=np.uint8)
                for i in range(Nagents):
                    ai = 0
                    for j in range(Nstrains):
                        if CurrentImm[i, j] == 1:
                            ai = 1
                            break
                    any_imm[i] = ai
                for i in range(Nagents):
                    if any_imm[i] == 1:
                        for j in range(Nstrains):
                            if CurrentImm[i, j] == 0:
                                InfectionProb[i, j] *= (1.0 - StrengthCrossImmunity)

            for a in infected_agents:
                ns = 0
                for j in range(DD.shape[1]):
                    if DD[a, j] > 0:
                        ns += 1
                if ns == 0:
                    continue
                infecting_strains = np.empty(ns, dtype=np.int64)
                kk = 0
                for j in range(DD.shape[1]):
                    if DD[a, j] > 0:
                        infecting_strains[kk] = j
                        kk += 1

                X = _take1d(ContactRand, cr_ptr)
                if X <= 0:
                    continue

                if X > tmpU.size:
                    tmpU  = np.empty(X, dtype=np.float64)
                    tmpU2 = np.empty(X, dtype=np.float64)
                    tmpU3 = np.empty(X, dtype=np.float64)

                _takeU_many(U, u_ptr, X, tmpU)
                contacts = (tmpU[:X] * (Nagents - 1)).astype(np.int64)
                for i in range(X):
                    if contacts[i] >= a:
                        contacts[i] += 1

                _takeU_many(U, u_ptr, X, tmpU2)
                chosen = np.empty(X, dtype=np.int64)
                if infecting_strains.size == 1:
                    s0 = infecting_strains[0]
                    for i in range(X):
                        chosen[i] = s0
                else:
                    m = infecting_strains.size
                    for i in range(X):
                        idx = int(tmpU2[i] * m)
                        if idx >= m:
                            idx = m - 1
                        chosen[i] = infecting_strains[idx]

                _takeU_many(U, u_ptr, X, tmpU3)
                succ_n = 0
                for i in range(X):
                    if tmpU3[i] < InfectionProb[contacts[i], chosen[i]]:
                        succ_n += 1
                if succ_n == 0:
                    continue

                succ_contacts = np.empty(succ_n, dtype=np.int64)
                succ_chosen   = np.empty(succ_n, dtype=np.int64)
                kk = 0
                for i in range(X):
                    if tmpU3[i] < InfectionProb[contacts[i], chosen[i]]:
                        succ_contacts[kk] = contacts[i]
                        succ_chosen[kk]   = chosen[i]
                        kk += 1

                keep = np.ones(succ_n, dtype=np.uint8)
                for i in range(1, succ_n):
                    ci = succ_contacts[i]
                    for j in range(i):
                        if succ_contacts[j] == ci:
                            keep[i] = 0
                            break

                for i in range(succ_n):
                    if keep[i] == 1:
                        c = succ_contacts[i]
                        s = succ_chosen[i]
                        AgentCharacteristics[c, s] = CurrentAC[c, s] + 1.0

                for i in range(succ_n):
                    if keep[i] == 1:
                        c = succ_contacts[i]
                        s = succ_chosen[i]
                        for j in range(Nstrains):
                            if j == s:
                                continue
                            if AgentCharacteristics[c, j] > 0.0:
                                CICI[c, j] = 1

                for i in range(succ_n):
                    if keep[i] == 1:
                        c = succ_contacts[i]
                        for j in range(Nstrains):
                            InfectionProb[c, j] = 0.0

        # ===== AGE, DEATH, BIRTH =====
        for i in range(Nagents):
            AgentCharacteristics[i, Nstrains] = dt_years + CurrentAC[i, Nstrains]
            if AgentCharacteristics[i, Nstrains] > AgeDeath:
                for j in range(Nstrains):
                    AgentCharacteristics[i, j] = 0.0
                    ImmuneStatus[i, j] = 0
                    CICI[i, j] = 0
                AgentCharacteristics[i, Nstrains] = 0.001

        # ===== MIGRATION =====
        NumMig = _take1d(MRRand, mr_ptr)
        if NumMig > 0:
            mig_idx = _fisher_yates_first_k(
                np.arange(Nagents, dtype=np.int64), NumMig, U, u_ptr)

            infected_mig = np.zeros(NumMig, dtype=np.uint8)
            for i in range(NumMig):
                infected_mig[i] = 1 if _take1d(U, u_ptr) < prevalence_in_migrants else 0

            n_im = 0
            for i in range(NumMig):
                if infected_mig[i] == 1:
                    n_im += 1
            mig_strains = np.empty(n_im, dtype=np.int64)
            kk = 0
            for i in range(NumMig):
                if infected_mig[i] == 1:
                    u = _take1d(U, u_ptr)
                    s = int(u * Nst)
                    if s >= Nst:
                        s = Nst - 1
                    mig_strains[kk] = s
                    kk += 1

            ci = 0
            for m in range(NumMig):
                idxm = mig_idx[m]
                for j in range(Nstrains):
                    ImmuneStatus[idxm, j] = 0
                    CICI[idxm, j] = 0
                    AgentCharacteristics[idxm, j] = 0.0
                AgentCharacteristics[idxm, Nstrains] = _take1d(U, u_ptr) * AgeDeath
                if infected_mig[m] == 1:
                    AgentCharacteristics[idxm, mig_strains[ci]] = 1.0
                    ci += 1

        # ===== RECORDING =====
        BB = AgentCharacteristics[:, :Nstrains]
        for j in range(Nstrains):
            SSPrev[j, t + 1] = BB[:, j].sum()

        # ✅ MODIFIED: single-strain selection using separate U_obs stream
        if ii < Ntimesteps_selected and (t + 1) == time_obs_idx[ii]:
            k = consultations[ii]
            # ✅ pool selection uses U stream (same as v5_numba)
            sel = _fisher_yates_first_k(pool, k, U, u_ptr)

            for r in range(sel.size):
                agent = sel[r]
                ns = 0
                for s in range(Nstrains):
                    if BB[agent, s] > 0:
                        ns += 1
                if ns > 0:
                    # ✅ strain selection uses U_obs stream (separate)
                    u = _take1d(U_obs, u_obs_ptr)
                    idx = int(u * ns)
                    if idx >= ns:
                        idx = ns - 1
                    count = 0
                    for s in range(Nstrains):
                        if BB[agent, s] > 0:
                            if count == idx:
                                SSPrev_selected[s, ii] += 1.0
                                break
                            count += 1
            ii += 1

        # ===== AgentsInfectedByKStrains =====
        tot = 0.0
        for i in range(Nagents):
            for j in range(Nstrains):
                tot += BB[i, j]
        if tot > 1:
            kmax = 0
            for i in range(Nagents):
                kv = 0
                for j in range(Nstrains):
                    if BB[i, j] > 0.0:
                        kv += 1
                if kv > kmax:
                    kmax = kv
            if kmax > 0:
                counts = np.zeros(kmax, dtype=np.int64)
                for i in range(Nagents):
                    kv = 0
                    for j in range(Nstrains):
                        if BB[i, j] > 0.0:
                            kv += 1
                    if kv > 0:
                        counts[kv - 1] += 1
                Klen = counts.size if counts.size < Nstrains else Nstrains
                for i in range(Klen):
                    AgentsInfectedByKStrains[i, t + 1] = counts[i]
        elif tot == 1:
            AgentsInfectedByKStrains[0, t + 1] = 1

    return SSPrev_selected, SSPrev, AgentsInfectedByKStrains

    
def simulator_v6_numba(AgentCharacteristics, ImmuneStatus, params,
                       specifyPtransmission: int = 0,
                       cross_immunity_effect_on_coinfections: int = 1,
                       seed: int = 123,
                       stream_len: int = 2_000_000):

    # ----------------------------
    # 1) Unpack parameters
    # ----------------------------
    (Nagents, Nstrains, Nst, AgeDeath, _NI0, _NR0,
     Cpertimestep, MRpertimestep, Precovery, Pimmunityloss,
     Ptransmission, x, StrengthImmunity, Immunity,
     StrengthCrossImmunity, prevalence_in_migrants, CCC,
     time, Ntimesteps, dt_years) = parameters(params)

    if specifyPtransmission == 1:
        Ptransmission = 0.0301

    # ----------------------------
    # 2) Study schedule
    # ----------------------------
    Ntimesteps_selected = 23
    consultations = np.array([27, 21, 42, 51, 36, 69,
                              122,149,172,170,142,147,
                              40,193,183,211,190,182,
                              199,130,191,188,161], dtype=np.int64)
    time_obs = np.array([1, 32, 62, 93, 123, 154,
                         185, 214, 245, 275, 306, 336,
                         367, 398, 428, 459, 489, 520,
                         551, 579, 610, 640, 671], dtype=np.int64)
    time_obs_idx = (time_obs[time_obs > 0] + 365*18 - 1).astype(np.int64)

    # ----------------------------
    # 3) Master seed → 5 independent child RNGs
    #    (added ss_obs for observation sampling)
    # ----------------------------
    ss = SeedSequence(int(seed))
    ss_pool, ss_contact, ss_migrant, ss_uniform, ss_obs = ss.spawn(5)  # ← 5 streams

    rng_pool    = default_rng(ss_pool)
    rng_contact = default_rng(ss_contact)
    rng_migrant = default_rng(ss_migrant)
    rng_uniform = default_rng(ss_uniform)
    rng_obs     = default_rng(ss_obs)  # ← separate stream for observation

    # ----------------------------
    # 4) Random draws
    # ----------------------------
    enrolled = 548
    pool = rng_pool.choice(int(Nagents), size=enrolled, replace=False).astype(np.int64)

    ContactRand = rng_contact.poisson(float(Cpertimestep), size=stream_len).astype(np.int64)
    MRRand      = rng_migrant.poisson(float(MRpertimestep), size=stream_len).astype(np.int64)
    U           = rng_uniform.random(stream_len).astype(np.float64)

    # ← separate observation random stream
    # max consultations = 211, max strains per agent = 42
    # so 23 * 211 * 42 is more than enough
    U_obs = rng_obs.random(23 * 211 * 42).astype(np.float64)

    # ----------------------------
    # 5) Call JIT core
    # ----------------------------
    return _simulator_v6_core(
        np.ascontiguousarray(AgentCharacteristics, dtype=np.float64),
        np.ascontiguousarray(ImmuneStatus, dtype=np.int64),
        int(Nagents), int(Nstrains), int(Nst), float(AgeDeath),
        float(Cpertimestep), float(MRpertimestep), float(Precovery), float(Pimmunityloss),
        float(Ptransmission), float(x), float(StrengthImmunity), int(Immunity),
        float(StrengthCrossImmunity), float(prevalence_in_migrants), float(CCC),
        np.ascontiguousarray(time, dtype=np.float64), int(Ntimesteps), float(dt_years),
        int(Ntimesteps_selected),
        np.ascontiguousarray(consultations, dtype=np.int64),
        np.ascontiguousarray(time_obs_idx, dtype=np.int64),
        np.ascontiguousarray(pool, dtype=np.int64),
        np.ascontiguousarray(ContactRand, dtype=np.int64),
        np.ascontiguousarray(MRRand, dtype=np.int64),
        np.ascontiguousarray(U, dtype=np.float64),
        np.ascontiguousarray(U_obs, dtype=np.float64),  # ← separate obs stream
    )