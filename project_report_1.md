---
title: Characterizing quantum sensors with noise
subtitle: report 1
author: Ian Mitchell
titlepage: true
date: 2024-03-27
csl: /home/pines/.pandoc/csl/nature.csl
bibliography: ref.bib
crossReference: true
reportNo: linear algebra group project
numberSections: true
---

Assume that we have a single atom interacting with a single frequency of light 
within an optical cavity. We can write the interactions between the atom and
cavity as[@scully_quantum_1994]
$$ \mathcal{H}_\text{JC} = \hbar \omega a^\dagger a
    + \frac{\hbar\Delta}{2} \sigma^z
    + g\left( a\sigma^- + a\sigma^+ \right) \,,$$ {#eq:jaynes_cummings}
where $a^\dagger$ the creation operator for a photon, $\sigma^-$ is the raising
operator for the atom, $\omega$ is the cavity frequency, $\Delta$ is the
detuning of the cavity frequency from the atom's excited state,
and $g$ is the interaction strength between the atom and cavity. Next, assume
that our atom has some Zeeman splitting,[@griffiths_introduction_2018] we may
then write our total Hamiltonian---neglecting the atom's own angular
momentum---for this system as
$$ \mathcal{H} = \mathcal{H}_\text{JC}
    - \frac{e_0}{m} \mathbf{S} \cdot \mathbf{B}
    + V(t) \,, $$ {#eq:jc_with_zeeman}
where $e_0$ is the electron charge, $m$ is the atom's mass, $\mathbf{S}$ is the
electron's total spin, and $\mathbf{B}$ is the input magnetic field, and $V(t)$
is a noise function such that $\expval{V(t + dt) V(t)} = 0$.

This Hamiltonian serves as a decent prototype for a quantum
sensor,[@degen_quantum_2017] since its energy levels change directly with
an applied magnetic field. If we know the change in energy levels for an atom
with zero noise, or at the very least *reduced* noise, then we can characterize
the noise in a quantum sensor based off of the change in the sensor's 
sensitivity, which can be calculated by getting a least-squares fit of the
energy of incident photons versus photons emitted from the atom.

If one is assuming an open quantum system---as quantum systems are in the real
world---then it is prudent to use the Lindblad equation,[@manzano_short_2020]
$$ \frac{d\rho(t)}{dt} = -\frac{i}{\hbar} \comm{\mathcal{H}}{\rho}
    + \sum_j \left( L_j \rho L_j^\dagger
        - \frac{1}{2} \acomm{L_j^\dagger L_j}{\rho}\right) \,,
$$ {#eq:lindblad}
where $L_i$ is a "jump-operator" describing how a system interacts with its
environment.

Least-squares fits can be calculated using the singular value decomposition
(SVD),[@lay_linear_2016; @verschelde_numerical_2022] where we assume that an
$m \times n$ matrix with rank $r$, $A$, can be broken up into three
components,[@lay_linear_2016] $A = U \Sigma V^\text{T}$, where $U$ and $V$ are
orthogonal, and $\Sigma$ is a matrix with the single values of $A$, usually
expressed as
$$ \Sigma = \begin{bmatrix}
D & 0 \\
0 & 0
\end{bmatrix} \,, $$
where
$$ D = \begin{bmatrix}
\sigma_1 & \cdots & 0 \\
\vdots & \ddots & \vdots \\
0 & \cdots & \sigma_n
\end{bmatrix} \,, $$
and $\sigma_n = \sqrt{\lambda_n}$ are the singular values of $A$. $U$ and
$V$ are constructed by the left and right singular vectors of $A$, respectively.
A full treatment of least-squares will be covered in the next report, as I
scrapped the entire report I worked on last week in favor of this, and I've
spent all morning---and most of this afternoon---finishing *this* particular
report. (I was doing some big power-grid optimization problem, and honestly it
was just really bad.)

Calculations will be performed numerically, since finding solutions to this
problem will be *extremely* nontrivial analytically even for the most talented
physicists and mathematicians. In the next report, I will be showing my
calculations, going into more detail with the singular value decomposition,
and expanding my analysis to cover an extended, many-body interacting
Hamiltonian. I would like to apologize for how brief and sparse this report is.



# References
