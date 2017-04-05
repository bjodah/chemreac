<%doc> coding: utf-8; compile-command: "cd ~/vc/chemreac; rm -r build/; PYTHONPATH=~/vc/pycompilation:~/vc/pycodeexport python setup.py build_ext -i; cd build/temp.linux-x86_64-2.7; /usr/bin/g++ -O2 -c -std=c++0x -fPIC -Wall -Wextra -o src/chemreac.o -I/home/bjorn/vc/chemreac/src -I/home/bjorn/vc/chemreac/src/finitediff/include -I/home/bjorn/vc/chemreac/src/finitediff/external/newton_interval/include -I/home/bjorn/.local/lib/python2.7/site-packages/numpy/core/include src/chemreac.cpp" -*-
// This is a templated source file.
// Render template using Mako (Python templating engine)
</%doc>
// ${'{0} eval: (read-only-mode) {0}'.format('-*-')}
// ${_warning_in_the_generated_file_not_to_edit}

#include <algorithm> // std::count
//#include <vector>    // std::vector
#include <algorithm> // std::max, std::min
#include <cmath>
#include <cstdlib> // free,  C++11 aligned_alloc
#include "chemreac.hpp"
#include "finitediff_templated.hpp" // fintie differences

#include <iostream> //DEBUG


#if defined(WITH_DATA_DUMPING)
#include <cstdio>
#include <iostream>
#include <sstream>
#include <iomanip>
#define PRINT_ARR(ARR, LARR) for(int i_=0; i_<LARR; ++i_) {std::cout << ARR[i_] << " ";}; std::cout << std::endl;
#include "chemreac_util.h" // save_array, load_array
#endif

%if WITH_OPENMP:
#include <omp.h>
%else:
#define omp_get_thread_num() 0
%endif


namespace chemreac {
using std::vector;
using std::count;
using std::min;
using std::max;

#include <cstdio>

#define expb(arg) (use_log2 ? std::exp2(arg) : std::exp(arg))
#define logb(arg) (use_log2 ? std::log2(arg) : std::pow(2, arg))

// 1D discretized reaction diffusion
template<typename Real_t>
ReactionDiffusion<Real_t>::ReactionDiffusion(
    uint n,
    const vector<vector<uint> > stoich_active,
    const vector<vector<uint> > stoich_prod,
    vector<Real_t> k,
    uint N,
    vector<Real_t> D,
    const vector<int> z_chg,
    vector<Real_t> mobility,
    const vector<Real_t> x, // separation
    vector<vector<uint> > stoich_inact, // vectors of size 0 in stoich_actv_ => "copy from stoich_reac"
    int geom_,
    bool logy,
    bool logt,
    bool logx,
    uint nstencil,
    bool lrefl,
    bool rrefl,
    bool auto_efield,
    pair<Real_t, Real_t> surf_chg,
    Real_t eps_rel,
    Real_t faraday_const,
    Real_t vacuum_permittivity,
    vector<vector<Real_t>> g_values,
    vector<int> g_value_parents,
    vector<vector<Real_t>> fields,
    vector<int> modulated_rxns,
    vector<vector<Real_t> > modulation,
    Real_t ilu_limit,
    uint n_jac_diags,
    bool use_log2):
    n(n), N(N), nstencil(nstencil), nsidep((nstencil-1)/2), nr(stoich_active.size()),
    logy(logy), logt(logt), logx(logx), stoich_active(stoich_active),
    stoich_inact(stoich_inact), stoich_prod(stoich_prod),
    k(k),  D(D), z_chg(z_chg), mobility(mobility), x(x), lrefl(lrefl), rrefl(rrefl),
    auto_efield(auto_efield),
    surf_chg(surf_chg), eps_rel(eps_rel), faraday_const(faraday_const),
    vacuum_permittivity(vacuum_permittivity),
    g_value_parents(g_value_parents), modulated_rxns(modulated_rxns), modulation(modulation),
    ilu_limit(ilu_limit), n_jac_diags((n_jac_diags == 0) ? nsidep : n_jac_diags), use_log2(use_log2),
    efield(new Real_t[N]), netchg(new Real_t[N])
{
    if (N == 0) throw std::logic_error("Zero bins sounds boring.");
    if (N == 2) throw std::logic_error("2nd order PDE requires at least 3 stencil points.");
    if (nstencil % 2 == 0) throw std::logic_error("Only odd number of stencil points supported");
    if ((N == 1) && (nstencil != 1)) throw std::logic_error("You must set nstencil=1 for N=1");
    if ((N > 1) && (nstencil <= 1)) throw std::logic_error("You must set nstencil>1 for N>1");
    if (stoich_active.size() != stoich_prod.size())
        throw std::length_error(
            "stoich_active and stoich_prod of different sizes.");
    if (k.size() != stoich_prod.size())
        throw std::length_error(
            "k and stoich_prod of different sizes.");
    if (N>1){
        if (D.size() != n)
            throw std::length_error(
                "Length of D does not match number of species.");
        if (mobility.size() != n)
            throw std::length_error(
                "Length of mobility does not match number of species.");
        if (z_chg.size() != n)
            throw std::length_error(
                "Length of z_chg does not match number of species.");
        if (x.size() != N + 1)
            throw std::length_error(
                "Number bin edges != number of compartments + 1.");
    }
    switch(geom_) {
    case 0:
        geom = Geom::FLAT;
        break;
    case 1:
        geom = Geom::CYLINDRICAL;
        break;
    case 2:
        geom = Geom::SPHERICAL;
        break;
    case 3:
        geom = Geom::PERIODIC;
        break;
    default:
        throw std::logic_error("Unknown geom.");
    }

    // Finite difference scheme
    D_weight = new Real_t[nstencil*N];
    A_weight = new Real_t[nstencil*N];
    for (uint i=0; i<N; ++i) efield[i] = 0.0;
    xc = new Real_t[nsidep + N + nsidep]; // xc padded with virtual bins
    for (uint i=0; i<N; ++i)
        xc[nsidep + i] = (x[i] + x[i + 1])/2;

    for (uint i=0; i<nsidep; ++i){
        // reflection
        xc[nsidep - i - 1] = 2*x[0] - xc[nsidep + i];
        xc[nsidep + i + N] = 2*x[N] - xc[nsidep + N - i - 1];
    }

    // Precalc coeffs for Jacobian for current geom.
    // not centered diffs close to boundaries
    for (uint bi=0; bi<N; bi++)
        apply_fd_(bi);

    // Stoichiometry
    for (uint ri=0; ri<nr; ++ri){
        for (auto si=stoich_active[ri].begin(); si != stoich_active[ri].end(); ++si)
            if (*si > n-1)
                throw std::logic_error("At least one species index in stoich_active > (n-1)");
        for (auto si=stoich_prod[ri].begin(); si != stoich_prod[ri].end(); ++si)
            if (*si > n-1)
                throw std::logic_error("At least one species index in stoich_prod > (n-1)");
        for (auto si=stoich_inact[ri].begin(); si != stoich_inact[ri].end(); ++si)
            if (*si > n-1)
                throw std::logic_error("At least one species index in stoich_inact > (n-1)");
    }

    coeff_active = new int[nr*n];
    coeff_prod = new int[nr*n];
    coeff_total = new int[nr*n];
    coeff_inact = new int[nr*n];

    stoich_inact.reserve(nr);
    for (uint rxni=0; rxni<nr; ++rxni){ // reaction index
        for (uint si=0; si<n; ++si){ // species index
            coeff_active[rxni*n+si] = count(stoich_active[rxni].begin(),
                                          stoich_active[rxni].end(), si);
            coeff_inact[rxni*n+si] = count(stoich_inact[rxni].begin(),
                                            stoich_inact[rxni].end(), si);
            coeff_prod[rxni*n+si] = count(stoich_prod[rxni].begin(),
                                        stoich_prod[rxni].end(), si);
            coeff_total[rxni*n+si] = coeff_prod[rxni*n+si] -\
                coeff_active[rxni*n+si] - coeff_inact[rxni*n+si];
        }
    }

    // Handle g_values
    if (fields.size() != g_values.size())
        throw std::logic_error("fields and g_values need to be of equal length");
    if (this->g_value_parents.size() != g_values.size())
        throw std::logic_error("g_value_parents and g_values need to be of equal length");

    for (const auto& gs : g_values)
        if (gs.size() != n)
            throw std::logic_error("vectors in g_values need to be of length n");

    for (const auto& fs : fields)
        if (fs.size() != N)
            throw std::logic_error("vectors in fields need to be of length N");

    this->g_values = g_values;
    this->fields = fields;

    // Sanity check modulation
    for (const auto rxni : this->modulated_rxns)
        if (rxni >= (int)nr || rxni < 0)
            throw std::logic_error("illegal reaction index in modulated_rxns");
    if (this->modulation.size() != this->modulated_rxns.size())
        throw std::logic_error("modulation size differs from modulated_rxns");
    for (const auto& mdltn : this->modulation)
        if (mdltn.size() != N)
            throw std::logic_error("illegally sized vector in modulation");
}

template<typename Real_t>
ReactionDiffusion<Real_t>::~ReactionDiffusion()
{
    delete []xc;
    delete []efield;
    delete []netchg;
    delete []A_weight;
    delete []D_weight;
    delete []coeff_active;
    delete []coeff_prod;
    delete []coeff_total;
    delete []coeff_inact;
    if (prec_cache != nullptr)
        delete prec_cache;
    if (jac_cache != nullptr)
        delete jac_cache;
}

template<typename Real_t>
void
ReactionDiffusion<Real_t>::zero_counters(){
    nfev = 0;
    njev = 0;
    nprec_setup = 0;
    nprec_solve = 0;
    njacvec_dot = 0;
    nprec_solve_ilu = 0;
    nprec_solve_lu = 0;
}

template<typename Real_t>
int
ReactionDiffusion<Real_t>::get_ny() const
{
    return n*N;
}

template<typename Real_t>
int
ReactionDiffusion<Real_t>::get_mlower() const
{
    if (N > 1)
        return n*n_jac_diags;
    else
        return -1;
}

template<typename Real_t>
int
ReactionDiffusion<Real_t>::get_mupper() const
{
    return this->get_mlower();
}


template<typename Real_t>
uint
ReactionDiffusion<Real_t>::stencil_bi_lbound_(uint bi) const
{
    const uint le = lrefl ? 0 : nsidep;
    const uint re = rrefl ? 0 : nsidep;
    return max(le, min(N + 2*nsidep - re - nstencil, bi));
}

template<typename Real_t>
uint
ReactionDiffusion<Real_t>::xc_bi_map_(uint xci) const
{
    if (xci < nsidep)
        return nsidep - xci - 1;
    else if (xci >= N+nsidep)
        return nsidep + 2*N - xci - 1;
    else
        return xci - nsidep;
}


#define D_WEIGHT(bi, li) D_weight[nstencil*(bi) + li]
#define A_WEIGHT(bi, li) A_weight[nstencil*(bi) + li]
#define FDWEIGHT(order, local_index) c[nstencil*(order) + local_index]
template<typename Real_t>
void
ReactionDiffusion<Real_t>::apply_fd_(uint bi){
    Real_t * const c = new Real_t[3*nstencil];
    Real_t * const lxc = new Real_t[nstencil]; // local shifted x-centers
    uint around = bi + nsidep;
    uint start  = bi;
    if (!lrefl) // shifted finite diff
        start = max(nsidep, start);
    if (!rrefl) // shifted finite diff
        start = min(N - nstencil + nsidep, start);
    for (uint li=0; li<nstencil; ++li) // li: local index
        lxc[li] = xc[start + li] - xc[around];
    finitediff::populate_weights<Real_t>(0, lxc, nstencil-1, 2, c);
    delete []lxc;

    const Real_t logbdenom = use_log2 ? M_LOG2E : 1; // M_LOG2E == log(e)/log(2) == 1/log(2)

    for (uint li=0; li<nstencil; ++li){ // li: local index
        D_WEIGHT(bi, li) = FDWEIGHT(2, li);
        A_WEIGHT(bi, li) = FDWEIGHT(1, li);
        if (logx){
            D_WEIGHT(bi, li) *= logbdenom*logbdenom;
            A_WEIGHT(bi, li) *= logbdenom;
            switch(geom){
            case Geom::FLAT:
                D_WEIGHT(bi, li) -= FDWEIGHT(1, li)*logbdenom;
                break;
            case Geom::CYLINDRICAL:
                A_WEIGHT(bi, li) += FDWEIGHT(0, li);
                break;
            case Geom::SPHERICAL:
                D_WEIGHT(bi, li) += FDWEIGHT(1, li)*logbdenom;
                A_WEIGHT(bi, li) += 2*FDWEIGHT(0, li);
                break;
            }
            D_WEIGHT(bi, li) *= expb(-2*xc[around]);
            A_WEIGHT(bi, li) *= expb(-xc[around]);
        } else {
            switch(geom){
            case Geom::CYLINDRICAL: // Laplace operator in cyl coords.
                D_WEIGHT(bi, li) += FDWEIGHT(1, li)*1/xc[around];
                A_WEIGHT(bi, li) += FDWEIGHT(0, li)*1/xc[around];
                break;
            case Geom::SPHERICAL: // Laplace operator in sph coords.
                D_WEIGHT(bi, li) += FDWEIGHT(1, li)*2/xc[around];
                A_WEIGHT(bi, li) += FDWEIGHT(0, li)*2/xc[around];
                break;
            default:
                break;
            }
        }
    }
    delete []c;
}
#undef FDWEIGHT
// D_WEIGHT still defined

template<typename Real_t>
Real_t
ReactionDiffusion<Real_t>::get_mod_k(int bi, int ri) const{
    Real_t tmp = k[ri];
    // Modulation
    int enumer = -1;
    for (auto mi : this->modulated_rxns){
        enumer++;
        if (mi == (int)ri)
            tmp *= this->modulation[enumer][bi];
    }
    return tmp;
}

template<typename Real_t>
void
ReactionDiffusion<Real_t>::fill_local_r_(int bi, const Real_t * const __restrict__ C,
                                 Real_t * const __restrict__ local_r) const
{
    // intent(out) :: local_r
    for (uint rxni=0; rxni<nr; ++rxni){
        // reaction rxni
        Real_t tmp = 1;

        // Kinetically active reactants (law of massaction)
        for (uint rnti=0; rnti < stoich_active[rxni].size(); ++rnti){
            // reactant index rnti
            int si = stoich_active[rxni][rnti];
            tmp *= C[bi*n+si];
        }
        // Rate constant
        local_r[rxni] = get_mod_k(bi, rxni)*tmp;
    }
}
// D_WEIGHT still defined

// The indices of x, fluxes and bins
// <indices.png>


#define Y(bi, si) y[(bi)*n+(si)]
template<typename Real_t>
const Real_t *
ReactionDiffusion<Real_t>::alloc_and_populate_linC(const Real_t * const __restrict__ y,
                                                   bool apply_exp, bool recip) const
{
    int nlinC = n*N;
    Real_t * const linC = (Real_t * const)malloc(nlinC*sizeof(Real_t));
    // Possible optimization: tune 42...
    ${"#pragma omp parallel for schedule(static) if (N*n > 65536)" if WITH_OPENMP else ""}
    for (uint bi=0; bi<N; ++bi){
        for (uint si=0; si<n; ++si){
            if (recip)
                linC[bi*n + si] = (apply_exp) ? expb(-Y(bi, si)) : 1.0/Y(bi, si);
            else
                linC[bi*n + si] = (apply_exp) ? expb(Y(bi, si)) : Y(bi, si);
        }
    }
    return linC;
}
// Y, D_WEIGHT(bi, li) still defined
#define LINC(bi, si) linC[(bi)*n+(si)]
#define RLINC(bi, si) rlinC[(bi)*n+(si)]

#define DYDT(bi, si) dydt[(bi)*(n)+(si)]
template<typename Real_t>
AnyODE::Status
ReactionDiffusion<Real_t>::rhs(Real_t t, const Real_t * const y, Real_t * const __restrict__ dydt)
{
    // note condifiontal call to free at end of this function
    const Real_t * const linC = (logy) ? alloc_and_populate_linC(y, true) : y;
    const Real_t * const rlinC = (logy) ? alloc_and_populate_linC(y, true, true) : nullptr;
    std::vector<Real_t> velocity;
    if (auto_efield){
        calc_efield(linC);

    }
    const Real_t expb_t = (logt) ? expb(t) : 0.0;
    ${"Real_t * const local_r = new Real_t[nr];" if not WITH_OPENMP else ""}
    ${"#pragma omp parallel for schedule(static) if (N*n > 65536)" if WITH_OPENMP else ""}
    for (uint bi=0; bi<N; ++bi){
        // compartment bi
        ${"Real_t * const local_r = new Real_t[nr];" if WITH_OPENMP else ""}

        for (uint si=0; si<n; ++si)
            DYDT(bi, si) = 0.0; // zero out

        // Contributions from reactions
        // ----------------------------
        fill_local_r_(bi, linC, local_r);
        for (uint rxni=0; rxni<nr; ++rxni){
            // reaction index rxni
            for (uint si=0; si<n; ++si){
                // species index si
                const int overall = coeff_total[rxni*n + si];
                if (overall != 0)
                    DYDT(bi, si) += overall*local_r[rxni];
            }
        }
        // Contribution from particle/electromagnetic fields
        for (uint fi=0; fi<this->fields.size(); ++fi){
            if (fields[fi][bi] == 0)
                continue; // exit early
            const Real_t gfact = (g_value_parents[fi] == -1) ? \
                1.0 : LINC(bi, g_value_parents[fi]);
            for (uint si=0; si<n; ++si)
                if (g_values[fi][si] != 0)
                    DYDT(bi, si) += fields[fi][bi]*g_values[fi][si]*gfact;
        }

        if (N>1){
            // Contributions from diffusion and advection
            // ------------------------------------------
            int starti;
            if ((bi < nsidep) && (!lrefl)){
                starti = 0;
            } else if ((bi >= N-nsidep) && (!rrefl)){
                starti = N - nstencil;
            } else{
                starti = bi - nsidep;
            }
            for (uint si=0; si<n; ++si){ // species index si
                if ((D[si] == 0.0) && (mobility[si] == 0.0)) continue;
                Real_t unscaled_diffusion = 0;
                Real_t unscaled_advection = 0;
                for (uint xi=0; xi<nstencil; ++xi){
                    int biw = starti + xi;
                    // reflective logic:
                    if (starti < 0){
                        biw = (biw < 0) ? (-1 - biw) : biw; // lrefl==true
                    } else if (starti >= (int)N - (int)nstencil + 1){
                        biw = (biw >= (int)N) ? (2*N - biw - 1) : biw; // rrefl==true
                    }
                    unscaled_diffusion += D_WEIGHT(bi, xi) * LINC(biw, si);
                    unscaled_advection += A_WEIGHT(bi, xi) * \
                        (LINC(biw, si)*efield[bi] + LINC(bi, si)*efield[biw]);
                }
                DYDT(bi, si) += unscaled_diffusion*D[si];
                DYDT(bi, si) += unscaled_advection*-mobility[si];
            }
        }
        for (uint si=0; si<n; ++si){
            if (logy){
                DYDT(bi, si) *= RLINC(bi, si);
                if (!logt and use_log2)
                    DYDT(bi, si) /= M_LN2;
            }
            if (logt){
                DYDT(bi, si) *= expb_t;
                if (!logy and use_log2)
                    DYDT(bi, si) *= M_LN2;
            }
        }
        ${"delete []local_r;" if WITH_OPENMP else ""}
    }
    ${"delete []local_r;" if not WITH_OPENMP else ""}
    if (logy){
        free((void*)linC);
        free((void*)rlinC);
    }
    nfev++;
    return AnyODE::Status::success;
}
#undef DYDT
// D_WEIGHT(bi, li), Y(bi, si) and LINC(bi, si) still defined.


#define FOUT(bi, si) fout[(bi)*n+si]
#define SUP(di, bi, li) jac.sup(di, bi, li)
%for token in ['dense_jac_rmaj', 'dense_jac_cmaj', 'banded_jac_cmaj', 'compressed_jac_cmaj']:
template<typename Real_t>
AnyODE::Status
ReactionDiffusion<Real_t>::${token}(Real_t t,
                                    const Real_t * const __restrict__ y,
                                    const Real_t * const __restrict__ fy,
                                    Real_t * const __restrict__ ja, long int ldj
                                    ${', double * const __restrict__ /* dfdt */' if token.startswith('dense') else ''})
{
    // Note: blocks are zeroed out, diagnoals only incremented
    // `t`: time (log(t) if logt=1)
    // `y`: concentrations (log(conc) if logy=True)
    // `ja`: jacobian (allocated 1D array to hold dense or banded)
    // `ldj`: leading dimension of ja (useful for padding, ignored by compressed_*)
    %if token.startswith('compressed'):
    ignore(ldj);
    block_diag_ilu::ColMajBlockDiagView<Real_t> jac {ja,
            ja + N*n*n,
            ja + N*n*n + n*(N*n_jac_diags - (n_jac_diags*n_jac_diags + n_jac_diags)/2),
            N, n, n_jac_diags};
    %elif token.startswith('banded_jac_cmaj'):
    block_diag_ilu::ColMajBandedView<Real_t> jac {ja, N, n, n_jac_diags, static_cast<uint>(ldj), 0};
    %elif token.startswith('dense_jac_cmaj'):
    block_diag_ilu::DenseView<Real_t> jac {ja, N, n, n_jac_diags, static_cast<uint>(ldj)};
    %elif token.startswith('dense_jac_rmaj'):
    block_diag_ilu::DenseView<Real_t, false> jac {ja, N, n, n_jac_diags, static_cast<uint>(ldj)};
    %else:
#error "Unhandled token."
    %endif
    const Real_t exp_t = (logt) ? expb(t) : 0.0;
    const Real_t logbfactor = use_log2 ? M_LN2 : 1;

    Real_t * fout = nullptr;
    if (logy){ // fy useful..
        if (fy){
            fout = const_cast<Real_t *>(fy);
        } else {
            fout = new Real_t[n*N];
            rhs(t, y, fout);
        }
    }

    // note conditional call to free at end of this function
    const Real_t * const linC = (logy) ? alloc_and_populate_linC(y, true, false) : y;
    const Real_t * const rlinC = (logy) ? alloc_and_populate_linC(y, true, true) :
        alloc_and_populate_linC(y, false, true);
    if (auto_efield)
        calc_efield(linC);

    ${'#pragma omp parallel for schedule(static) if (N*n*n > 65536)' if WITH_OPENMP else ''}
    for (uint bi=0; bi<N; ++bi){
        // Conc. in `bi:th` compartment
        // Contributions from reactions and fields
        // ---------------------------------------
        for (uint si=0; si<n; ++si){
            // species si
            for (uint dsi=0; dsi<n; ++dsi){
                // derivative wrt species dsi
                jac.block(bi, si, dsi) = 0.0;
                for (uint rxni=0; rxni<nr; ++rxni){
                    // reaction rxni
                    const int Akj = coeff_active[rxni*n + dsi];
                    const int Ski = coeff_total[rxni*n + si];
                    if (Akj == 0 || Ski == 0)
                        continue;
                    Real_t qkj = get_mod_k(bi, rxni)*Akj*pow(LINC(bi, dsi), Akj-1);
                    for (uint rnti=0; rnti < stoich_active[rxni].size(); ++rnti){
                        const uint rnti_si = stoich_active[rxni][rnti];
                        if (rnti_si == dsi)
                            continue;
                        qkj *= LINC(bi, rnti_si);
                    }
                    jac.block(bi, si, dsi) += Ski*qkj;
                    // std::cout << jac.block(bi, si, dsi) << "\n";
                }
                // Contribution from particle/electromagnetic fields
                for (uint fi=0; fi<(this->fields.size()); ++fi){
                    const int Ski = (g_values[fi][si] != 0.0) ? 1 : 0;
                    const int Akj = ((int)dsi == g_value_parents[fi]) ? 1 : 0;
                    const Real_t rk = fields[fi][bi]*g_values[fi][si];
                    if (Ski == 0 || Akj == 0 || rk == 0)
                        continue;
                    jac.block(bi, si, dsi) += Akj*Ski*rk;
                }
            }
        }


        // Contributions from diffusion
        // ----------------------------
        if (N > 1) {
            uint lbound = stencil_bi_lbound_(bi);
            for (uint si=0; si<n; ++si){ // species index si
                if ((D[si] == 0.0) && (mobility[si] == 0.0)) continue; // exit early if possible
                for (uint k=0; k<nstencil; ++k){
                    const uint sbi = xc_bi_map_(lbound+k);
                    jac.block(bi, si, si) += -mobility[si]*efield[sbi]*A_WEIGHT(bi, k);
                    if (sbi == bi) {
                        jac.block(bi, si, si) += D[si]*D_WEIGHT(bi, k);
                        jac.block(bi, si, si) += -mobility[si]*efield[bi]*A_WEIGHT(bi, k);
                    } else {
                        for (uint di=0; di<n_jac_diags; ++di){
                            if ((bi >= di+1) and (sbi == bi-di-1)){
                                jac.sub(di, bi-di-1, si) += D[si]*D_WEIGHT(bi, k);
                                jac.sub(di, bi-di-1, si) += efield[bi]*-mobility[si]*A_WEIGHT(bi, k);
                            }
                            if ((bi < N-di-1) and (sbi == bi+di+1)){
                                jac.sup(di, bi, si) += D[si]*D_WEIGHT(bi, k);
                                jac.sup(di, bi, si) += efield[bi]*-mobility[si]*A_WEIGHT(bi, k);
                            }
                        }
                    }
                }
            }
        }

        // Logartihmic transformations
        // ---------------------------
        if (logy || logt){
            for (uint si=0; si<n; ++si){
                for (uint dsi=0; dsi<n; ++dsi){
                    if (logy){
                        jac.block(bi, si, dsi) *= LINC(bi, dsi)*RLINC(bi, si);
                    }
                    if (logt)
                        jac.block(bi, si, dsi) *= exp_t*logbfactor;
                    if (logy && dsi == si)
                        jac.block(bi, si, si) -= FOUT(bi, si)*logbfactor;
                }
                for (uint di=0; di<n_jac_diags; ++di){
                    if (bi > di){
                        if (logy)
                            jac.sub(di, bi-di-1, si) *= LINC(bi-di-1, si)*RLINC(bi, si);
                        if (logt)
                            jac.sub(di, bi-di-1, si) *= exp_t*logbfactor;
                    }
                    if (bi < N-di-1){
                        if (logy)
                            jac.sup(di, bi, si) *= LINC(bi+di+1, si)*RLINC(bi, si);
                        if (logt)
                            jac.sup(di, bi, si) *= exp_t*logbfactor;
                    }
                }
            }
        }
    }
    if (logy && !fy)
        delete []fout;
    free((void*)rlinC);
    if (logy)
        free((void*)linC);
    njev++;
#if defined(WITH_DATA_DUMPING)
    std::ostringstream fname;
    fname << "jac_" << std::setfill('0') << std::setw(5) << njev << ".dat";
    save_array(ja, ldj*n*N, fname.str());
#endif
    return AnyODE::Status::success;
}
%endfor
#undef FOUT


template<typename Real_t>
AnyODE::Status
ReactionDiffusion<Real_t>::jac_times_vec(const Real_t * const __restrict__ vec,
                                         Real_t * const __restrict__ out,
                                         Real_t t,
                                         const Real_t * const __restrict__ y,
                                         const Real_t * const __restrict__ fy
                                         )
{
    // See 4.6.7 on page 67 (77) in cvs_guide.pdf (Sundials 2.5)
    ignore(t);
    if (jac_cache == nullptr){
        jac_cache = new block_diag_ilu::ColMajBlockDiagMat<Real_t>(N, n, nsidep);
        jac_cache->view.zero_out_diags(); // compressed_jac_cmaj only increments diagonals
        const int dummy = 0;
        compressed_jac_cmaj(t, y, fy, jac_cache->get_block_data_raw_ptr(), dummy);
    }
    jac_cache->view.dot_vec(vec, out);
    njacvec_dot++;
    return AnyODE::Status::success;
}

template<typename Real_t>
AnyODE::Status
ReactionDiffusion<Real_t>::prec_setup(Real_t t,
                                      const Real_t * const __restrict__ y,
                                      const Real_t * const __restrict__ fy,
                                      bool jok, bool& jac_recomputed, Real_t gamma
                                      )
{
    auto status = AnyODE::Status::success;
    ignore(gamma);
    // See 4.6.9 on page 68 (78) in cvs_guide.pdf (Sundials 2.5)
    if (jac_cache == nullptr)
        jac_cache = new block_diag_ilu::ColMajBlockDiagMat<Real_t>(N, n, nsidep);
    if (!jok){
        const int dummy = 0;
        jac_cache->view.zero_out_diags();
        status = compressed_jac_cmaj(t, y, fy, jac_cache->get_block_data_raw_ptr(), dummy);
        update_prec_cache = true;
        jac_recomputed = true;
    } else jac_recomputed = false;
    nprec_setup++;
    return status;
}
#undef LINC
#undef Y
#undef D_WEIGHT
#undef A_WEIGHT

template<typename Real_t>
AnyODE::Status
ReactionDiffusion<Real_t>::prec_solve_left(const Real_t t,
                                           const Real_t * const __restrict__ y,
                                           const Real_t * const __restrict__ fy,
                                           const Real_t * const __restrict__ r,
                                           Real_t * const __restrict__ z,
                                           Real_t gamma,
                                           Real_t delta,
                                           const Real_t * const __restrict__ ewt
                                           )
{
    // See 4.6.9 on page 75 in cvs_guide.pdf (Sundials 2.6.2)
    // Solves P*z = r, where P ~= I - gamma*J
    ignore(delta);
    if (ewt)
        throw std::runtime_error("Not implemented.");
    nprec_solve++;

    ignore(t); ignore(fy); ignore(y);
    bool recompute = false;
    if (prec_cache == nullptr){
        prec_cache = new block_diag_ilu::ColMajBlockDiagMat<Real_t>(N, n, nsidep);
        recompute = true;
    } else {
        if (update_prec_cache or (old_gamma != gamma))
            recompute = true;
    }
    if (recompute){
        old_gamma = gamma;
        prec_cache->view.set_to_1_minus_gamma_times_view(gamma, jac_cache->view);
#if defined(WITH_DATA_DUMPING)
        {
            std::ostringstream fname;
            fname << "prec_M_" << std::setfill('0') << std::setw(5) << nprec_solve << ".dat";
            const auto data_len = prec_cache->view.block_data_len + 2*prec_cache->view.diag_data_len;
            save_array(prec_cache->get_block_data_raw_ptr(), data_len, fname.str());
        }
#endif
    }

#if defined(WITH_DATA_DUMPING)
    {
        std::ostringstream fname;
        fname << "prec_r_" << std::setfill('0') << std::setw(5) << nprec_solve << ".dat";
        save_array(r, n*N, fname.str());
    }
    {
        std::ostringstream fname;
        fname << "prec_g_" << std::setfill('0') << std::setw(5) << nprec_solve << ".dat";
        save_array(&gamma, 1, fname.str());
    }

#endif

    int info;
    if (prec_cache->view.average_diag_weight(0) > ilu_limit) {
        block_diag_ilu::ILU<Real_t> ilu {prec_cache->view};
        nprec_solve_ilu++;
        info = ilu.solve(r, z);
    } else {
        block_diag_ilu::LU<Real_t> lu {prec_cache->view};
        nprec_solve_lu++;
        info = lu.solve(r, z);
    }
    if (info == 0)
        return AnyODE::Status::success;
    return AnyODE::Status::recoverable_error;
}

template<typename Real_t>
void
ReactionDiffusion<Real_t>::per_rxn_contrib_to_fi(Real_t t, const Real_t * const __restrict__ y,
                                              uint si, Real_t * const __restrict__ out) const
{
    ignore(t);
    Real_t * const local_r = new Real_t[nr];
    fill_local_r_(0, y, local_r);
    for (uint ri=0; ri<nr; ++ri){
	out[ri] = coeff_total[ri*n+si]*local_r[ri];
    }
    delete []local_r;
}

template<typename Real_t>
int
ReactionDiffusion<Real_t>::get_geom_as_int() const
{
    switch(geom){
    case Geom::FLAT :        return 0;
    case Geom::CYLINDRICAL : return 1;
    case Geom::SPHERICAL :   return 2;
    case Geom::PERIODIC :   return 3;
    default:                 return -1;
    }
}

template<typename Real_t>
void
ReactionDiffusion<Real_t>::calc_efield(const Real_t * const linC)
{
    // Prototype for self-generated electric field
    const Real_t F = this->faraday_const; // Faraday's constant
    const Real_t eps = eps_rel*vacuum_permittivity;
    Real_t nx, cx = logx ? expb(x[0]) : x[0];
    for (uint bi=0; bi<N; ++bi){
        netchg[bi] = 0.0;
        for (uint si=0; si<n; ++si)
            netchg[bi] += z_chg[si]*linC[bi*n+si];
    }
    Real_t Q = surf_chg.first;
    for (uint bi=0; bi<N; ++bi){
        const Real_t r = logx ? expb(xc[nsidep+bi]) : xc[nsidep+bi];
        nx = logx ? expb(x[bi+1]) : x[bi+1];
        switch(geom){
        case Geom::FLAT:
            efield[bi] = F*Q/eps;
            Q += netchg[bi]*(nx - cx);
            break;
        case Geom::CYLINDRICAL:
            efield[bi] = F*Q/(2*M_PI*eps*r); // Gauss's law
            Q += netchg[bi]*M_PI*(nx*nx - cx*cx);
            break;
        case Geom::SPHERICAL:
            efield[bi] = F*Q/(4*M_PI*eps*r*r); // Gauss's law
            Q += netchg[bi]*4*M_PI/3*(nx*nx*nx - cx*cx*cx);
            break;
        case Geom::PERIODIC:
            efield[bi] = 0;
        }
        cx = nx;
    }
    if (geom == Geom::FLAT){
        Q = surf_chg.second;
        for (uint bi=N; bi>0; --bi){ // unsigned int..
            nx = logx ? expb(x[bi-1]) : x[bi-1];
            efield[bi-1] -= F*Q/eps;
            Q += netchg[bi-1]*(cx - nx);
            cx = nx;
        }
    }
}
} // namespace chemreac

template class chemreac::ReactionDiffusion<double>; // instantiate template
