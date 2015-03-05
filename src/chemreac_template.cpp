## -*- coding: utf-8 -*-
// ${_warning_in_the_generated_file_not_to_edit}
<%doc>
// This is a templated source file.
// Render template using Mako (Python templating engine)
</%doc>
#include <algorithm> // std::count
//#include <vector>    // std::vector
#include <algorithm> // std::max, std::min
#include <cstdlib> // free,  C++11 aligned_alloc
#include "chemreac.hpp"
#include "c_fornberg.h" // fintie differences (remember to link fortran object fornberg.o)

#if defined(WITH_DATA_DUMPING)
#include <cstdio>
#include <iostream>
#include <sstream>
#include <iomanip>
#define PRINT_ARR(ARR, LARR) for(int i_=0; i_<LARR; ++i_) {std::cout << ARR[i_] << " ";}; std::cout << std::endl;
#include "chemreac_util.h"
#endif

%if USE_OPENMP:
#ifndef _OPENMP
  #error "Have you forgotten -fopenmp flag?"
#endif
#include <omp.h>
%else:
#ifdef _OPENMP
  #error "You should render OpenMP enabled sources"
#endif
#define omp_get_thread_num() 0
%endif


namespace chemreac {

using std::vector;
using std::count;
using std::min;
using std::max;

template<class T> void ignore( const T& ) { } // ignore compiler warnings about unused parameter

// 1D discretized reaction diffusion
ReactionDiffusion::ReactionDiffusion(
    uint n,
    const vector<vector<uint> > stoich_reac,
    const vector<vector<uint> > stoich_prod,
    vector<double> k,
    uint N,
    vector<double> D,
    const vector<int> z_chg,
    vector<double> mobility,
    const vector<double> x, // separation
    vector<vector<uint> > stoich_actv_, // vectors of size 0 in stoich_actv_ => "copy from stoich_reac"
    vector<vector<double> > bin_k_factor, // per bin modulation of first k's
    vector<uint> bin_k_factor_span, // modulation over reactions
    int geom_,
    bool logy,
    bool logt,
    bool logx,
    uint nstencil,
    bool lrefl,
    bool rrefl,
    bool auto_efield,
    pair<double, double> surf_chg,
    double eps_rel,
    double faraday_const,
    double vacuum_permittivity):
    n(n), N(N), nstencil(nstencil), nsidep((nstencil-1)/2), nr(stoich_reac.size()),
    logy(logy), logt(logt), logx(logx), stoich_reac(stoich_reac), stoich_prod(stoich_prod),
    k(k),  D(D), z_chg(z_chg), mobility(mobility), x(x), bin_k_factor(bin_k_factor),
    bin_k_factor_span(bin_k_factor_span), lrefl(lrefl), rrefl(rrefl), auto_efield(auto_efield),
    surf_chg(surf_chg), eps_rel(eps_rel), faraday_const(faraday_const),
    vacuum_permittivity(vacuum_permittivity), efield(new double[N]), netchg(new double[N])
{
    if (N == 0) throw std::logic_error("Zero bins sounds boring.");
    if (N == 2) throw std::logic_error("2nd order PDE requires at least 3 stencil points.");
    if (nstencil % 2 == 0) throw std::logic_error("Only odd number of stencil points supported");
    if ((N == 1) && (nstencil != 1)) throw std::logic_error("You must set nstencil=1 for N=1");
    if ((N > 1) && (nstencil <= 1)) throw std::logic_error("You must set nstencil>1 for N>1");
    if (stoich_reac.size() != stoich_prod.size())
        throw std::length_error(
            "stoich_reac and stoich_prod of different sizes.");
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
    default:
        throw std::logic_error("Unknown geom.");
    }

    // Finite difference scheme
    D_weight = new double[nstencil*N];
    A_weight = new double[nstencil*N];
    for (uint i=0; i<N; ++i) efield[i] = 0.0;
    xc = new double[nsidep + N + nsidep]; // xc padded with virtual bins
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
        _apply_fd(bi);

    // Stoichiometry
    for (uint ri=0; ri<nr; ++ri){
        for (auto si=stoich_reac[ri].begin(); si != stoich_reac[ri].end(); ++si)
            if (*si > n-1)
                throw std::logic_error("At least one species index in stoich_reac > (n-1)");
        for (auto si=stoich_prod[ri].begin(); si != stoich_prod[ri].end(); ++si)
            if (*si > n-1)
                throw std::logic_error("At least one species index in stoich_prod > (n-1)");
        for (auto si=stoich_actv_[ri].begin(); si != stoich_actv_[ri].end(); ++si)
            if (*si > n-1)
                throw std::logic_error("At least one species index in stoich_actv > (n-1)");
    }

    coeff_reac = new int[nr*n];
    coeff_prod = new int[nr*n];
    coeff_totl = new int[nr*n];
    coeff_actv = new int[nr*n];

    stoich_actv.reserve(nr);
    for (uint rxni=0; rxni<nr; ++rxni){ // reaction index
        if (stoich_actv_[rxni].size() == 0)
            stoich_actv.push_back(stoich_reac[rxni]); // massaction
        else
            stoich_actv.push_back(stoich_actv_[rxni]);

        for (uint si=0; si<n; ++si){ // species index
            coeff_reac[rxni*n+si] = count(stoich_reac[rxni].begin(),
                                        stoich_reac[rxni].end(), si);
            coeff_actv[rxni*n+si] = count(stoich_actv[rxni].begin(),
                                        stoich_actv[rxni].end(), si);
            coeff_prod[rxni*n+si] = count(stoich_prod[rxni].begin(),
                                        stoich_prod[rxni].end(), si);
            coeff_totl[rxni*n+si] = coeff_prod[rxni*n+si] -\
                coeff_reac[rxni*n+si];
        }
    }

    // Handle bin_k_factors:
    for (uint i=0; i<bin_k_factor_span.size(); ++i)
        for (uint j=0; j<bin_k_factor_span[i]; ++j)
            i_bin_k.push_back(i);
    n_factor_affected_k = i_bin_k.size();
}

ReactionDiffusion::~ReactionDiffusion()
{
    delete []xc;
    delete []efield;
    delete []netchg;
    delete []A_weight;
    delete []D_weight;
    delete []coeff_reac;
    delete []coeff_prod;
    delete []coeff_totl;
    delete []coeff_actv;
    if (prec_cache != nullptr)
        delete prec_cache;
    if (jac_cache != nullptr)
        delete jac_cache;
}

uint ReactionDiffusion::_stencil_bi_lbound(uint bi) const
{
    const uint le = lrefl ? 0 : nsidep;
    const uint re = rrefl ? 0 : nsidep;
    return max(le, min(N + 2*nsidep - re - nstencil, bi));
}

uint ReactionDiffusion::_xc_bi_map(uint xci) const
{
    if (xci < nsidep)
        return nsidep - xci - 1;
    else if (xci >= N+nsidep)
        return 2*N - xci;
    else
        return xci - nsidep;
}


#define D_WEIGHT(bi, li) D_weight[nstencil*(bi) + li]
#define A_WEIGHT(bi, li) A_weight[nstencil*(bi) + li]
#define FDWEIGHT(order, local_index) c[nstencil*(order) + local_index]
void ReactionDiffusion::_apply_fd(uint bi){
    double * const c = new double[3*nstencil];
    double * const lxc = new double[nstencil]; // local shifted x-centers
    uint around = bi + nsidep;
    uint start  = bi;
    if (!lrefl) // shifted finite diff
        start = max(nsidep, start);
    if (!rrefl) // shifted finite diff
        start = min(N - nstencil + nsidep, start);
    for (uint li=0; li<nstencil; ++li) // li: local index
        lxc[li] = xc[start + li] - xc[around];
    fornberg_populate_weights(0, lxc, nstencil-1, 2, c);
    delete []lxc;

    for (uint li=0; li<nstencil; ++li){ // li: local index
        D_WEIGHT(bi, li) = FDWEIGHT(2, li);
        A_WEIGHT(bi, li) = FDWEIGHT(1, li);
        if (logx){
            switch(geom){
            case Geom::FLAT:
                D_WEIGHT(bi, li) -= FDWEIGHT(1, li);
                break;
            case Geom::CYLINDRICAL:
                A_WEIGHT(bi, li) += FDWEIGHT(0, li);
                break;
            case Geom::SPHERICAL:
                D_WEIGHT(bi, li) += FDWEIGHT(1, li);
                A_WEIGHT(bi, li) += 2*FDWEIGHT(0, li);
                break;
            }
            D_WEIGHT(bi, li) *= exp(-2*xc[around]);
            A_WEIGHT(bi, li) *= exp(-xc[around]);
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

#define FACTOR(ri, bi) ( ((ri) < n_factor_affected_k) ? \
            bin_k_factor[bi][i_bin_k[ri]] : 1 )
void
ReactionDiffusion::_fill_local_r(int bi, const double * const __restrict__ y,
                 double * const __restrict__ local_r) const
{
    // intent(out) :: local_r
    for (uint rxni=0; rxni<nr; ++rxni){
        // reaction rxni
        if (logy)
            local_r[rxni] = 0;
        else
            local_r[rxni] = 1;

        for (uint rnti=0; rnti<stoich_actv[rxni].size(); ++rnti){
            // reactant index rnti
            int si = stoich_actv[rxni][rnti];
            if (logy)
                local_r[rxni] += y[bi*n+si];
            else
                local_r[rxni] *= y[bi*n+si];
        }
        if (logy)
            local_r[rxni] = exp(local_r[rxni]);

        local_r[rxni] *= FACTOR(rxni, bi)*k[rxni];
    }
}
#undef FACTOR
// D_WEIGHT still defined

// The indices of x, fluxes and bins
// <indices.png>


#define Y(bi, si) y[(bi)*n+(si)]
#define LINC(bi, si) linC[(bi)*n+(si)]
const double *
ReactionDiffusion::_alloc_and_populate_linC(const double * const __restrict__ y) const
{
    int nlinC = n*N;
    double * const linC = (double * const)malloc(nlinC*sizeof(double));
    // TODO: Tune 42...
    ${"#pragma omp parallel for if (N > 42)" if USE_OPENMP else ""}
    for (uint bi=0; bi<N; ++bi)
        for (uint si=0; si<n; ++si)
            LINC(bi, si) = exp(Y(bi, si));
    return linC;
}
// Y, LINC, D_WEIGHT(bi, li) still defined

#define DYDT(bi, si) dydt[(bi)*(n)+(si)]
void
ReactionDiffusion::f(double t, const double * const y, double * const __restrict__ dydt)
{
    // note condifiontal call to free at end of this function
    const double * const linC = (logy) ? _alloc_and_populate_linC(y) : y;
    if (auto_efield){
        calc_efield(linC);
    }

    ${"double * const local_r = new double[nr];" if not USE_OPENMP else ""}
    ${"#pragma omp parallel for if (N > 2)" if USE_OPENMP else ""}
    for (uint bi=0; bi<N; ++bi){
        // compartment bi
        ${"double * const local_r = new double[nr];" if USE_OPENMP else ""}

        for (uint si=0; si<n; ++si)
            DYDT(bi, si) = 0.0; // zero out

        // Contributions from reactions
        // ----------------------------
        _fill_local_r(bi, y, local_r);
        for (uint rxni=0; rxni<nr; ++rxni){
            // reaction index rxni
            for (uint si=0; si<n; ++si){
                // species index si
                int overall = coeff_totl[rxni*n + si];
                if (overall != 0)
                    DYDT(bi, si) += overall*local_r[rxni];
            }
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
                double unscaled_diffusion = 0;
                double unscaled_advection = 0;
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
        if (logy){
            if (logt)
                for (uint si=0; si<n; ++si){
                    DYDT(bi, si) *= exp(t-Y(bi, si));
                }
            else
                for (uint si=0; si<n; ++si){
                    //DYDT(bi, si) *= exp(-Y(bi, si));
                    DYDT(bi, si) /= LINC(bi, si);
                }
        } else {
            if (logt)
                for (uint si=0; si<n; ++si)
                    DYDT(bi, si) *= exp(t);
        }
        ${"delete []local_r;" if USE_OPENMP else ""}

    }
    ${"delete []local_r;" if not USE_OPENMP else ""}

    if (logy)
        free((void*)linC);
    neval_f++;
}
#undef DYDT
// D_WEIGHT(bi, li), Y(bi, si) and LINC(bi, si) still defined.

#define FOUT(bi, si) fout[(bi)*n+si]
%for token, imaj, imin in [\
    ('dense_jac_rmaj',         '(bri)*n+ri', '(bci)*n + ci'),\
    ('dense_jac_cmaj',         '(bci)*n+ci', '(bri)*n + ri'),\
    ('banded_packed_jac_cmaj', '(bci)*n+ci', '(1+bri-(bci))*n+ri-ci'),\
    ('banded_padded_jac_cmaj', '(bci)*n+ci', '(2+bri-(bci))*n+ri-ci'),\
    ('compressed_jac_cmaj', None, None),\
    ]:
%if token.startswith('compressed'):
#define JAC(bi, ignore_, ri, ci) ja[(bi*n + ci)*n + ri]
    //define SUB(di, bri, ci) std::cout<< di << " " << bri << " " << ci << std::endl; ja[N*n*n + n*(N*(di-1) - ((di-1)*(di-1) + (di-1))/2) + (bri-di)*n + ci]
#define SUB(di, bri, ci) ja[N*n*n + n*(N*(di-1) - ((di-1)*(di-1) + (di-1))/2) + (bri-di)*n + ci]
#define SUP(di, bri, ci) ja[N*n*n + n*(N*nsidep - (nsidep*nsidep + nsidep)/2) + n*(N*(di-1) - ((di-1)*(di-1) + (di-1))/2) + (bri)*n + ci]
%else:
#define JAC(bri, bci, ri, ci) ja[(${imaj})*ldj+${imin}]
#define SUB(di, bri, ci) JAC(bri, bri-di, ci, ci)
#define SUP(di, bri, ci) JAC(bri, bri+di, ci, ci)
%endif
void
ReactionDiffusion::${token}(double t,
                            const double * const __restrict__ y,
                            const double * const __restrict__ fy,
                            double * const __restrict__ ja, int ldj)
{
    // Note: does not return a strictly correct Jacobian for nstecil > 3
    // (only 1 pair of bands).
    // `t`: time (log(t) if logt=1)
    // `y`: concentrations (log(conc) if logy=True)
    // `ja`: jacobian (allocated 1D array to hold dense or banded)
    // `ldj`: leading dimension of ja (useful for padding, ignored by compressed_*)
    ${'ignore(ldj);' if token.startswith('compressed') else ''}

    double * fout = nullptr;
    if (logy){ // fy useful..
        if (fy){
            fout = const_cast<double *>(fy);
        } else {
            fout = new double[n*N];
            f(t, y, fout);
        }
    }

    // note condifiontal call to free at end of this function
    const double * const linC = (logy) ? _alloc_and_populate_linC(y) : y;
    if (auto_efield)
        calc_efield(linC);

    ${'double * const local_r = new double[nr];' if not USE_OPENMP else ''}
    ${'#pragma omp parallel for' if USE_OPENMP else ''}
    for (uint bi=0; bi<N; ++bi){
        // Conc. in `bi:th` compartment
        ${'double * const local_r = new double[nr];' if USE_OPENMP else ''}

        // Contributions from reactions
        // ----------------------------
        _fill_local_r(bi, y, local_r);
        for (uint si=0; si<n; ++si){
            // species si
            for (uint dsi=0; dsi<n; ++dsi){
                // derivative wrt species dsi
                // j_i[si, dsi] = Sum_l(n_lj*Derivative(r[l], local_y[dsi]))
                JAC(bi, bi, si, dsi) = 0.0;
                for (uint rxni=0; rxni<nr; ++rxni){
                    // reaction rxni
                    if (coeff_totl[rxni*n + si] == 0)
                        continue; // species si unaffected by reaction
                    if (coeff_actv[rxni*n + dsi] == 0)
                        continue; // rate of reaction unaffected by species dsi
                    double tmp = coeff_totl[rxni*n + si]*\
                    coeff_actv[rxni*n + dsi]*local_r[rxni];
                    if (!logy)
                        tmp /= Y(bi, dsi);
                    JAC(bi, bi, si, dsi) += tmp;
                }
                if (logy)
                    //JAC(bi, bi, si, dsi) *= exp(-Y(bi, si));
                    JAC(bi, bi, si, dsi) /= LINC(bi, si);
            }
        }

        // Contributions from diffusion
        // ----------------------------
        if (N > 1) {
            uint lbound = _stencil_bi_lbound(bi);
            for (uint si=0; si<n; ++si){ // species index si
                if ((D[si] == 0.0) && (mobility[si] == 0.0)) continue; // exit early if possible
                // All versions expect the compressed Jacobian ignore any more sub/super
                // diagonals than the innermost.
                for (uint k=0; k<nstencil; ++k){
                    const uint sbi = _xc_bi_map(lbound+k);
                    JAC(bi, bi, si, si) += -mobility[si]*efield[sbi]*A_WEIGHT(bi, k);
                    if (sbi == bi) {
                        JAC(bi, bi, si, si) += D[si]*D_WEIGHT(bi, k);
                        JAC(bi, bi, si, si) += -mobility[si]*efield[bi]*A_WEIGHT(bi, k);
                    } else {
                        if (bi >= 1)
                            if (sbi == bi-1){
                                double Cfactor = (logy ? LINC(bi-1, si)/LINC(bi, si) : 1.0);
                                SUB(1, bi, si) += D[si]*D_WEIGHT(bi, k)*Cfactor;
                                SUB(1, bi, si) += efield[bi]*-mobility[si]*A_WEIGHT(bi, k)*\
                                    Cfactor;
                            }
                        if (bi < N-1)
                            if (sbi == bi+1){
                                double Cfactor = (logy ? LINC(bi+1, si)/LINC(bi, si) : 1.0);
                                SUP(1, bi, si) += D[si]*D_WEIGHT(bi, k)*Cfactor;
                                SUP(1, bi, si) += efield[bi]*-mobility[si]*A_WEIGHT(bi, k)*\
                                    Cfactor;
                            }
                    }
                }
            }
        }

        // Logartihmic time
        // ----------------------------
        if (logt || logy){
            for (uint si=0; si<n; ++si){
                if (logt){
                    for (uint dsi=0; dsi<n; ++dsi)
                        JAC(bi, bi, si, dsi) *= exp(t);
                    if (bi>0)
                        SUB(1, bi, si) *= exp(t);
                    if (bi<N-1)
                        SUP(1, bi, si) *= exp(t);
                }
                if (logy)
                    JAC(bi, bi, si, si) -= FOUT(bi, si);
            }
        }
        ${'delete []local_r;' if USE_OPENMP else ''}
    }
    ${'delete []local_r;' if not USE_OPENMP else ''}
    if (logy && !fy)
        delete []fout;
    if (logy)
        free((void*)linC);
    neval_j++;
#if defined(WITH_DATA_DUMPING)
    std::ostringstream fname;
    fname << "jac_" << std::setfill('0') << std::setw(5) << neval_j << ".dat";
    save_array(ja, ldj*n*N, fname.str());
#endif
}
#undef JAC
#undef SUB
#undef SUP
%endfor

// #define JAC(ri, ci) ja[ri*n + ci]
// void ReactionDiffusion::local_reaction_jac(const uint bi, const double * const y,
//                                            double * const __restrict__ ja) const
// {
//     const double * const linC = (logy) ? _alloc_and_populate_linC(y) : y;
//     std::unique_ptr<double[]> local_r {new double[nr]};
//     _fill_local_r(bi, y, local_r);
//     for (uint si=0; si<n; ++si){
//         // species si
//         for (uint dsi=0; dsi<n; ++dsi){
//             // derivative wrt species dsi
//             // j_i[si, dsi] = Sum_l(n_lj*Derivative(r[l], local_y[dsi]))
//             JAC(si, dsi) = 0.0;
//             for (uint rxni=0; rxni<nr; ++rxni){
//                 // reaction rxni
//                 if (coeff_totl[rxni*n + si] == 0)
//                     continue; // species si unaffected by reaction
//                 if (coeff_actv[rxni*n + dsi] == 0)
//                     continue; // rate of reaction unaffected by species dsi
//                 double tmp = coeff_totl[rxni*n + si]*\
//                 coeff_actv[rxni*n + dsi]*local_r[rxni];
//                 if (!logy)
//                     tmp /= Y(bi, dsi);
//                 JAC(si, dsi) += tmp;
//             }
//             if (logy)
//                 //JAC(bi, bi, si, dsi) *= gamma*exp(-Y(bi, si));
//                 JAC(si, dsi) /= LINC(bi, si);
//             JAC(si, dsi) *= gamma;
//         }
//     }
//     if (logy)
//         free((void*)linC);
// }
// #undef JAC

void ReactionDiffusion::jac_times_vec(const double * const __restrict__ vec,
                                      double * const __restrict__ out,
                                      double t,
                                      const double * const __restrict__ y,
                                      const double * const __restrict__ fy
                                      )
{
    // See 4.6.7 on page 67 (77) in cvs_guide.pdf (Sundials 2.5)
    ignore(t);
    // {
    //     // Do we need a fresh jacobian?
    //     block_diag_ilu::ColMajBlockDiagMat<double> jmat {this->N, this->n, this->nsidep};
    //     jmat.view.zero_out_diags(); // compressed_jac_cmaj only increments diagonals
    //     const int dummy = 0;
    //     compressed_jac_cmaj(t, y, fy, jmat.get_block_data_raw_ptr(), dummy);
    //     jmat.view.dot_vec(vec, out);
    // }
    {
        // or can we do with the cache?
        ignore(y); ignore(fy);
        if (jac_cache == nullptr){
            jac_cache = new block_diag_ilu::ColMajBlockDiagMat<double>(N, n, nsidep);
            jac_cache->view.zero_out_diags(); // compressed_jac_cmaj only increments diagonals
            const int dummy = 0;
            compressed_jac_cmaj(t, y, fy, jac_cache->get_block_data_raw_ptr(), dummy);
        }
        jac_cache->view.dot_vec(vec, out);
    }
    njacvec_dot++;
}

void ReactionDiffusion::prec_setup(double t,
                const double * const __restrict__ y,
                const double * const __restrict__ fy,
                bool jok, bool& jac_recomputed, double gamma)
{
    ignore(gamma);
    // See 4.6.9 on page 68 (78) in cvs_guide.pdf (Sundials 2.5)
    if (!jok){
        if (jac_cache == nullptr)
            jac_cache = new block_diag_ilu::ColMajBlockDiagMat<double>(N, n, nsidep);
        const int dummy = 0;
        jac_cache->view.zero_out_diags();
        compressed_jac_cmaj(t, y, fy, jac_cache->get_block_data_raw_ptr(), dummy);
        jac_recomputed = true;
    } else jac_recomputed = false;
    nprec_setup++;
}
#undef FOUT
#undef LINC
#undef Y
#undef D_WEIGHT
#undef A_WEIGHT

void ReactionDiffusion::prec_solve_left(const double t,
                                        const double * const __restrict__ y,
                                        const double * const __restrict__ fy,
                                        const double * const __restrict__ r,
                                        double * const __restrict__ z,
                                        double gamma)
{
    // See 4.6.8 on page 68 (78) in cvs_guide.pdf
    // Solves P*z = r, where P ~= I - gamma*J
    // see page  in cvs_guide.pdf (Sundials 2.5)
    nprec_solve++;

    ignore(t); ignore(fy); ignore(y);
    bool recompute = false;
    if (prec_cache == nullptr){
        prec_cache = new block_diag_ilu::ColMajBlockDiagMat<double>(N, n, nsidep);
        recompute = true;
    } else {
        if (old_gamma != gamma) // TODO: what about when jac_cahce updated?
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

    // {
    //     block_diag_ilu::ILU_inplace ilu {prec_cache->view};
    //     ilu.solve(r, z);
    // }
    {
        // This section is for debuggging.
        // The ILU preconditioning seem to be working
        // so and so.
        // This performs a full LU decomposition
        // which should essentially be equivalent with
        // the direct linear solver.
        block_diag_ilu::LU lu {prec_cache->view};
        lu.solve(r, z);
    }
}

void ReactionDiffusion::per_rxn_contrib_to_fi(double t, const double * const __restrict__ y,
                                              uint si, double * const __restrict__ out) const
{
    ignore(t);
    double * const local_r = new double[nr];
    _fill_local_r(0, y, local_r);
    for (uint ri=0; ri<nr; ++ri){
	out[ri] = coeff_totl[ri*n+si]*local_r[ri];
    }
    delete []local_r;
}

int ReactionDiffusion::get_geom_as_int() const
{
    switch(geom){
    case Geom::FLAT :        return 0;
    case Geom::CYLINDRICAL : return 1;
    case Geom::SPHERICAL :   return 2;
    default:                 return -1;
    }
}

void ReactionDiffusion::calc_efield(const double * const linC)
{
    // Prototype for self-generated electric field
    const double F = this->faraday_const; // Faraday's constant
    const double pi = 3.14159265358979324;
    const double eps = eps_rel*vacuum_permittivity;
    double nx, cx = logx ? exp(x[0]) : x[0];
    for (uint bi=0; bi<N; ++bi){
        netchg[bi] = 0.0;
        for (uint si=0; si<n; ++si)
            netchg[bi] += z_chg[si]*linC[bi*n+si];
    }
    double Q = surf_chg.first;
    for (uint bi=0; bi<N; ++bi){
        const double r = logx ? exp(xc[nsidep+bi]) : xc[nsidep+bi];
        nx = logx ? exp(x[bi+1]) : x[bi+1];
        switch(geom){
        case Geom::FLAT:
            efield[bi] = F*Q/eps;
            Q += netchg[bi]*(nx - cx);
            break;
        case Geom::CYLINDRICAL:
            efield[bi] = F*Q/(2*pi*eps*r); // Gauss's law
            Q += netchg[bi]*pi*(nx*nx - cx*cx);
            break;
        case Geom::SPHERICAL:
            efield[bi] = F*Q/(4*pi*eps*r*r); // Gauss's law
            Q += netchg[bi]*4*pi/3*(nx*nx*nx - cx*cx*cx);
            break;
        }
        cx = nx;
    }
    if (geom == Geom::FLAT){
        Q = surf_chg.second;
        for (uint bi=N; bi>0; --bi){ // unsigned int..
            nx = logx ? exp(x[bi-1]) : x[bi-1];
            efield[bi-1] -= F*Q/eps;
            Q += netchg[bi-1]*(cx - nx);
            cx = nx;
        }
    }
}

} // namespace chemreac
