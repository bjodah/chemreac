#ifndef CHEMREAC_PVHQOBGMVZECTIJSMOKFUXJXXM
#define CHEMREAC_PVHQOBGMVZECTIJSMOKFUXJXXM

#include <vector>
#include <utility>
#include <stdexcept>
#include <memory> // unique_ptr
#include <unordered_map>
#include "block_diag_ilu.hpp"
#include "anyode/anyode.hpp"


namespace chemreac {

enum class Geom {FLAT, CYLINDRICAL, SPHERICAL, PERIODIC};

using std::vector;
using std::pair;

template<class T> void ignore( const T& ) { } // ignore compiler warnings about unused parameter

template <typename Real_t = double>
class ReactionDiffusion : public AnyODE::OdeSysBase
{
public:
    int * coeff_active;
    int * coeff_prod;
    int * coeff_total;
    int * coeff_inact;
    Real_t * D_weight; // diffusion weights
    Real_t * A_weight; // Advection weights
    uint n_factor_affected_k;
    Geom geom; // Geometry: 0: 1D flat, 1: 1D Cylind, 2: 1D Spherical.
    void * integrator {nullptr};

    void fill_local_r_(int, const Real_t * const __restrict__, Real_t * const __restrict__) const;
    void apply_fd_(uint);
    const Real_t * alloc_and_populate_linC(const Real_t * const __restrict__, bool=false, bool=false) const;
    uint stencil_bi_lbound_(uint bi) const;
    uint xc_bi_map_(uint xci) const;

public:
    const uint n; // number of species
    const uint N; // number of compartments
    const uint nstencil; // number of points used in finite difference stencil
    const uint nsidep; // (nstencil-1)/2
    const uint nr; // number of reactions
    const bool logy; // use logarithmic concenctraction
    const bool logt; // use logarithmic time
    const bool logx; // use logarithmic x (space coordinate)
    const vector<vector<uint> > stoich_active; // Reactants per reaction
    const vector<vector<uint> > stoich_inact; // Active reactants per reaction
    const vector<vector<uint> > stoich_prod; // Products per reaction
    vector<Real_t> k; // Rate coefficients (law of mass action)
    vector<Real_t> D; // Diffusion coefficients
    vector<int> z_chg; // ion charge
    vector<Real_t> mobility; // electrical mobility
    const vector<Real_t> x; // Bin edges (length = N+1)
    const bool lrefl, rrefl;
    const bool auto_efield;
    const pair<Real_t, Real_t> surf_chg;
    const Real_t eps_rel; // relative permittivity
    const Real_t faraday_const;
    const Real_t vacuum_permittivity;
    vector<vector<Real_t>> g_values;
    vector<int> g_value_parents;
    vector<vector<Real_t>> fields;
    vector<int> modulated_rxns;
    vector<vector<Real_t> > modulation;
    const Real_t ilu_limit;
    const uint n_jac_diags;
    const bool use_log2;

    Real_t * const efield; // v_d = mu_el*E
    Real_t * const netchg;

    const int nroots = 0;
private:
    block_diag_ilu::ColMajBlockDiagMat<Real_t> *jac_cache {nullptr};
    block_diag_ilu::ColMajBlockDiagMat<Real_t> *prec_cache {nullptr};
    bool update_prec_cache = false;
    Real_t old_gamma;

public:
    Real_t * xc; // bin centers (length = N+nstencil-1), first bin center: xc[(nstencil-1)/2]

    // counters
    long nfev {0};
    long njev {0};
    long nprec_setup {0};
    long nprec_solve {0};
    long njacvec_dot {0};
    long nprec_solve_ilu {0};
    long nprec_solve_lu {0};

    ReactionDiffusion(uint,
		      const vector<vector<uint> >,
		      const vector<vector<uint> >,
		      vector<Real_t>,
		      uint,
		      vector<Real_t>,
                      const vector<int>,
                      vector<Real_t>,
		      const vector<Real_t>,
		      vector<vector<uint> >,
		      int geom_=0,
		      bool logy=false,
		      bool logt=false,
                      bool logx=false,
                      uint nstencil=3,
                      bool lrefl=false,
                      bool rrefl=false,
                      bool auto_efield=false,
                      pair<Real_t, Real_t> surf_chg={0, 0},
                      Real_t eps_rel=1.0,
                      Real_t faraday_const=9.64853399e4, // C/mol
                      Real_t vacuum_permittivity=8.854187817e-12, // F/m
                      vector<vector<Real_t> > g_values={},
                      vector<int> g_value_parents={},
                      vector<vector<Real_t> > fields={},
                      vector<int> modulated_rxns={},
                      vector<vector<Real_t> > modulation={},
                      Real_t ilu_limit=1000.0,
                      uint n_jac_diags=0,
                      bool use_log2=false
                      );
    ~ReactionDiffusion();

    void zero_counters();

    int get_ny() const override;
    int get_mlower() const override;
    int get_mupper() const override;

    AnyODE::Status rhs(Real_t, const Real_t * const, Real_t * const __restrict__) override;
    // AnyODE::Status roots(Real_t xval, const Real_t * const y, Real_t * const out) override;

    AnyODE::Status dense_jac_rmaj(Real_t, const Real_t * const __restrict__, const Real_t * const __restrict__, Real_t * const __restrict__, long int, double * const __restrict__ dfdt=nullptr) override;
    AnyODE::Status dense_jac_cmaj(Real_t, const Real_t * const __restrict__, const Real_t * const __restrict__, Real_t * const __restrict__, long int, double * const __restrict__ dfdt=nullptr) override;
    AnyODE::Status banded_jac_cmaj(Real_t, const Real_t * const __restrict__,  const Real_t * const __restrict__, Real_t * const __restrict__, long int) override;
    AnyODE::Status compressed_jac_cmaj(Real_t, const Real_t * const __restrict__, const Real_t * const __restrict__, Real_t * const __restrict__, long int);

    Real_t get_mod_k(int bi, int ri) const;

    // For iterative linear solver
    // void local_reaction_jac(const uint, const Real_t * const, Real_t * const __restrict__, Real_t) const;
    AnyODE::Status jac_times_vec(const Real_t * const __restrict__ vec,
                                 Real_t * const __restrict__ out,
                                 Real_t t, const Real_t * const __restrict__ y,
                                 const Real_t * const __restrict__ fy
                                 ) override;
    AnyODE::Status prec_setup(Real_t t, const Real_t * const __restrict__ y,
                              const Real_t * const __restrict__ fy,
                              bool jok, bool& jac_recomputed, Real_t gamma
                              ) override;
    AnyODE::Status prec_solve_left(const Real_t t, const Real_t * const __restrict__ y,
                                   const Real_t * const __restrict__ fy,
                                   const Real_t * const __restrict__ r,
                                   Real_t * const __restrict__ z,
                                   Real_t gamma,
                                   Real_t delta,
                                   const Real_t * const __restrict__ ewt
                                   ) override;

    void per_rxn_contrib_to_fi(Real_t, const Real_t * const __restrict__, uint, Real_t * const __restrict__) const;
    int get_geom_as_int() const;
    void calc_efield(const Real_t * const);

}; // class ReactionDiffusion

} // namespace chemreac
#endif // CHEMREAC_PVHQOBGMVZECTIJSMOKFUXJXXM
