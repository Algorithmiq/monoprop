#include <boost/test/unit_test.hpp>

#include "TestUtilities.h"
#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/mpi/MPICompat.h"

// Snapshot-invariance test (plan Phase 6).
//
// Calling the energy functional twice with identical parameters must give
// results that agree to tight numerical tolerance. Strict bit-equality is not
// guaranteed because inner_product uses TBB parallel_reduce_indices which is
// non-associative: sequential evaluations on the same graph can differ by 1 ULP
// due to dynamic work-stealing changing floating-point accumulation order.
//
// Even-parity vs Default backend comparison is covered by the existing
// fastpath_matches_mainline_* tests; no duplication needed here.

using namespace test_utils;
using namespace monoprop;

BOOST_FIXTURE_TEST_CASE(snapshot_invariance_repeated_evaluation, ExampleDataFix) {
    SimulatorConfig cfg{.comm = MPI_COMM_SELF};
    auto sim = build_simulator<n_modes>(data, cfg);
    sim.build_graph(data.majoranas, data.param_inds, data.gen_coeffs);

    auto fn = sim.expectation_value_functional();
    const double e1 = fn(data.parameters);
    const double e2 = fn(data.parameters);

    BOOST_CHECK_SMALL(e1 - e2, 1e-13);
    BOOST_TEST_MESSAGE("snapshot_invariance energy=" << e1);
}
