// whole-volume driver.

#ifndef DN_RUN_H
#define DN_RUN_H

#include <stdint.h>

#include "dn_patch.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	const dn_geom *g;
	const float *img;       // nvox3d * nvol, volume-major
	const uint8_t *mask;    // nvox3d, or NULL for "everywhere"
	float *out;             // nvox3d * nvol, zero-initialised by the caller; NULL
	                        // to estimate sigma only
	const size_t *list;     // voxels to visit in place of the mask, or NULL
	float *noise;           // nvox3d, or NULL
	uint16_t *rank;         // nvox3d, or NULL
	size_t n_work;      // voxels visited: list length, mask count, or nvox3d
	int nthreads;
	// Overlapping-patch averaging (dn_agg.c); DN_AGG_EXCLUSIVE keeps one patch
	// per voxel, its centre row only.
	int aggregator;
	double fwhm;        // Gaussian FWHM, in units of the patch-centre spacing
	int stride;         // patch centres on every stride-th voxel
} dn_run;

enum { DN_AGG_EXCLUSIVE = 0, DN_AGG_GAUSSIAN, DN_AGG_UNIFORM };

// Denoise with overlapping patches: every patch's estimate is added, weighted,
// into each voxel it covers, in an order fixed by geometry alone, so the output
// is byte-identical at any thread count.  r->out must be zero-initialised.
int dn_agg_execute(const dn_run *r);

// Denoise every requested voxel.  Returns 0 on success.
//
// Output is byte-identical for any nthreads: each voxel's arithmetic depends
// only on its own patch, in a fixed order, and no two workers ever touch the
// same output element.  Scheduling therefore cannot influence the result.
int dn_run_execute(const dn_run *r);

// Number of hardware threads to use by default.
int dn_default_threads(void);

// The team size that will ACTUALLY be used for this geometry: `requested`,
// capped by the core count, by the number of work chunks, and by a scratch-memory
// budget.  Exposed
// so the CLI reports the effective count rather than the requested one.
int dn_effective_threads(const dn_geom *g, size_t n_work, int requested);

// Scratch bytes per worker, to the dominant term; full adds dn_work_full's.
size_t dn_worker_bytes(const dn_geom *g, int full);

// `requested` capped by the core count, by `units` of work (0: no cap) and by a
// 1 GiB scratch budget at `per` bytes a worker; never below 1.
int dn_cap_threads(int requested, size_t units, size_t per);

// Report a worker pool's outcome, shared by both drivers: non-zero on a scratch
// or convergence failure, a note for full-solve fallbacks.
int dn_pool_report(int failed, unsigned long eig_fail, unsigned long fallbacks);

// The team size dn_agg_execute will ACTUALLY use: `requested`, capped by the
// core count, by the most tiles any one colour holds, and by the scratch budget.
int dn_agg_threads(const dn_geom *g, int stride, int requested);

#ifdef __cplusplus
}
#endif

#endif // DN_RUN_H
