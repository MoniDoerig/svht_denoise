// Overlapping-patch averaging (Manjon et al. 2013): every patch's low-rank
// estimate is added, weighted, into EVERY voxel it covers, instead of keeping
// only its centre row.  Each voxel then averages many estimates, not one.
//
// Determinism.  Contributions to one voxel come from many patches, and float
// addition is not associative, so the ORDER of those additions must not depend
// on scheduling.  Patch centres are split into tiles whose size is fixed by the
// geometry alone, and tiles are coloured 2x2x2.  Same-colour tiles never write
// the same voxel, so a colour runs fully parallel; colours run in a fixed order,
// one tile per worker, centres in raster order.  Every voxel therefore receives
// its contributions in the same order at any thread count.
//
// Why a tile is so wide: an edge-shifted cube reaches k-1 voxels on ONE side of
// its anchor, and an edge-grown sphere further still, so a tile's writes span its
// own width plus `reach` on both sides.  Two same-colour tiles are one tile apart,
// which needs T >= 2*reach; tile_grid uses 2*reach + 1.  Do not "simplify" T to
// the patch width: same-colour tiles would then race at the image edges.

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "dn.h"
#include "dn_run.h"

typedef struct {
	const dn_run *r;
	pthread_mutex_t lock;
	const size_t *tiles;  // this colour's tiles
	size_t ntiles, next;
	int T[3], nt[3], reach[3];
	double *wax[3];       // per-axis weights, indexed by offset + reach
	double *wsum, *racc, *sacc;
	int failed;
	unsigned long eig_fail, fallbacks;
} dn_agg;

// Tile index -> its 2x2x2 colour.
static int tile_colour(size_t t, const int nt[3]) {
	const size_t tx = t % (size_t)nt[0], ty = (t / (size_t)nt[0]) % (size_t)nt[1];
	const size_t tz = t / ((size_t)nt[0] * nt[1]);
	return (int)((tx & 1) | (ty & 1) << 1 | (tz & 1) << 2);
}

// Tile width and count per axis: T >= 2*reach + 1, rounded up to whole strides
// so every tile starts on the centre grid (see the head of this file).
static void tile_grid(const dn_geom *g, int stride, int reach[3], int T[3], int nt[3]) {
	const int dims[3] = {g->nx, g->ny, g->nz};
	dn_patch_reach(g, reach);
	for (int a = 0; a < 3; a++) {
		T[a] = (2 * reach[a] + 1 + stride - 1) / stride * stride;
		nt[a] = (dims[a] + T[a] - 1) / T[a];
	}
}

int dn_agg_threads(const dn_geom *g, int stride, int requested) {
	int reach[3], T[3], nt[3], count[8] = {0}, maxper = 0;
	tile_grid(g, stride, reach, T, nt);
	const size_t ntile = (size_t)nt[0] * nt[1] * nt[2];
	for (size_t t = 0; t < ntile; t++) count[tile_colour(t, nt)]++;
	for (int c = 0; c < 8; c++) if (count[c] > maxper) maxper = count[c];
	return dn_cap_threads(requested, (size_t)maxper, dn_worker_bytes(g, 1));
}

// A centre is denoised when its patch touches the mask.  That is exactly the set
// of patches that write to some masked voxel, so every masked voxel receives the
// same contributions, in the same order, as in an unmasked run: the masked output
// is identical to the unmasked one inside the mask, the promise -help makes.
static int centre_wanted(const dn_run *r, dn_work *w, int x, int y, int z) {
	if (!r->mask) return 1;
	if (dn_patch_rows(r->g, w, x, y, z)) return 0;
	for (int i = 0; i < r->g->m; i++)
		if (r->mask[(size_t)((ptrdiff_t)w->base + w->rows[i])]) return 1;
	return 0;
}

static void *agg_worker(void *arg) {
	dn_agg *p = (dn_agg *)arg;
	const dn_run *r = p->r;
	const dn_geom *g = r->g;
	const int m = g->m, n = g->nvol, nx = g->nx, ny = g->ny, s = r->stride;
	const size_t nxy = (size_t)nx * ny;

	dn_work *w = dn_work_create(g);
	double *q = (double *)dn_malloc((size_t)m, sizeof(double));   // per-row update coefficient
	size_t *rv = (size_t *)dn_malloc((size_t)m, sizeof(size_t));
	if (!w || dn_work_full(w, g) || !q || !rv) {
		pthread_mutex_lock(&p->lock);
		p->failed = 1;
		pthread_mutex_unlock(&p->lock);
		dn_work_free(w); free(q); free(rv);
		return NULL;
	}
	unsigned long fail = 0;

	for (;;) {
		pthread_mutex_lock(&p->lock);
		if (p->failed || p->next >= p->ntiles) {
			pthread_mutex_unlock(&p->lock);
			break;
		}
		const size_t t = p->tiles[p->next++];
		pthread_mutex_unlock(&p->lock);

		const int tx = (int)(t % (size_t)p->nt[0]);
		const int ty = (int)((t / (size_t)p->nt[0]) % (size_t)p->nt[1]);
		const int tz = (int)(t / ((size_t)p->nt[0] * p->nt[1]));   // tile position
		const int x0 = tx * p->T[0], y0 = ty * p->T[1], z0 = tz * p->T[2];
		const int x1 = x0 + p->T[0] < nx ? x0 + p->T[0] : nx;
		const int y1 = y0 + p->T[1] < ny ? y0 + p->T[1] : ny;
		const int z1 = z0 + p->T[2] < g->nz ? z0 + p->T[2] : g->nz;

		for (int z = z0; z < z1; z += s)
			for (int y = y0; y < y1; y += s)
				for (int x = x0; x < x1; x += s) {
					if (!centre_wanted(r, w, x, y, z)) continue;
					float sigma = 0.0f;
					uint16_t rank = 0;
					int empty = 0;
					if (dn_denoise_patch(g, w, r->img, x, y, z, &sigma, &rank, &empty)) {
						fail++;
						continue;
					}
					if (empty) continue;   // an all-zero patch has nothing to say
					for (int i = 0; i < m; i++) {
						const size_t v = (size_t)((ptrdiff_t)w->base + w->rows[i]);
						const int vx = (int)(v % (size_t)nx), vy = (int)((v / (size_t)nx) % (size_t)ny);
						const int vz = (int)(v / nxy);
						rv[i] = v;
						const double wq = p->wax[0][vx - x + p->reach[0]] * p->wax[1][vy - y + p->reach[1]] *
						                  p->wax[2][vz - z + p->reach[2]];
						// Running weighted means: a contribution enters with coefficient
						// wq / (weights so far), the first with exactly 1, so nothing
						// scale-dependent can underflow to 0 (see AGENTS.md).
						p->wsum[v] += wq;
						q[i] = wq > 0.0 ? wq / p->wsum[v] : 0.0;
						if (p->racc) p->racc[v] += q[i] * (rank - p->racc[v]);
						if (p->sacc) p->sacc[v] += q[i] * (sigma - p->sacc[v]);
					}
					for (int j = 0; j < n; j++) {
						const double *xr = w->xhat + (size_t)j * m;
						float *o = r->out + (size_t)j * g->nvox3d;
						for (int i = 0; i < m; i++)
							o[rv[i]] = (float)((double)o[rv[i]] + q[i] * (xr[i] - (double)o[rv[i]]));
					}
				}
	}
	const unsigned long fb = dn_eig_fallbacks(w->eig);
	dn_work_free(w);
	free(q);
	free(rv);
	if (fail || fb) {
		pthread_mutex_lock(&p->lock);
		p->eig_fail += fail;
		p->fallbacks += fb;
		pthread_mutex_unlock(&p->lock);
	}
	return NULL;
}

int dn_agg_execute(const dn_run *r) {
	const dn_geom *g = r->g;
	const int s = r->stride;
	int status = 1;
	dn_agg p;
	memset(&p, 0, sizeof(p));
	p.r = r;
	tile_grid(g, s, p.reach, p.T, p.nt);

	const double sp[3] = {g->dx, g->dy, g->dz};
	// Gaussian on the distance in mm from the patch's anchor voxel, separable so
	// it is three small tables.  FWHM is r->fwhm times the geometric mean spacing
	// of the patch centres, as dwidenoise2 defines it.
	const double fwhm = r->fwhm * s * cbrt(sp[0] * sp[1] * sp[2]);
	const double c = -4.0 * log(2.0) / (fwhm * fwhm);
	for (int a = 0; a < 3; a++) {
		p.wax[a] = (double *)dn_malloc((size_t)(2 * p.reach[a] + 1), sizeof(double));
		if (!p.wax[a]) goto done;
		for (int t = -p.reach[a]; t <= p.reach[a]; t++)
			p.wax[a][t + p.reach[a]] = r->aggregator == DN_AGG_UNIFORM ? 1.0 : exp(c * (t * sp[a]) * (t * sp[a]));
	}

	const size_t nv = g->nvox3d, ntile = (size_t)p.nt[0] * p.nt[1] * p.nt[2];
	size_t *tiles = (size_t *)dn_malloc(ntile, sizeof(size_t));
	p.wsum = (double *)dn_calloc(nv, sizeof(double));
	if (r->rank) p.racc = (double *)dn_calloc(nv, sizeof(double));
	if (r->noise) p.sacc = (double *)dn_calloc(nv, sizeof(double));
	if (!tiles || !p.wsum || (r->rank && !p.racc) || (r->noise && !p.sacc)) goto free_tiles;
	if (pthread_mutex_init(&p.lock, NULL) != 0) goto free_tiles;

	const int nthreads = dn_agg_threads(g, s, r->nthreads);
	for (int colour = 0; colour < 8 && !p.failed; colour++) {
		size_t k = 0;
		for (size_t t = 0; t < ntile; t++)
			if (tile_colour(t, p.nt) == colour) tiles[k++] = t;
		p.tiles = tiles;
		p.ntiles = k;
		p.next = 0;
		// Never more workers than this colour has tiles: a surplus one would
		// allocate a full scratch arena only to find the queue empty.
		if (k) dn_thread_run(agg_worker, &p, (size_t)nthreads < k ? nthreads : (int)k);
	}
	pthread_mutex_destroy(&p.lock);
	if (dn_pool_report(p.failed, p.eig_fail, p.fallbacks)) goto free_tiles;

	// Restore the shell means (the buffers already hold weighted means) and zero
	// everything outside the mask.
	// A masked voxel no patch covered (an all-zero neighbourhood, or a sphere on
	// anisotropic voxels that misses a diagonal) falls back to its own patch.
	size_t nfb = 0;
	for (size_t v = 0; v < nv; v++)
		if ((!r->mask || r->mask[v]) && !(p.wsum[v] > 0.0)) nfb++;
	size_t *fb = (size_t *)dn_malloc(nfb ? nfb : 1, sizeof(size_t));
	if (!fb) goto free_tiles;
	nfb = 0;
	for (size_t v = 0; v < nv; v++) {
		const int inside = !r->mask || r->mask[v];
		if (inside && p.wsum[v] > 0.0) {
			for (int j = 0; j < g->nvol; j++) {
				float *o = r->out + v + (size_t)j * nv;   // already the weighted mean
				*o = (float)(*o + dn_shell_mean(g, v, j));
			}
			if (r->rank) r->rank[v] = (uint16_t)lround(p.racc[v]);
			if (r->noise) r->noise[v] = (float)p.sacc[v];
		} else {
			for (int j = 0; j < g->nvol; j++) r->out[v + (size_t)j * nv] = 0.0f;
			if (inside) fb[nfb++] = v;
		}
	}
	if (nfb) {
		dn_run e = *r;
		e.mask = NULL;
		e.list = fb;
		e.n_work = nfb;
		if (dn_run_execute(&e)) { free(fb); goto free_tiles; }
	}
	free(fb);
	status = 0;

free_tiles:
	free(tiles);
	free(p.wsum);
	free(p.racc);
	free(p.sacc);
done:
	for (int a = 0; a < 3; a++) free(p.wax[a]);
	return status;
}
