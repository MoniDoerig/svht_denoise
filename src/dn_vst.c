// Variance-stabilising transform for magnitude data, after Foi 2011.
//
// Magnitude noise is non-central chi: its spread shrinks and its mean is biased
// upwards as the signal nears the noise floor, which breaks the homoscedastic
// Gaussian model the threshold assumes.  The forward transform
//     f(x) = integral dx / sd(theta(x)),  theta(x): the signal whose mean is x,
// gives unit variance at every SNR and becomes the identity at high SNR.  The
// denoised value estimates E[f(M)], so the exact-unbiased inverse maps it back
// through E[f(M) | theta] to the bias-free signal theta.
//
// Every expectation is taken by quadrature over M's construction rather than its
// density: M = sqrt((theta + z)^2 + r^2), z ~ N(0,1) and r chi with 2L-1 degrees
// of freedom.  The integrand is smooth and Gaussian-weighted, so the trapezoid
// rule converges exponentially and no Bessel function is needed for any L.

#include <math.h>
#include <stdlib.h>

#include "dn.h"
#include "dn_vst.h"

#define TH_STEP 0.05   // theta grid, in sigma
#define TH_MAX 30.0    // beyond this, sd(M) is 1 to within 1e-4 and f is the identity
#define X_STEP 0.005   // uniform grid for evaluating f inside the quadrature
#define Q_STEP 0.25    // quadrature spacing, in sigma: as accurate as 0.1, 4x cheaper
#define REFINE 20      // slope refinements; see dn_vst_create

struct dn_vst {
	int nth;           // theta nodes
	double *mu;        // E[M | theta_i]
	double *sd;        // sd(M | theta_i)
	double *fmu;       // f(mu_i): the forward transform at the mean nodes
	double *ef;        // E[f(M) | theta_i], increasing
	double *slope;     // f' on each interval [mu_i, mu_i+1]
	int nx;            // uniform x grid for f
	double *fx;
	int nz, nr;        // quadrature nodes and weights
	double *z, *wz, *r, *wr;
};

void dn_vst_free(dn_vst *t) {
	if (!t) return;
	free(t->mu); free(t->sd); free(t->fmu); free(t->ef); free(t->slope); free(t->fx);
	free(t->z); free(t->wz); free(t->r); free(t->wr);
	free(t);
}

// Last index i with a[i] <= v, for increasing a; -1 below a[0].
static int floor_index(const double *a, int n, double v) {
	if (v < a[0]) return -1;
	int lo = 0, hi = n - 1;
	while (lo < hi) {
		const int mid = (lo + hi + 1) / 2;
		if (a[mid] <= v) lo = mid; else hi = mid - 1;
	}
	return lo;
}

// f on the mean nodes, continued linearly with the end slopes: 1/sd(0) below
// the noise-only mean, the identity above the table.
static double forward_nodes(const dn_vst *t, double x) {
	const int n = t->nth;
	const int i = floor_index(t->mu, n, x);
	if (i < 0) return t->fmu[0] - (t->mu[0] - x) / t->sd[0];
	if (i >= n - 1) return t->fmu[n - 1] + (x - t->mu[n - 1]);
	const double a = (x - t->mu[i]) / (t->mu[i + 1] - t->mu[i]);
	return t->fmu[i] + a * (t->fmu[i + 1] - t->fmu[i]);
}

double dn_vst_forward(const dn_vst *t, double x) {
	if (x < 0.0) x = 0.0;
	const double u = x / X_STEP;
	// Bound before converting: a large x / sigma overflows int, which is undefined.
	if (!(u < t->nx - 1)) return forward_nodes(t, x);
	const int i = (int)u;
	return t->fx[i] + (u - i) * (t->fx[i + 1] - t->fx[i]);
}

// E[g(M) | theta] for g = M, M^2, f(M) and f(M)^2.
static void moments(const dn_vst *t, double th, double *m1, double *m2, double *mf, double *mf2) {
	double s1 = 0.0, s2 = 0.0, sf = 0.0, sf2 = 0.0;
	for (int a = 0; a < t->nz; a++) {
		const double u = th + t->z[a];
		for (int b = 0; b < t->nr; b++) {
			const double w = t->wz[a] * t->wr[b];
			const double m2v = u * u + t->r[b] * t->r[b], m = sqrt(m2v);
			s1 += w * m;
			s2 += w * m2v;
			if (mf) {
				const double f = dn_vst_forward(t, m);
				sf += w * f;
				sf2 += w * f * f;
			}
		}
	}
	*m1 = s1;
	*m2 = s2;
	if (mf) { *mf = sf; *mf2 = sf2; }
}

// f on the mean nodes from the interval slopes, anchored so f is the identity at
// high SNR -- the denoised series keeps the input's scale wherever the signal is
// well above the floor -- then sampled onto the uniform grid.
static void integrate(dn_vst *t) {
	const int n = t->nth;
	t->fmu[0] = 0.0;
	for (int i = 1; i < n; i++) t->fmu[i] = t->fmu[i - 1] + (t->mu[i] - t->mu[i - 1]) * t->slope[i - 1];
	const double shift = t->mu[n - 1] - t->fmu[n - 1];
	for (int i = 0; i < n; i++) t->fmu[i] += shift;
	for (int i = 0; i < t->nx; i++) t->fx[i] = forward_nodes(t, i * X_STEP);
}

dn_vst *dn_vst_create(int ncoil) {
	dn_vst *t = (dn_vst *)dn_calloc(1, sizeof(dn_vst));
	if (!t) return NULL;
	const int k = 2 * ncoil - 1;   // degrees of freedom of r
	t->nth = (int)lround(TH_MAX / TH_STEP) + 1;
	t->nz = (int)lround(20.0 / Q_STEP) + 1;
	const double rmax = sqrt((double)k) + 10.0;
	t->nr = (int)ceil(rmax / Q_STEP) + 1;
	const double xmax = TH_MAX + sqrt(2.0 * k) + 10.0;
	t->nx = (int)ceil(xmax / X_STEP) + 1;
	t->mu = (double *)dn_malloc((size_t)t->nth, sizeof(double));
	t->sd = (double *)dn_malloc((size_t)t->nth, sizeof(double));
	t->fmu = (double *)dn_malloc((size_t)t->nth, sizeof(double));
	t->ef = (double *)dn_malloc((size_t)t->nth, sizeof(double));
	t->slope = (double *)dn_malloc((size_t)t->nth, sizeof(double));
	t->fx = (double *)dn_malloc((size_t)t->nx, sizeof(double));
	t->z = (double *)dn_malloc((size_t)t->nz, sizeof(double));
	t->wz = (double *)dn_malloc((size_t)t->nz, sizeof(double));
	t->r = (double *)dn_malloc((size_t)t->nr, sizeof(double));
	t->wr = (double *)dn_malloc((size_t)t->nr, sizeof(double));
	if (!t->mu || !t->sd || !t->fmu || !t->ef || !t->slope || !t->fx || !t->z || !t->wz || !t->r || !t->wr) {
		dn_vst_free(t);
		return NULL;
	}

	// Trapezoid nodes, weights normalised to sum to one: N(0,1) on [-10, 10], and
	// the chi density r^(k-1) exp(-r^2/2) on [0, rmax] with a half weight at 0.
	double sz = 0.0, sr = 0.0;
	for (int a = 0; a < t->nz; a++) {
		t->z[a] = -10.0 + a * Q_STEP;
		t->wz[a] = exp(-0.5 * t->z[a] * t->z[a]);
		sz += t->wz[a];
	}
	for (int b = 0; b < t->nr; b++) {
		t->r[b] = b * Q_STEP;
		t->wr[b] = pow(t->r[b], k - 1) * exp(-0.5 * t->r[b] * t->r[b]) * (b == 0 ? 0.5 : 1.0);
		sr += t->wr[b];
	}
	for (int a = 0; a < t->nz; a++) t->wz[a] /= sz;
	for (int b = 0; b < t->nr; b++) t->wr[b] /= sr;

	// Mean and spread at each theta, and the first-order transform f' = 1/sd.
	const int n = t->nth;
	for (int i = 0; i < n; i++) {
		double m1, m2;
		moments(t, i * TH_STEP, &m1, &m2, NULL, NULL);
		t->mu[i] = m1;
		t->sd[i] = sqrt(fmax(m2 - m1 * m1, 1e-12));
	}
	for (int i = 0; i < n - 1; i++) t->slope[i] = 0.5 * (1.0 / t->sd[i] + 1.0 / t->sd[i + 1]);
	t->slope[n - 1] = 1.0;
	integrate(t);

	// First order leaves var f(M) between 0.79 and 1.13 for Rician noise, because
	// near the floor the noise is not small against the curvature of f.  Foi's
	// refinement: divide each slope by the spread f(M) actually has there, and
	// repeat.  Monte Carlo, Rician: 0.93-1.13 after 2 rounds, 0.99-1.05 after 20.
	// The bottom node uses the first interval's correction too, so the linear
	// continuation below the noise-only mean stays consistent.
	for (int it = 0; it <= REFINE; it++) {
		for (int i = 0; i < n; i++) {
			double m1, m2, mf2;
			moments(t, i * TH_STEP, &m1, &m2, &t->ef[i], &mf2);
			// Not on the last pass: fx and fmu were built from this sd[0], and the
			// algebraic inverse extrapolates below mu[0] with it, so changing it now
			// broke the round trip there (5e-3 sigma at x = 0, where the Rayleigh
			// background lives, compounding over the sigma rounds).
			if (it < REFINE)
				t->sd[i] = sqrt(fmax(mf2 - t->ef[i] * t->ef[i], 1e-12)) / (i < n - 1 ? t->slope[i] : 1.0);
		}
		if (it == REFINE) break;   // ef matches the final f
		for (int i = 0; i < n - 1; i++) {
			const double sf_i = t->sd[i] * t->slope[i];
			t->slope[i] /= sf_i;
		}
		integrate(t);
	}
	return t;
}

double dn_vst_inverse(const dn_vst *t, double d, int unbiased) {
	const int n = t->nth;
	if (unbiased) {
		// E[f(M) | theta] is increasing; below its value at theta = 0 the best
		// noise-free estimate is no signal at all.
		const int i = floor_index(t->ef, n, d);
		if (i < 0) return 0.0;
		if (i >= n - 1) {
			// Above the table f is the identity and E[M]^2 -> theta^2 + (2L - 1), so
			// E[f(M)] ~ sqrt(theta^2 + c): a constant offset instead over-corrects
			// for many channels, by ~1 sigma at theta 63 with L = 64.  c is matched
			// to the last node, so the inverse is continuous there.
			const double th = (n - 1) * TH_STEP, c = t->ef[n - 1] * t->ef[n - 1] - th * th;
			return sqrt(fmax(d * d - c, th * th));
		}
		return (i + (d - t->ef[i]) / (t->ef[i + 1] - t->ef[i])) * TH_STEP;
	}
	// Algebraic: the x with f(x) = d, on the same piecewise-linear f.
	const int i = floor_index(t->fmu, n, d);
	if (i < 0) return fmax(t->mu[0] - (t->fmu[0] - d) * t->sd[0], 0.0);
	if (i >= n - 1) return t->mu[n - 1] + (d - t->fmu[n - 1]);
	const double a = (d - t->fmu[i]) / (t->fmu[i + 1] - t->fmu[i]);
	return t->mu[i] + a * (t->mu[i + 1] - t->mu[i]);
}
