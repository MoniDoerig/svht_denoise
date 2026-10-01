// DWI denoising by local PCA with the singular value rules of Gavish & Donoho:
// the optimal hard threshold (IEEE Trans. Inf. Theory 60(8):5040-5053, 2014) and
// the Frobenius-optimal shrinkage (63(4):2137-2152, 2017, the default), applied
// patch-wise to a 4D diffusion series.  This is NOT Marchenko-Pastur PCA and is
// not an emulation of MRtrix3 dwidenoise; the noise level comes from their median
// estimator, not a Marchenko-Pastur fit.

#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>    // readlink, unlink

#include "dn.h"
#include "dn_bval.h"
#include "dn_nii.h"
#include "dn_patch.h"
#include "dn_phase.h"
#include "dn_run.h"
#include "dn_sigma.h"
#include "dn_vst.h"
#ifdef DN_DEGIBBS
#include "mrdegibbs/dg.h"

// The factor as the fraction everyone writes, for the run summary: -help and the
// README both say 7/8 and 6/8, not 0.875 and 0.75.  Empty for full k-space.
static const char *pf_note(double pf) {
	return pf == DG_PF_6_8 ? ", partial Fourier 6/8"
	     : pf == DG_PF_7_8 ? ", partial Fourier 7/8" : "";
}
#endif

// Noise-map estimates under -vst: the first from raw magnitudes, each later one
// corrected in the stabilised domain.  Two, measured on Rician synthetic truth
// (benchmark/): RMSE at sigma 30 / 60 was 13.77 / 30.24 with one, 13.49 / 25.20
// with two, 13.75 / 25.30 with three, and worse beyond -- the correction does not
// converge on the true sigma but overshoots it (64 against 60 after six).
#define DN_VST_ROUNDS 2

#define DN_VERSION "0.1.20261001"

// -degibbs: no, yes (after denoising), or only (instead of denoising).
enum { DN_DG_NO = 0, DN_DG_YES, DN_DG_ONLY };

static void usage(void) {
	printf("svht_denoise %s -- DWI denoising by local PCA with optimal singular value shrinkage\n", DN_VERSION);
	printf("\n");
	printf("USAGE\n");
	printf("  svht_denoise <input> <output> [options]\n");
	printf("\n");
	printf("  <input>   4D NIfTI diffusion series, at least 2 volumes.  Real datatypes\n");
	printf("            only: a complex-valued NIfTI is rejected.  For complex data,\n");
	printf("            pass the magnitude here and the phase via -phase.\n");
	printf("            .nii and .nii.gz are supported; the reader also accepts\n");
	printf("            whatever else this NIfTI build supports.\n");
	printf("  <output>  denoised series, always written as float32\n");
	printf("\n");
	printf("OPTIONS\n");
	printf("  -phase <image>    treat <input> as the MAGNITUDE of a complex series and\n");
	printf("                    <image> as its phase, and rotate the two onto the real\n");
	printf("                    axis before denoising.  Noise in the rotated data is\n");
	printf("                    the original zero-mean Gaussian rather than the Rician\n");
	printf("                    of a magnitude image, so the low-SNR noise floor is\n");
	printf("                    avoided instead of denoised.  The phase must match the\n");
	printf("                    input in dimensions and world transform.\n");
#ifdef DN_DEGIBBS
	printf("                    The OUTPUT IS THEN SIGNED, unless -degibbs is also\n");
	printf("                    given, which gives that up.  If the phase turns out to\n");
#else
	printf("                    The OUTPUT IS THEN SIGNED.  If the phase turns out to\n");
#endif
	printf("                    encode no rotation at all, that is reported rather than\n");
	printf("                    silently returning the magnitude.\n");
	printf("  -phaseunits <u>   how to read the phase values: radians, degrees, turns\n");
	printf("                    (or its synonym cycles), or auto.  Default auto, which\n");
	printf("                    maps the observed range onto one full turn -- except\n");
	printf("                    that a range already within 0.1 of 2*pi is kept as\n");
	printf("                    radians unchanged.  That is correct for any encoding\n");
	printf("                    that COVERS a turn, which every acquired\n");
	printf("                    phase image does -- its background is noise with\n");
	printf("                    uniform phase.  Say the unit explicitly for data that\n");
	printf("                    does not cover one: a range of -0.5 to 0.5 is half a\n");
	printf("                    radian or one whole cycle, and the file cannot say.\n");
	printf("                    The convention used is reported unless -quiet, so\n");
	printf("                    check it.  Needs -phase.\n");
	printf("  -real <image>     write the real-axis-rotated input, before denoising\n");
	printf("                    (float32).  Needs -phase.\n");
	printf("  -mask <image>     only denoise voxels where the mask is > 0.\n");
	printf("                    Patches still read the whole image, so a masked run is\n");
	printf("                    identical to an unmasked one inside the mask; voxels\n");
	printf("                    outside the mask are written as zero.\n");
	printf("  -noise <image>    write the estimated noise level (float32)\n");
	printf("  -rank <image>     write the number of retained components (uint16)\n");
	printf("  -bval <file>      FSL b-values, one per volume.  Default: the input's name\n");
	printf("                    with .nii/.nii.gz replaced by .bval.  Needs -demean y.\n");
	printf("  -demean <y/n>     remove each voxel's mean over each b-value shell before\n");
	printf("                    PCA and restore it after, which lowers the signal rank.\n");
	printf("                    Default n: measured, it does not help.  Shells are\n");
	printf("                    clustered as MRtrix3 does (b <= 10 is b=0; neighbours\n");
	printf("                    less than 80 s/mm^2 apart share a shell); a shell of one\n");
	printf("                    volume is not demeaned.\n");
	printf("  -filter <f>       how components are kept: optshrink (default), the\n");
	printf("                    Frobenius-optimal shrinkage of Gavish & Donoho 2017;\n");
	printf("                    optthresh, their 2014 unknown-noise hard threshold; or\n");
	printf("                    truncate, a hard cut at the noise bulk edge.  optshrink\n");
	printf("                    and truncate read sigma from a noise map estimated\n");
	printf("                    first, from larger patches on every %dth voxel.\n", DN_SIGMA_STRIDE);
	printf("  -shape <s>        patch shape: sphere (default) or cube.  The sphere is\n");
	printf("                    measured in mm, so it stays round on anisotropic voxels,\n");
	printf("                    and holds the fewest whole distance shells with at\n");
	printf("                    least k^3 voxels (k from -extent).  At the image edge it\n");
	printf("                    is the same number of voxels nearest the one denoised.\n");
	printf("  -aggregator <a>   how patches make the output: gaussian (default), every\n");
	printf("                    patch adds its estimate to every voxel it covers,\n");
	printf("                    weighted by distance from its centre (Manjon et al.\n");
	printf("                    2013); uniform, the same unweighted; or exclusive, each\n");
	printf("                    voxel from the one patch centred on it.\n");
	printf("  -aggregator_fwhm <f>  Gaussian width, in units of the patch-centre spacing\n");
	printf("                    (default 2).  Needs -aggregator gaussian.\n");
	printf("  -stride <s>       centre a patch on every s-th voxel per axis, 1 or 2,\n");
	printf("                    which runs ~1/8 as many.  Default 2, or 1 under\n");
	printf("                    -aggregator exclusive, which needs a patch per voxel.\n");
	printf("  -vst <y/n>        variance-stabilise MAGNITUDE data before denoising: map\n");
	printf("                    its Rician / non-central chi noise to unit-variance\n");
	printf("                    Gaussian (Foi 2011), and map back after.  Default y,\n");
	printf("                    or n with -phase, whose rotated data are already\n");
	printf("                    Gaussian.  By default the way back is the exact-unbiased\n");
	printf("                    inverse, which also removes the noise-floor bias.  Also\n");
	printf("                    n by default for input with any negative value, which\n");
	printf("                    cannot be a magnitude image.\n");
	printf("  -noise_dof <L>    receive channels behind each magnitude (sum of squares),\n");
	printf("                    so the noise is non-central chi with 2L degrees of\n");
	printf("                    freedom.  Default 1, i.e. Rician.  Needs -vst y.\n");
	printf("  -preserve_noise_bias  map back algebraically instead, keeping the noise-floor\n");
	printf("                    bias of an ordinary magnitude image.  Needs -vst y.\n");
	printf("  -extent <k>       patch size: cube side length, odd, k^3 > number of\n");
	printf("                    volumes; a sphere holds at least k^3 voxels.\n");
	printf("                    Default: the smallest odd k that satisfies this.\n");
#ifdef DN_DEGIBBS
	printf("  -degibbs <y/n/o>  remove Gibbs ringing by local subvoxel shifts: yes, no\n");
	printf("                    (default) or only.  \"yes\" denoises first and degibbses\n");
	printf("                    the result, which is the order these belong in; \"only\"\n");
	printf("                    skips the denoising entirely and accepts a 3D image.\n");
	printf("                    Use -pF 0.875 or 0.75 for supported partial-Fourier\n");
	printf("                    acquisitions; without it this assumes full k-space.\n");
	printf("                    Assumes MAGNITUDE data: negative input is truncated to\n");
	printf("                    zero, which is a no-op on a magnitude image and lossy on\n");
	printf("                    a -phase rotated one.  The output can still go negative,\n");
	printf("                    because ringing correction undershoots -- except under\n");
	printf("                    -pF 0.875, which clamps its output as its reference does.\n");
	printf("                    In-plane dimensions must be at least 5, and at least 5\n");
	printf("                    again after a -pF factor splits them.\n");
	printf("                    \"only\" also refuses -noise, -rank, -phase, -real and\n");
	printf("                    -extent, which all describe a denoiser it is not running.\n");
	printf("                    Cannot be combined with -mask: a mask leaves a hard\n");
	printf("                    zero edge, which is exactly what this method rings on.\n");
	printf("                    Within-slice axes are x and y.  Adapted from MRtrix3\n");
	printf("                    mrdegibbs (MPL-2.0); method of Kellner et al. 2016.\n");
	printf("  -pF <factor>      partial-Fourier factor: 0.875 (7/8) or 0.75 (6/8).\n");
	printf("                    Asymmetric k-space truncation adds a SECOND ringing along\n");
	printf("                    y that -degibbs alone leaves in place; this removes it too\n");
	printf("                    (Lee et al. 2021).  Needs -degibbs, assumes y is the\n");
	printf("                    phase-encode direction, and 6/8 needs an even y dimension.\n");
	printf("                    1.0 is accepted and means full k-space, the default.\n");
#endif
	printf("  -nthreads <n>     worker threads (default: all cores). -p is an alias.\n");
	printf("  -quiet            suppress the run summary on stderr\n");
	printf("  -version, --version   print the version and exit\n");
	printf("  -help, --help, -h     print this message and exit\n");
	printf("\n");
	printf("NOTES\n");
	printf("  An input of \"-\" is read from standard input.  It has no name on disk,\n");
	printf("  so it is exempt from the collision rule below.  \"-\" is NOT accepted as\n");
	printf("  an output: <output>, -noise, -rank and -real each need a filename.\n");
	printf("  Every input and output must be a distinct file, compared as the files\n");
	printf("  actually opened and written rather than as typed.  A collision is\n");
	printf("  refused rather than silently overwriting one of them.\n");
	printf("  -nthreads is a request: it is capped by the core count, by the amount of\n");
	printf("  work available, and by a scratch-memory budget.  The summary reports the\n");
	printf("  number that will really run.\n");
	printf("  No noise level need be supplied: it is estimated from the data.\n");
	printf("  '-filter optthresh -aggregator exclusive -shape cube -vst n' reproduces\n");
	printf("  the original pipeline exactly.\n");
	printf("  .bval is read only to group volumes into shells for -demean; gradient\n");
	printf("  directions (.bvec) are never used.\n");
	printf("  Non-finite input is rejected rather than propagated, because any input\n");
	printf("  voxel can enter a patch.\n");
}

// Follow a chain of symlinks by hand, into `out`.
//
// realpath() cannot do this job: it fails outright when the final component does
// not exist, which is exactly the case its caller is handling.  A DANGLING
// symlink -- `ln -s noise.nii out.nii` before noise.nii has been written --
// makes stat() fail on BOTH sides, so without this the link's own name is
// compared against its target's, the two look distinct, and one output silently
// overwrote the other at exit 0.
//
// Returns 0 when the chain was followed to a non-link, non-zero when it could
// not be: too many hops, an unreadable link, or a name that will not fit.  That
// distinction matters -- leaving `out` at a partly-resolved name and calling it
// resolved is NOT conservative, because the caller would then compare two names
// that the kernel will still resolve further at write time.
//
// The hop limit is above both platforms' own: macOS SYMLOOP_MAX is 32 and Linux
// follows 40, so a 33-40 link alias would have gone unresolved here and resolved
// during the write -- exactly the window this function exists to close.
#define DN_MAX_SYMLINK_HOPS 64
static int resolve_links(const char *in, char *out, size_t outsz) {
	if (snprintf(out, outsz, "%s", in) >= (int)outsz) return 1;
	for (int hops = 0; hops < DN_MAX_SYMLINK_HOPS; hops++) {
		struct stat st;
		if (lstat(out, &st) != 0 || !S_ISLNK(st.st_mode)) return 0;   // not a link: done
		char target[PATH_MAX];
		const ssize_t n = readlink(out, target, sizeof target - 1);
		if (n <= 0) return 1;
		target[n] = '\0';

		char next[PATH_MAX];
		if (target[0] == '/') {
			if (snprintf(next, sizeof next, "%s", target) >= (int)sizeof next) return 1;
		} else {
			// Relative targets are relative to the LINK's directory, not the cwd.
			char dir[PATH_MAX];
			if (snprintf(dir, sizeof dir, "%s", out) >= (int)sizeof dir) return 1;
			char *slash = strrchr(dir, '/');
			if (slash) {
				*slash = '\0';
				if (snprintf(next, sizeof next, "%s/%s", dir[0] ? dir : "/", target) >= (int)sizeof next) return 1;
			} else if (snprintf(next, sizeof next, "%s", target) >= (int)sizeof next) return 1;
		}
		if (snprintf(out, outsz, "%s", next) >= (int)outsz) return 1;
	}
	return 1;   // hop limit reached: a cycle, or a chain longer than any kernel follows
}

// Are these two paths the same file?  (The POLICY that they must not be lives
// on check_path_collisions below; this answers only the mechanical question.)
//
// By inode, always.  When both already exist that is a plain stat; when one or
// both do not -- the usual case for outputs -- they are CREATED empty, compared,
// and removed again.
//
// Creating them is the point.  Every attempt to answer this from the NAMES was
// wrong in a way that lost a file, because the rule belongs to the filesystem:
//   - case-insensitive mounts exist on both platforms, so strcmp under-matches
//     and strcasecmp over-matches, and which applies is per-mount;
//   - Apple filesystems are normalization-insensitive -- NFC and NFD spellings
//     of one name are one file -- INCLUDING when formatted case-sensitive;
//   - the case fold maps non-ASCII onto ASCII, so U+212A KELVIN SIGN and "k"
//     are one file, which defeats any "pure ASCII must be distinct" shortcut.
// Doing that properly by hand needs CoreFoundation, which this tool does not
// link and package_macos.sh actively verifies it does not.  The filesystem
// answers all of it, exactly as it will at write time, for about ten lines.
static int same_file(const char *a, const char *b) {
	// Resolve first, so that creating through a dangling symlink -- and removing
	// what we created -- both act on the target rather than on the link.
	char la[PATH_MAX], lb[PATH_MAX];
	if (resolve_links(a, la, sizeof la) || resolve_links(b, lb, sizeof lb)) return 1;

	struct stat sa, sb;
	const int had_a = (stat(la, &sa) == 0);
	const int had_b = (stat(lb, &sb) == 0);
	if (had_a && had_b) return sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino;

	const int fa = open(la, O_CREAT | O_RDWR, 0600);
	const int fb = (fa >= 0) ? open(lb, O_CREAT | O_RDWR, 0600) : -1;
	if (fa >= 0) close(fa);
	if (fb >= 0) close(fb);

	int same;
	if (fa < 0 || fb < 0) {
		// Could not create: a missing or unwritable directory, most likely.
		// Nothing can be written there either, so nothing can be lost.  Fall back
		// to an exact match and let the write report the real problem, which is a
		// far better message than a collision would be.
		same = (strcmp(la, lb) == 0);
	} else {
		struct stat xa, xb;
		same = (stat(la, &xa) == 0 && stat(lb, &xb) == 0 &&
		        xa.st_dev == xb.st_dev && xa.st_ino == xb.st_ino);
	}
	// ONLY what did not exist a moment ago.  Keying the cleanup off "the open
	// succeeded" instead deletes an existing INPUT, which is the opposite of this
	// function's purpose -- the suite caught it on the first run.
	if (!had_b && fb >= 0) unlink(lb);
	if (!had_a && fa >= 0) unlink(la);
	return same;
}

// Every input and output must be a different file.  Each output is written
// separately from its own buffer, so two sharing a name leave only the last; an
// output naming an INPUT destroys it.  Both at exit 0 without this.
//
// The comparison is over the names nifti will REALLY use, not the ones the user
// typed: the two differ whenever an extension is missing or is half of a
// .hdr/.img pair, and comparing the typed strings let `svht_denoise in.nii in`
// normalise the output onto the input and destroy it.  `nifti_type` decides .nii
// versus .hdr/.img for an extensionless OUTPUT prefix, and comes from the loaded
// input because that is what the outputs inherit.
//
// Each path carries its own label and its own input/output classification, in
// one struct.  An earlier version kept those in parallel arrays indexed by slot,
// where the index silently decided three separate things -- which resolver ran,
// which read/write error was printed, and which refusal text was used -- under
// an ordering invariant that lived only in a comment.  Nothing here may depend
// on the order of the table.
typedef struct {
	const char *label;    // how this path is named in messages
	const char *prefix;   // what the user typed, or NULL if unused
	int is_input;         // read from, so must survive the run; 2: a raw file, not NIfTI
	char *hdr, *img;      // resolved; owned by this struct
} dn_path;

static int check_path_collisions(dn_path *p, int n, int nifti_type) {
	int rc = 0;

	// Inputs and outputs are resolved by DIFFERENT rules, and using the output
	// rule on an input is not a cosmetic slip: it is what let `-phase ph -noise
	// ph.hdr` overwrite an existing ph.hdr/ph.img pair, because "ph" was modelled
	// as the "ph.nii" an output would create and so never matched.  See
	// dn_resolve_input_names.
	//
	// The main input is resolved here like any other, rather than borrowing the
	// strings the reader already produced.  It is the same code path and gives
	// the same answer, for one extra header open measured at ~0.1 ms; an
	// ownership exception for a single row cost more than that in reader
	// attention.
	for (int i = 0; i < n; i++) {
		if (!p[i].prefix) continue;
		// "-" is stdin, which nifti accepts as an input.  There is no file on disk
		// for an output to collide with, and a header-only reopen of a stream the
		// reader has already drained cannot succeed -- so skip the row rather than
		// fail the run.  (Resolving it unconditionally is what briefly broke
		// `svht_denoise - out.nii < in.nii`.)
		if (p[i].is_input && p[i].prefix[0] == '-' && p[i].prefix[1] == '\0') continue;
		// A raw file (the .bval) is opened exactly as named: both "halves" are it.
		const int bad = p[i].is_input == 2
		              ? !(p[i].hdr = strdup(p[i].prefix)) || !(p[i].img = strdup(p[i].prefix))
		              : p[i].is_input
		              ? dn_resolve_input_names(p[i].prefix, &p[i].hdr, &p[i].img)
		              : dn_resolve_names(p[i].prefix, nifti_type, &p[i].hdr, &p[i].img);
		if (bad) {
			if (p[i].is_input) dn_err("unable to read %s image '%s'\n", p[i].label, p[i].prefix);
			else dn_err("cannot use '%s' as the %s name\n", p[i].prefix, p[i].label);
			rc = 1;
			goto done;
		}
	}

	for (int i = 0; i < n; i++) {
		if (!p[i].hdr) continue;
		for (int j = i + 1; j < n; j++) {
			if (!p[j].hdr) continue;
			// Both members of each pair, since .hdr and .img can collide crosswise.
			const char *a[2] = {p[i].hdr, p[i].img};
			const char *b[2] = {p[j].hdr, p[j].img};
			for (int u = 0; u < 2; u++) {
				for (int w = 0; w < 2; w++) {
					if (!same_file(a[u], b[w])) continue;
					dn_err("the %s and %s paths both resolve to '%s'.\n",
					       p[i].label, p[j].label, a[u]);
					// Two INPUTS colliding costs nothing on disk, so the reason to
					// refuse is different: they are distinct images by definition
					// -- a magnitude is not its own phase -- and one file named for
					// both is a mistake that would otherwise produce quiet nonsense.
					if (p[i].is_input && p[j].is_input) {
						dn_err("  Refusing to run: these are two different images, so one\n");
						dn_err("  file cannot be both.\n");
					} else {
						dn_err("  Refusing to run: outputs are written separately, so this would\n");
						// Either side may be the input: nothing here may assume the
						// inputs come first in the table.
						dn_err("  discard one of them%s.\n",
						       (p[i].is_input || p[j].is_input) ? " and destroy an input" : "");
					}
					rc = 1;
					goto done;
				}
			}
		}
	}

done:
	for (int i = 0; i < n; i++) { free(p[i].hdr); free(p[i].img); }
	return rc;
}

static int parse_int(const char *s, int *out) {
	char *end = NULL;
	long v = strtol(s, &end, 10);
	if (!end || *end != '\0' || end == s) return 1;
	if (v < INT32_MIN || v > INT32_MAX) return 1;
	*out = (int)v;
	return 0;
}

// One stabilising pass over the live voxels: magnitude / sigma -> f.  `prev` is
// the sigma each voxel is currently in units of, ignored when `first`; later
// passes recover the raw magnitude through f's exact algebraic inverse instead of
// keeping a second copy of the series.  `prev` is updated to `sigma`.
static void vst_stabilise(const dn_vst *vt, float *data, size_t nvox3d, int nvol, const uint8_t *live,
                          const float *sigma, float *prev, int first) {
	for (size_t v = 0; v < nvox3d; v++) {
		if (!live[v]) continue;
		const double s = sigma[v], p = first ? 0.0 : prev[v];
		for (int j = 0; j < nvol; j++) {
			float *x = data + v + (size_t)j * nvox3d;
			const double raw = first ? *x : p * dn_vst_inverse(vt, *x, 0);
			*x = (float)dn_vst_forward(vt, raw / s);
		}
		prev[v] = (float)s;
	}
}

// Map a denoised, stabilised series back to magnitude: every sample of a live,
// in-mask voxel -- a denoised 0 is a value too, and the algebraic inverse maps it
// onto the noise floor.  The denoiser saw sigma = 1, so the noise map it wrote is
// replaced by the data's own.
static void vst_restore(const dn_vst *vt, float *out, float *noise, size_t nvox3d, int nvol,
                        const uint8_t *live, const uint8_t *mask, const float *sigma, int unbiased) {
	for (size_t v = 0; v < nvox3d; v++) {
		const double s = sigma[v];
		if (live[v] && (!mask || mask[v]))
			for (int j = 0; j < nvol; j++) {
				float *o = out + v + (size_t)j * nvox3d;
				*o = (float)(s * dn_vst_inverse(vt, *o, unbiased));
			}
		if (noise && noise[v] != 0.0f) noise[v] = (float)s;
	}
}

// Index of `v` in the NULL-terminated `names`, into *out; the index is the option's
// enum value.  Returns non-zero, reported, for anything else.
static int parse_choice(const char *flag, const char *v, const char *const *names, int *out) {
	for (int i = 0; names[i]; i++)
		if (!strcmp(v, names[i])) { *out = i; return 0; }
	dn_err("%s must be one of:", flag);
	for (int i = 0; names[i]; i++) fprintf(stderr, " %s", names[i]);
	fprintf(stderr, " (got '%s')\n", v);
	return 1;
}

int main(int argc, char *argv[]) {
	const char *fin = NULL, *fout = NULL;
	const char *fmask = NULL, *fnoise = NULL, *frank = NULL;
	const char *fphase = NULL, *freal = NULL, *fbval = NULL;
	// Off by default: measured neutral under the hard threshold (RMSE -0.4% at
	// sigma 10, +1.3% at sigma 30, synthetic truth on benchmark/).
	int demean = 0;
	// Defaults chosen by measurement against a synthetic truth (AGENTS.md dead ends):
	// each beat the original exclusive / optthresh / cube pipeline, which
	// `-filter optthresh -aggregator exclusive -shape cube -vst n` still reproduces
	// byte for byte.
	int filter = DN_FILTER_OPTSHRINK;
	int shape = DN_SHAPE_SPHERE;
	int aggregator = DN_AGG_GAUSSIAN, stride = 0;   // 0: auto
	double fwhm = 2.0;
	int fwhm_given = 0;
	int vst = -1, ncoil = 1, ncoil_given = 0, keep_bias = 0;   // -1: auto
	// Options that only the denoiser reads, counted so -degibbs o can refuse them.
	int denoise_opts = 0;
	static const char *const yes_no[] = {"n", "y", NULL};
	static const char *const filters[] = {"optthresh", "optshrink", "truncate", NULL};   // DN_FILTER_*
	static const char *const shapes[] = {"cube", "sphere", NULL};                        // DN_SHAPE_*
	static const char *const aggregators[] = {"exclusive", "gaussian", "uniform", NULL}; // DN_AGG_*
	dn_phase_units punits = DN_PHASE_AUTO;
	// Presence, not value: "-phaseunits auto" leaves punits at its default, so
	// testing the enum would let that one spelling slip past the check below
	// while every other spelling was rejected.
	int punits_given = 0;
	// extent_given, rather than `extent != 0`, is what separates "not supplied"
	// from "supplied as 0".  Sharing 0 for both meant an explicit `-extent 0`
	// silently selected the automatic size and reported it as "(auto)", while
	// every other value the help text calls invalid -- 2, -1 -- was rejected.
	int extent = 0, extent_given = 0, nthreads = 0, quiet = 0;
	int degibbs = DN_DG_NO;
#ifdef DN_DEGIBBS
	// pf_given, not `pf != DG_PF_FULL`, is what separates "not supplied" from
	// "supplied as 1.0" -- the same distinction -extent and -phaseunits draw, and
	// without it an explicit `-pF 1.0` was the one factor accepted with no
	// -degibbs to act on it.
	double pf = DG_PF_FULL;   // symmetric k-space
	int pf_given = 0;
#endif

	for (int i = 1; i < argc; i++) {
		const char *a = argv[i];
		if (!strcmp(a, "-help") || !strcmp(a, "--help") || !strcmp(a, "-h")) {
			usage();
			return EXIT_SUCCESS;
		}
		if (!strcmp(a, "-version") || !strcmp(a, "--version")) {
			printf("svht_denoise %s\n", DN_VERSION);
			return EXIT_SUCCESS;
		}
		if (!strcmp(a, "-quiet")) { quiet = 1; continue; }
		if (!strcmp(a, "-preserve_noise_bias")) { keep_bias = 1; denoise_opts++; continue; }

		if (a[0] == '-' && a[1] != '\0') {
			// Every remaining flag takes exactly one argument.
			if (i + 1 >= argc) {
				dn_err("option '%s' needs a value\n", a);
				return EXIT_FAILURE;
			}
			const char *v = argv[++i];
			if (!strcmp(a, "-mask")) fmask = v;
			else if (!strcmp(a, "-noise")) fnoise = v;
			else if (!strcmp(a, "-rank")) frank = v;
			else if (!strcmp(a, "-phase")) fphase = v;
			else if (!strcmp(a, "-real")) freal = v;
			else if (!strcmp(a, "-bval")) { fbval = v; denoise_opts++; }
			else if (!strcmp(a, "-vst") || !strcmp(a, "-demean") || !strcmp(a, "-filter") ||
			         !strcmp(a, "-shape") || !strcmp(a, "-aggregator")) {
				denoise_opts++;
				int bad;
				if (!strcmp(a, "-vst")) bad = parse_choice(a, v, yes_no, &vst);
				else if (!strcmp(a, "-demean")) bad = parse_choice(a, v, yes_no, &demean);
				else if (!strcmp(a, "-filter")) bad = parse_choice(a, v, filters, &filter);
				else if (!strcmp(a, "-shape")) bad = parse_choice(a, v, shapes, &shape);
				else bad = parse_choice(a, v, aggregators, &aggregator);
				if (bad) return EXIT_FAILURE;
			}
			else if (!strcmp(a, "-noise_dof")) {
				if (parse_int(v, &ncoil) || ncoil < 1 || ncoil > 64) {
					dn_err("-noise_dof needs a channel count from 1 to 64 (got '%s')\n", v);
					return EXIT_FAILURE;
				}
				ncoil_given = 1;
				denoise_opts++;
			}
			else if (!strcmp(a, "-aggregator_fwhm")) {
				char *pend = NULL;
				fwhm = strtod(v, &pend);
				if (*pend != '\0' || pend == v || !(fwhm > 0.0) || !isfinite(fwhm)) {
					dn_err("-aggregator_fwhm needs a positive number (got '%s')\n", v);
					return EXIT_FAILURE;
				}
				fwhm_given = 1;
				denoise_opts++;
			}
			else if (!strcmp(a, "-stride")) {
				if (parse_int(v, &stride) || stride < 1 || stride > 2) {
					dn_err("-stride must be 1 or 2 (got '%s')\n", v);
					return EXIT_FAILURE;
				}
				denoise_opts++;
			}
			else if (!strcmp(a, "-phaseunits")) {
				punits_given = 1;
				if (!strcmp(v, "auto")) punits = DN_PHASE_AUTO;
				else if (!strcmp(v, "radians")) punits = DN_PHASE_RADIANS;
				else if (!strcmp(v, "degrees")) punits = DN_PHASE_DEGREES;
				else if (!strcmp(v, "turns") || !strcmp(v, "cycles")) punits = DN_PHASE_TURNS;
				else {
					dn_err("-phaseunits must be radians, degrees, turns/cycles or auto (got '%s')\n", v);
					return EXIT_FAILURE;
				}
			}
#ifdef DN_DEGIBBS
			else if (!strcmp(a, "-degibbs")) {
				if (!strcmp(v, "y")) degibbs = DN_DG_YES;
				else if (!strcmp(v, "n")) degibbs = DN_DG_NO;
				else if (!strcmp(v, "o")) degibbs = DN_DG_ONLY;
				else {
					dn_err("-degibbs must be y, n or o (got '%s')\n", v);
					return EXIT_FAILURE;
				}
			}
			else if (!strcmp(a, "-pF")) {
				char *pend = NULL;
				pf = strtod(v, &pend);
				// Only the PARSE is checked here.  Which factors exist is
				// dn_degibbs_check's to say, and it says it precisely; a range
				// test here told a user that 0.9 was acceptable, then a second
				// message refused it.
				if (*pend != '\0' || pend == v) {
					dn_err("-pF needs a partial-Fourier factor (got '%s')\n", v);
					return EXIT_FAILURE;
				}
				pf_given = 1;
			}
#else
			// One refusal for both, since the message is the same and `a` names
			// whichever was given.
			else if (!strcmp(a, "-degibbs") || !strcmp(a, "-pF")) {
				dn_err("%s is not compiled into this build.\n", a);
				dn_err("  Rebuild with 'make DEGIBBS=1'.\n");
				return EXIT_FAILURE;
			}
#endif
			else if (!strcmp(a, "-extent")) {
				if (parse_int(v, &extent)) { dn_err("-extent needs an integer (got '%s')\n", v); return EXIT_FAILURE; }
				extent_given = 1;
			} else if (!strcmp(a, "-nthreads") || !strcmp(a, "-p")) {
				if (parse_int(v, &nthreads) || nthreads < 1) {
					dn_err("-nthreads needs a positive integer (got '%s')\n", v);
					return EXIT_FAILURE;
				}
			} else {
				dn_err("unknown option '%s'\n", a);
				dn_err("  run '%s -help' for the supported options\n", DN_NAME);
				return EXIT_FAILURE;
			}
			continue;
		}

		if (!fin) fin = a;
		else if (!fout) fout = a;
		else {
			dn_err("unexpected extra argument '%s'\n", a);
			return EXIT_FAILURE;
		}
	}

	if (!fin || !fout) {
		dn_err("missing required arguments.\n");
		dn_err("  usage: svht_denoise <input> <output> [options];  -help for details\n");
		return EXIT_FAILURE;
	}
	// -real names the rotated input, which only exists when there is a rotation.
	// Silently writing nothing, or writing an unrotated copy, would both be worse
	// than saying so.
	if (freal && !fphase) {
		dn_err("-real writes the real-axis-rotated input, which needs -phase.\n");
		return EXIT_FAILURE;
	}
	// "-" means stdin for an INPUT.  As an output it made nifti write the image
	// to stdout, which contradicts dn.h's stated invariant that nothing here
	// writes an image to stdout, and the collision check models it as the file
	// "-.nii" -- so the checker and the writer disagreed about where it goes.
	// Undocumented either way; rejecting keeps the invariant true.
	const char *sv_out[4] = {fout, fnoise, frank, freal};
	const char *sv_lbl[4] = {"output", "-noise", "-rank", "-real"};
	for (int i = 0; i < 4; i++) {
		if (sv_out[i] && sv_out[i][0] == '-' && sv_out[i][1] == '\0') {
			dn_err("\"-\" is an input (stdin), not an output; %s needs a filename.\n", sv_lbl[i]);
			return EXIT_FAILURE;
		}
	}
	// -degibbs o does no denoising, so everything describing the denoiser -- its
	// mask, its by-products, its patch size, and the phase rotation that exists
	// only to change the noise the denoiser sees -- has nothing to act on.
	// Ignoring them silently would give a run that looked fine and wrote fewer
	// files than were asked for.
	if (degibbs == DN_DG_ONLY && (fmask || fnoise || frank || fphase || freal || extent_given ||
	                              denoise_opts)) {
		dn_err("-degibbs o does no denoising, so none of the denoiser's options have\n");
		dn_err("  anything to act on.\n");
		return EXIT_FAILURE;
	}
	// A masked run leaves hard zeros outside the mask, and a hard zero edge is
	// precisely the discontinuity the Kellner method rings on.  Degibbsing that
	// both breaks the "outside the mask reads as zero" promise -- measured, 647
	// of 648 outside voxels came back non-zero, peaking at a quarter of the data
	// range -- and rings the mask boundary INWARDS, corrupting voxels that are
	// inside it.  Neither is recoverable by re-zeroing afterwards, so refuse the
	// combination rather than quietly returning something worse than either
	// stage alone.  `-degibbs o` already refuses -mask, for its own reason.
	if (degibbs == DN_DG_YES && fmask) {
		dn_err("-mask cannot be combined with -degibbs y.\n");
		dn_err("  Masking leaves hard zeros outside the mask, and ringing removal would\n");
		dn_err("  treat that edge as signal -- corrupting voxels inside the mask too.\n");
		dn_err("  Denoise with -mask, then run -degibbs o on the result if you need both.\n");
		return EXIT_FAILURE;
	}
#ifdef DN_DEGIBBS
	// -pF changes how ringing is removed, so it means nothing without a stage
	// that removes it.  Silently ignoring it would leave PF ringing in place
	// while the command line says otherwise.
	if (pf_given && degibbs == DN_DG_NO) {
		dn_err("-pF describes how -degibbs should work, but -degibbs is not enabled.\n");
		return EXIT_FAILURE;
	}
#endif
	// Auto: stride 2 under averaging, 1 when only centres are kept.  An automatic
	// VST is decided once the input is read (see below).
	if (!stride) stride = aggregator == DN_AGG_EXCLUSIVE ? 1 : 2;
	if (vst == 1 && fphase) {
		dn_err("-vst stabilises magnitude noise; -phase data are rotated to real and are\n");
		dn_err("  already Gaussian.\n");
		return EXIT_FAILURE;
	}
	if (fwhm_given && aggregator != DN_AGG_GAUSSIAN) {
		dn_err("-aggregator_fwhm sets the Gaussian width, which needs -aggregator gaussian.\n");
		return EXIT_FAILURE;
	}
	if (stride > 1 && aggregator == DN_AGG_EXCLUSIVE) {
		dn_err("-stride %d leaves voxels with no patch centred on them; it needs\n", stride);
		dn_err("  -aggregator gaussian or uniform.\n");
		return EXIT_FAILURE;
	}
	if (fbval && !demean) {
		dn_err("-bval is only used by -demean y.\n");
		return EXIT_FAILURE;
	}
	if (punits_given && !fphase) {
		dn_err("-phaseunits describes the -phase image, which was not given.\n");
		return EXIT_FAILURE;
	}
	int status = EXIT_FAILURE;
	dn_image img;
	uint8_t *mask = NULL, *live = NULL;
	float *out = NULL, *noise = NULL, *means = NULL, *sigma_map = NULL, *unit_map = NULL;
	dn_vst *vt = NULL;
	int need_map = 0;
	int *group = NULL;
	double *shell_b = NULL;
	int *shell_n = NULL;
	char *bval_auto = NULL;
	uint16_t *rank = NULL;

	// Not require4d in "only" mode: mrdegibbs accepts a 3D image, and the
	// >= 2 volume rule belongs to the denoiser, which is not running.
	if (dn_image_read(fin, "input", degibbs != DN_DG_ONLY, &img)) return EXIT_FAILURE;

	// After the read, not before: resolving an output prefix needs the input's
	// nifti_type.  Still ahead of every write, and ahead of the denoising itself.
	// The b-values -demean y will read, explicit or the input's sidecar, so the
	// collision check below covers them: an output named over them would replace
	// the text it read at exit 0.
	if (demean && !fbval) {
		if (!strcmp(fin, "-")) {
			dn_err("-demean y on stdin needs -bval: there is no input name to find it by.\n");
			goto done;
		}
		bval_auto = dn_bval_sidecar(fin);
		if (!bval_auto) goto done;
		fbval = bval_auto;
	}
	dn_path paths[] = {
		{"input",  fin,    1, NULL, NULL},
		{"-bval",  fbval,  2, NULL, NULL},
		{"-mask",  fmask,  1, NULL, NULL},
		{"-phase", fphase, 1, NULL, NULL},
		{"output", fout,   0, NULL, NULL},
		{"-noise", fnoise, 0, NULL, NULL},
		{"-rank",  frank,  0, NULL, NULL},
		{"-real",  freal,  0, NULL, NULL},
	};
	if (check_path_collisions(paths, (int)(sizeof paths / sizeof paths[0]),
	                          img.nifti_type)) goto done;

#ifdef DN_DEGIBBS
	// Before any work, not after: this needs only the header, and reaching it
	// from the -degibbs y call site meant a full denoise ran and was then thrown
	// away when the geometry turned out to be unusable.
	if (degibbs != DN_DG_NO && dn_degibbs_check(img.nx, img.ny, pf)) goto done;
#endif

#ifdef DN_DEGIBBS
	// "Only" short-circuits everything the denoiser needs -- geometry, mask,
	// scratch budget -- and rewrites the input buffer in place, so there is no
	// second copy of the series.
	if (degibbs == DN_DG_ONLY) {
		if (nthreads < 1) nthreads = dn_default_threads();
		// Report what will really run, not what was asked for -- same contract the
		// denoising path has, and it was missing here.
		nthreads = dn_degibbs_threads(img.nx, img.ny, img.nz, img.nvol, nthreads, pf);
		if (!quiet) {
			dn_err("input        : %s (%dx%dx%d, %d volume%s)\n", fin,
			        img.nx, img.ny, img.nz, img.nvol, img.nvol == 1 ? "" : "s");
			dn_err("degibbs      : only, x-y planes (no denoising)%s\n", pf_note(pf));
			dn_err("threads      : %d\n", nthreads);
		}
		if (dn_degibbs(img.data, img.nx, img.ny, img.nz, img.nvol, nthreads, pf)) goto done;
		if (dn_write_f32(&img, fout, img.data, img.nvol)) goto done;
		status = EXIT_SUCCESS;
		goto done;
	}
#endif

	// A magnitude image is never negative, so a negative voxel means signed data
	// -- typically a series already rotated to real elsewhere -- whose noise is
	// Gaussian, and stabilising it as Rician would bias it.  Only the automatic
	// choice yields to this; an explicit -vst y is obeyed.
	// -phase also says the input is not magnitude.
	int vst_off_signed = 0;
	if (vst < 0) {
		vst = !fphase;
		for (size_t i = 0; vst && i < img.nvox3d * (size_t)img.nvol; i++)
			if (img.data[i] < 0.0f) { vst = 0; vst_off_signed = 1; }
	}
	if ((ncoil_given || keep_bias) && !vst) {
		dn_err("-noise_dof and -preserve_noise_bias describe -vst y, which is %s.\n",
		       vst_off_signed ? "off because the input has negative values" : "not enabled");
		goto done;
	}

	const int extent_was_auto = !extent_given;
	if (!extent_given) extent = dn_auto_extent(img.nvol);

	int ngroups = 0;
	if (demean) {
		group = (int *)dn_malloc((size_t)img.nvol, sizeof(int));
		shell_b = (double *)dn_malloc((size_t)img.nvol, sizeof(double));
		shell_n = (int *)dn_malloc((size_t)img.nvol, sizeof(int));
		if (!group || !shell_b || !shell_n) goto done;
		ngroups = dn_bval_groups(fbval, img.nvol, group, shell_b, shell_n);
		if (ngroups < 0) goto done;
	}
	// The noise level is a MEDIAN singular value, which needs a noise bulk: with
	// two columns left after demeaning, the "median" averages signal and noise and
	// read 5.8 against a true 1, and the VST then zeroed 80% of the output.  A
	// noise map from such data is meaningless, so -noise is refused; otherwise the
	// hard threshold, which uses no sigma, takes over rather than refuse a run the
	// minimum input (2 volumes) is allowed.
	need_map = filter != DN_FILTER_OPTTHRESH || vst;
	if (img.nvol - ngroups < 3) {
		if (fnoise) {
			dn_err("-noise needs at least 3 noise columns (volumes minus demeaned shells);\n");
			dn_err("  this run has %d, from which no noise level can be estimated.\n", img.nvol - ngroups);
			goto done;
		}
		if (need_map) {
			dn_err("note: %d noise column%s cannot give a noise level, so this run uses\n",
			       img.nvol - ngroups, img.nvol - ngroups == 1 ? "" : "s");
			dn_err("  -filter optthresh without -vst.\n");
			filter = DN_FILTER_OPTTHRESH;
			vst = 0;
			need_map = 0;
		}
	}

	dn_geom g, g1;
	// A spacing the header cannot vouch for is taken as isotropic: non-positive,
	// non-finite, or axes more than 1000x apart, which no acquisition has and which
	// would size the sphere's neighbour box from a nonsense ratio.
	double sp[3] = {img.nim->dx, img.nim->dy, img.nim->dz};
	int sp_ok = 1;
	for (int a = 0; a < 3; a++)
		if (!(sp[a] > 0.0) || !isfinite(sp[a])) sp_ok = 0;
	if (sp_ok && fmax(fmax(sp[0], sp[1]), sp[2]) > 1000.0 * fmin(fmin(sp[0], sp[1]), sp[2])) {
		dn_err("note: voxel spacing %g x %g x %g is implausible; treating it as isotropic\n",
		       sp[0], sp[1], sp[2]);
		sp_ok = 0;
	}
	if (!sp_ok) sp[0] = sp[1] = sp[2] = 1.0;
	if (dn_geom_init(&g, shape, img.nx, img.ny, img.nz, img.nvol, sp[0], sp[1], sp[2], extent, ngroups))
		goto done;
	if (need_map) {
		// A cube must fit the image; a sphere need not.
		int cap = img.nx < img.ny ? img.nx : img.ny;
		if (img.nz < cap) cap = img.nz;
		// A sphere need not fit inside the image, but its voxels must: half the image
		// leaves room for the whole-shell overshoot.  Never below the main patch.
		if (shape == DN_SHAPE_SPHERE)
			for (cap = 3; (double)(cap + 2) * (cap + 2) * (cap + 2) <= 0.5 * (double)img.nvox3d; cap += 2) ;
		const int k1 = dn_sigma_extent(img.nvol, extent, cap);
		if (dn_geom_init(&g1, shape, img.nx, img.ny, img.nz, img.nvol, sp[0], sp[1], sp[2], k1, ngroups))
			goto done;
	}
	g.filter = filter;

	size_t n_in_mask = img.nvox3d;
	if (fmask) {
		mask = dn_mask_build(&img, fmask, &n_in_mask);
		if (!mask) goto done;
	}

	if (nthreads < 1) nthreads = dn_default_threads();
	// The REQUEST, kept separate from the denoiser's effective count below.
	// -degibbs has its own cap and none of the denoiser's constraints -- no mask,
	// no patch arena -- so handing it the throttled number would let a small mask
	// or the eigensolver's scratch budget starve a stage limited by neither.  The
	// sigma pass has its own geometry and visits no mask, so the same applies.
	const int nthreads_req = nthreads;
	// Report what will actually run, not what was asked for.  Averaging visits
	// patch centres in tiles, not masked voxels, so it sizes its own team.
	nthreads = aggregator == DN_AGG_EXCLUSIVE ? dn_effective_threads(&g, n_in_mask, nthreads)
	                                          : dn_agg_threads(&g, stride, nthreads);

	if (!quiet) {
		dn_err("input        : %s (%dx%dx%d, %d volumes)\n",
		        fin, img.nx, img.ny, img.nz, img.nvol);
		if (shape == DN_SHAPE_SPHERE)
			dn_err("patch        : sphere, radius %.3g mm = %d voxels x %d volumes\n",
			        sqrt(g.r2), g.m, g.nvol);
		else
			dn_err("patch        : %dx%dx%d = %d voxels x %d volumes%s\n",
			        g.extent, g.extent, g.extent, g.m, g.nvol,
			        extent_was_auto ? " (auto)" : "");
		if (ngroups) {
			dn_err("demean       : %d shell%s from %s:", ngroups, ngroups == 1 ? "" : "s", fbval);
			for (int s = 0; s < ngroups; s++)
				fprintf(stderr, " b=%.0f x%d", shell_b[s], shell_n[s]);
			fprintf(stderr, "\n");
		} else if (demean)
			dn_err("demean       : none (no shell of 2+ volumes in %s)\n", fbval);
		dn_err("filter       : %s\n", filters[filter]);
		if (need_map)
			dn_err("sigma map    : from %d-voxel patches on every %dth voxel\n", g1.m, DN_SIGMA_STRIDE);
		if (vst_off_signed)
			dn_err("vst          : off, the input has negative values so is not magnitude\n");
		if (vst)
			dn_err("vst          : non-central chi, %d channel%s%s, %s inverse\n", ncoil,
			        ncoil == 1 ? "" : "s", ncoil == 1 ? " (Rician)" : "",
			        keep_bias ? "algebraic" : "exact-unbiased");
		if (aggregator != DN_AGG_EXCLUSIVE)
			dn_err("aggregator   : %s, patch centred on every %svoxel\n",
			        aggregators[aggregator],
			        stride == 1 ? "" : "2nd ");
		dn_err("beta         : %.9f\n", g.beta);
		dn_err("omega(beta)  : %.9f\n", g.omega);
		dn_err("threads      : %d\n", nthreads);
		if (mask)
			dn_err("mask         : %s (%zu of %zu voxels)\n", fmask, n_in_mask, img.nvox3d);
#ifdef DN_DEGIBBS
		// Its OWN effective count, not the denoiser's: degibbs caps by planes and
		// by its own scratch budget, so the two stages can legitimately run
		// different team sizes and one number would be wrong for one of them.
		if (degibbs == DN_DG_YES)
			dn_err("degibbs      : yes, x-y planes%s (%d threads)\n", pf_note(pf),
			        dn_degibbs_threads(img.nx, img.ny, img.nz, img.nvol, nthreads_req, pf));
#endif
	}


	// Before anything is allocated for the denoising, and before the first write:
	// on failure this leaves img exactly as it was read, and the run stops with
	// nothing on disk.  Afterwards img.data is the real-valued rotated series and
	// every stage below is unchanged -- the denoiser neither knows nor cares that
	// its input is now signed.
	if (fphase && dn_rotate_to_real(&img, fphase, punits, quiet)) goto done;

#ifdef DN_DEGIBBS
	// AFTER the rotation, so it describes a run that got this far, and OUTSIDE
	// any -quiet test: -quiet suppresses the run SUMMARY, and dn_phase.c's
	// whole-turn warning already establishes that a warning about the data is not
	// a summary line.  The two options pull opposite ways -- -phase keeps
	// background noise zero-mean and degibbs truncates it -- and a scripted
	// -quiet run is the case that most needs telling.  One dn_err per line: it
	// prefixes per CALL.
	// Any signed input, not only -phase: a series rotated to real elsewhere is
	// truncated just the same, and once did so silently.
	int signed_input = fphase || vst_off_signed;
	for (size_t i = 0; degibbs == DN_DG_YES && !signed_input && i < img.nvox3d * (size_t)img.nvol; i++)
		signed_input = img.data[i] < 0.0f;
	if (degibbs == DN_DG_YES && signed_input) {
		dn_err("warning: %s, and -degibbs truncates it to zero,\n",
		       fphase ? "-phase output is signed" : "the input has negative values");
		dn_err("  reinstating a positive bias in background voxels.  Use -degibbs n if\n");
		dn_err("  the denoised series needs to stay signed.\n");
	}
#endif

	// After the rotation: the means are of the data the denoiser actually sees.
	if (ngroups) {
		means = dn_bval_means(img.data, img.nvox3d, img.nvol, group, ngroups);
		if (!means) goto done;
		g.group = g1.group = group;
		g.mean = g1.mean = means;
	}
	if (need_map) {
		sigma_map = dn_sigma_map(&g1, img.data, nthreads_req);
		if (!sigma_map) goto done;
		g.sigma_map = sigma_map;
	}
	// Stabilise in place, in units of the local sigma, so the denoiser then sees
	// unit-variance Gaussian noise everywhere and reads sigma = 1.
	//
	// The map above was estimated from RAW magnitudes, whose spread shrinks near
	// the noise floor, so it reads low there: 47 against a true 60 on benchmark/
	// at that noise level.  So sigma is re-estimated in the stabilised domain,
	// where it should come out as 1, and corrected by that factor (dwidenoise2
	// iterates for the same reason).  Each round recovers the raw value through
	// f's exact algebraic inverse rather than keeping a second copy of the series.
	if (vst) {
		const int filled = dn_sigma_fill(sigma_map, img.nvox3d);
		if (filled < 0) goto done;
		if (filled) {
			// Noiseless input: nothing to stabilise against, and nothing to denoise.
			dn_err("note: no noise level could be estimated, so the data are not stabilised\n");
			vst = 0;
		}
	}
	if (vst) {
		vt = dn_vst_create(ncoil);
		unit_map = (float *)dn_malloc(img.nvox3d, sizeof(float));
		live = (uint8_t *)dn_calloc(img.nvox3d, 1);
		if (!vt || !unit_map || !live) goto done;
		// A voxel zero in EVERY volume is masked-out background, not magnitude
		// samples, and stays zero so all-zero patches still short-circuit.  A lone
		// zero sample in a live voxel is real data (quantised magnitude can hit 0)
		// and is transformed with the rest of its vector.
		for (int j = 0; j < img.nvol; j++)
			for (size_t v = 0; v < img.nvox3d; v++)
				if (img.data[v + (size_t)j * img.nvox3d] != 0.0f) live[v] = 1;
		for (int it = 0; it < DN_VST_ROUNDS; it++) {
			// unit_map holds the sigma each voxel is in units of, until the end.
			vst_stabilise(vt, img.data, img.nvox3d, img.nvol, live, sigma_map, unit_map, it == 0);
			if (ngroups) {
				free(means);
				means = dn_bval_means(img.data, img.nvox3d, img.nvol, group, ngroups);
				if (!means) goto done;
				g.mean = g1.mean = means;
			}
			if (it == DN_VST_ROUNDS - 1) break;
			float *ratio = dn_sigma_map(&g1, img.data, nthreads_req);
			if (!ratio) goto done;
			for (size_t v = 0; v < img.nvox3d; v++)
				if (ratio[v] > 0.0f) sigma_map[v] *= ratio[v];
			free(ratio);
		}
		for (size_t v = 0; v < img.nvox3d; v++) unit_map[v] = 1.0f;
		g.sigma_map = unit_map;
	}

	// Zero-initialised: voxels outside the mask are never visited and must read
	// as zero in every output.
	out = (float *)dn_calloc(img.nvox3d * (size_t)img.nvol, sizeof(float));
	if (!out) goto done;
	if (fnoise) {
		noise = (float *)dn_calloc(img.nvox3d, sizeof(float));
		if (!noise) goto done;
	}
	if (frank) {
		rank = (uint16_t *)dn_calloc(img.nvox3d, sizeof(uint16_t));
		if (!rank) goto done;
	}

	dn_run r;
	memset(&r, 0, sizeof(r));
	r.g = &g;
	r.img = img.data;
	r.mask = mask;
	r.out = out;
	r.noise = noise;
	r.rank = rank;
	r.n_work = n_in_mask;
	r.nthreads = nthreads;
	r.aggregator = aggregator;
	r.fwhm = fwhm;
	r.stride = stride;

	if (aggregator == DN_AGG_EXCLUSIVE ? dn_run_execute(&r) : dn_agg_execute(&r)) goto done;

	if (vst) {
		vst_restore(vt, out, noise, img.nvox3d, img.nvol, live, mask, sigma_map, !keep_bias);
		// The unbiased inverse returns 0 for anything at or below the noise floor,
		// which is right for the model given -- and a mis-stated -noise_dof puts all
		// of the data there (measured: -noise_dof 4 on Rician data zeroed every
		// sample, at exit 0).  Say so rather than hand back an empty image.
		size_t nlive = 0, nzero = 0;
		for (size_t v = 0; v < img.nvox3d; v++) {
			if (!live[v] || (mask && !mask[v])) continue;
			for (int j = 0; j < img.nvol; j++) {
				nlive++;
				nzero += out[v + (size_t)j * img.nvox3d] == 0.0f;
			}
		}
		if (nzero * 2 > nlive) {
			dn_err("warning: %.0f%% of the output is zero, i.e. at or below the noise floor.\n",
			       100.0 * nzero / nlive);
			dn_err("  Check -noise_dof (%d), or use -preserve_noise_bias or -vst n.\n", ncoil);
		}
	}

#ifdef DN_DEGIBBS
	// After the denoiser, never before: degibbsing first would alter the noise
	// structure the denoiser models.  -noise and -rank describe the denoising and
	// are untouched, as is -real, which is the input before either stage.
	if (degibbs == DN_DG_YES &&
	    dn_degibbs(out, img.nx, img.ny, img.nz, img.nvol, nthreads_req, pf)) goto done;
#endif

	// Write the auxiliary maps first: if the disk fills, failing before the main
	// output is written is less confusing than leaving a complete-looking result
	// beside a missing noise map.
	if (noise && dn_write_f32(&img, fnoise, noise, 1)) goto done;
	if (rank && dn_write_u16(&img, frank, rank)) goto done;
	// img.data is the ROTATED input: dn_run_execute only reads it.
	if (freal && dn_write_f32(&img, freal, img.data, img.nvol)) goto done;
	if (dn_write_f32(&img, fout, out, img.nvol)) goto done;

	status = EXIT_SUCCESS;

done:
	free(out);
	free(noise);
	free(means);
	free(sigma_map);
	free(unit_map);
	dn_vst_free(vt);
	free(group);
	free(shell_b);
	free(shell_n);
	free(bval_auto);
	free(rank);
	free(mask);
	free(live);
	dn_image_free(&img);
	return status;
}
