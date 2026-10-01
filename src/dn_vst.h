// Variance-stabilising transform for magnitude data (-vst).

#ifndef DN_VST_H
#define DN_VST_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dn_vst dn_vst;

// Tables for non-central chi noise from `ncoil` receive channels (1 = Rician),
// in units of sigma.  NULL on allocation failure.
dn_vst *dn_vst_create(int ncoil);
void dn_vst_free(dn_vst *t);

// Magnitude x/sigma -> stabilised value, of unit noise variance at any SNR.
double dn_vst_forward(const dn_vst *t, double x);

// Stabilised value -> signal/sigma.  unbiased: the exact-unbiased inverse,
// returning the noise-free signal nu/sigma; otherwise the algebraic inverse,
// returning a magnitude/sigma that keeps the noise-floor bias.
double dn_vst_inverse(const dn_vst *t, double d, int unbiased);

#ifdef __cplusplus
}
#endif

#endif // DN_VST_H
