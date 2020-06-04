#pragma once

#include <enoki/special.h>
#include <mitsuba/core/frame.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/quad.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fresnel.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <numeric>

NAMESPACE_BEGIN(mitsuba)

static const int p_max = 3;
static const float sqrt_pi_over_8 = 0.626657069f;

template <typename Float, typename Spectrum>
class HairBSDF final : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, m_flags, m_components)
    MTS_IMPORT_TYPES(Texture)

    using UInt = uint_array_t<Float>;

    Float Pi = math::Pi<Float>;

    HairBSDF(const Properties &props);

    enum Mode_sigma_a { Absorption, Reflectance, Concentration};

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext&,
                                             const SurfaceInteraction3f&,
                                             Float, 
                                             const Point2f&, 
                                             Mask) const override;

    Spectrum eval(const BSDFContext&, const SurfaceInteraction3f&, const Vector3f&, Mask) const override;

    Float pdf(const BSDFContext&, const SurfaceInteraction3f&, const Vector3f&, Mask) const override;

    std::string to_string() const override;

private:
    Float h_xml;
    Float eta;
    Mode_sigma_a mode;
    ref<Texture> sigma_a;
    ref<Texture> sigma_a_reflectance;
    Float ce, cp;

    Float beta_m, beta_n;
    Float v[p_max + 1];
    Float s;
    Float sin_2k_alpha[3], cos_2k_alpha[3];

    MTS_INLINE Float sqr(Float v) const { return v * v; }

    template <int n>
    Float pow(Float v) const{
        static_assert(n > 0, "Power can't be negative");
        Float n2 = pow<n / 2>(v);
        return n2 * n2 * pow<n & 1>(v);
    }

    template <>
    MTS_INLINE Float pow<1>(Float v) const{
        return v;
    }

    template <>
    MTS_INLINE Float pow<0>(__attribute__((unused)) Float v) const{
        return 1;
    }

    UInt32 compact_1_by_1(UInt32 x) const{
        // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
        x &= 0x55555555;
        // x = --fe --dc --ba --98 --76 --54 --32 --10
        x = (x ^ (x >> 1)) & 0x33333333;
        // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
        x = (x ^ (x >> 2)) & 0x0f0f0f0f;
        // x = ---- ---- fedc ba98 ---- ---- 7654 3210
        x = (x ^ (x >> 4)) & 0x00ff00ff;
        // x = ---- ---- ---- ---- fedc ba98 7654 3210
        x = (x ^ (x >> 8)) & 0x0000ffff;
        return x;
    }

    Point2f demux_float(Float f) const{
        UInt64 v =  f * (1ull << 32);
        UInt32 bits[2] = {compact_1_by_1(v), compact_1_by_1(v >> 1)};
        return {bits[0] / Float(1 << 16), bits[1] / Float(1 << 16)};
    }

    std::array<Spectrum, p_max + 1> Ap(Float cos_theta_i, Float eta, Float h, const Spectrum &T) const{
        std::array<Spectrum, p_max + 1> ap;

        // Compute $p=0$ attenuation at initial cylinder intersection
        Float cos_gamma_o = safe_sqrt(1 - h * h);
        Float cos_theta = cos_theta_i * cos_gamma_o;
        auto res = fresnel(cos_theta, eta); //F, cos_theta_t, eta_it, eta_ti
        Float f = std::get<0>(res);
        ap[0] = f;

        // Compute $p=1$ attenuation term
        ap[1] = sqr(1 - f) * T;

        // Compute attenuation terms up to $p=_pMax_$
        for (int p = 2; p < p_max; ++p) ap[p] = ap[p - 1] * T * f;

        // Compute attenuation term accounting for remaining orders of scattering
        ap[p_max] = ap[p_max - 1] * f * T / (Spectrum(1.f) - T * f);
        //std::cout << "End ap" << std::endl;
        return ap;
    }

    std::array<Float, p_max + 1> compute_ap_pdf(Float cos_theta_i, Float h, const SurfaceInteraction3f &si, Mask active) const {
        // Compute array of $A_p$ values for _cosThetaO_
        Float sin_theta_i = safe_sqrt(1 - cos_theta_i * cos_theta_i);

        // Compute $\cos \thetat$ for refracted ray
        Float sin_theta_t = sin_theta_i / eta;
        Float cos_theta_t = safe_sqrt(1 - sqr(sin_theta_t));

        // Compute $\gammat$ for refracted ray
        Float etap = sqrt(eta * eta - sqr(sin_theta_i)) / cos_theta_i;
        Float sin_gamma_t = h / etap;
        Float cos_gamma_t = safe_sqrt(1 - sqr(sin_gamma_t));

        // Compute the transmittance _T_ of a single path through the cylinder
        Spectrum T = exp(-evaluate_sigma_a(si, active) * (2.0f * cos_gamma_t / cos_theta_t));

        std::array<Spectrum, p_max + 1> ap = Ap(cos_theta_i, eta, h, T);

        // Compute $A_p$ PDF from individual $A_p$ terms
        std::array<Float, p_max + 1> ap_pdf;
        Float sum_y =
            std::accumulate(ap.begin(), ap.end(), Float(0),
                            [](Float s, const Spectrum &ap) { return s + ap.y(); });
        for (int i = 0; i <= p_max; ++i) ap_pdf[i] = ap[i].y() / sum_y;
        return ap_pdf;
    }

    Float abs_cos_theta(const Vector3f &w) const{
        return abs(Frame3f::cos_theta(w));
    }

    inline Float radians(Float deg) const{
        return (math::Pi<ScalarFloat>/180.f) * deg;
    }

    void tilt_scales(Float sin_theta_i, Float cos_theta_i, UInt p, Float &sin_theta_op, Float &cos_theta_op) const{
        Mask mask = p == 0, mask1 = p == 1, mask2 = p == 2;

        masked(sin_theta_op, mask) = sin_theta_i * cos_2k_alpha[1] - cos_theta_i * sin_2k_alpha[1];
        masked(cos_theta_op, mask) = cos_theta_i * cos_2k_alpha[1] + sin_theta_i * sin_2k_alpha[1];

        masked(sin_theta_op, mask1) =  sin_theta_i * cos_2k_alpha[0] + cos_theta_i * sin_2k_alpha[0];
        masked(cos_theta_op, mask1) = cos_theta_i * cos_2k_alpha[0] - sin_theta_i * sin_2k_alpha[0];

        masked(sin_theta_op, mask2) = sin_theta_i * cos_2k_alpha[2] + cos_theta_i * sin_2k_alpha[2];
        masked(cos_theta_op, mask2) = cos_theta_i * cos_2k_alpha[2] - sin_theta_i * sin_2k_alpha[2];

        masked(sin_theta_op, !mask && !mask1 && !mask2) = sin_theta_i;
        masked(cos_theta_op, !mask && !mask1 && !mask2) = cos_theta_i;
    }

    void get_angles(const Vector3f &w, Float &sin_theta, Float &cos_theta, Float &phi) const{
        sin_theta = w.x();
        cos_theta = sqrt(1 - sqr(sin_theta));
        phi = atan2(w.z(), w.y());
    }

    void print_basis(Vector3f w) const{
        Float sin_theta, cos_theta, phi;
        get_angles(w, sin_theta, cos_theta, phi);
        
        std::cout << "sin_theta: " << sin_theta << std::endl;
        std::cout << "cos_theta: " << cos_theta << std::endl;
        std::cout << "phi: " << phi << std::endl;
    }

    Spectrum sigma_a_from_reflectance(const Spectrum &c, Float beta_n) const{
        Spectrum sigma_a;
        for (size_t i = 0; i < Spectrum::Size; ++i)
            sigma_a[i] = sqr(log(c[i]) /
                             (5.969f - 0.215f * beta_n + 2.532f * sqr(beta_n) -
                              10.73f * pow<3>(beta_n) + 5.574f * pow<4>(beta_n) +
                              0.245f * pow<5>(beta_n)));
        return sigma_a;
    }

    Spectrum sigma_a_from_concentration(Float ce, Float cp) const{
        Float sigma_a[3];
        Float eumelanin_sigma_a[3] = {0.419f, 0.697f, 1.37f};
        Float pheomelanin_sigma_a[3] = {0.187f, 0.4f, 1.05f};
        for (size_t i = 0; i < 3; ++i)
            sigma_a[i] = (ce * eumelanin_sigma_a[i] + cp * pheomelanin_sigma_a[i]);
        return Spectrum(sigma_a[0], sigma_a[1], sigma_a[2]);
    }

    Spectrum evaluate_sigma_a(const SurfaceInteraction3f &si, Mask active) const{
        switch(mode){
            case Absorption: return sigma_a->eval(si, active); break;
            case Reflectance: return sigma_a_from_reflectance(sigma_a->eval(si, active), beta_n); break;
            case Concentration: return sigma_a_from_concentration(ce, cp); break;
        }
    }

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_VARIANT(HairBSDF, BSDF)
MTS_EXPORT_PLUGIN(HairBSDF, "HairBSDF")

NAMESPACE_END(mitsuba)