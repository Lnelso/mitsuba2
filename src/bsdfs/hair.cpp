#include <mitsuba/core/string.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/hair.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)
//Float h, Float eta, const Spectrum &sigma_a, Float beta_m, Float beta_n, Float alpha

template <typename Float, typename Spectrum>
HairBSDF<Float, Spectrum>::HairBSDF(const Properties &props) : Base(props) {
    h = props.float_("h", 0.f);
    Assert(h >= -1 && h <= 1);

    beta_m = props.float_("beta_m", 0.2f);
    Assert(beta_m >= 0 && beta_m <= 1);

    beta_n = props.float_("beta_n", 0.2f);
    Assert(beta_n >= 0 && beta_n <= 1);

    Float alpha = props.float_("alpha", 2.f);
    //sigma_a = props.color("sigma_a");
    sigma_a = props.texture<Texture>("sigma_a", .5f);
    gamma_o = safe_asin(h);
    eta = props.float_("eta", 1.55f); //TODO: props.texture<Texture>("eta", 0.f); 

    // Compute longitudinal variance from $\beta_m$
    static_assert( p_max >= 3, "do not handle low p_max");

    v[0] = sqr(0.726f * beta_m + 0.812f * sqr(beta_m) + 3.7f * pow<20>(beta_m));
    v[1] = .25f * v[0];
    v[2] = 4 * v[0];

    for (int p = 3; p <= p_max; ++p){
        v[p] = v[2];
    }

    // Compute azimuthal logistic scale factor from $\beta_n$
    s = sqrt_pi_over_8 * (0.265f * beta_n + 1.194f * sqr(beta_n) + 5.372f * pow<22>(beta_n));
    Assert(!std::isnan(s));

    // Compute $\alpha$ terms for hair scales
    sin_2k_alpha[0] = std::sin(radians(alpha));
    cos_2k_alpha[0] = safe_sqrt(1 - sqr(sin_2k_alpha[0]));
    for (int i = 1; i < 3; ++i) {
        sin_2k_alpha[i] = 2 * cos_2k_alpha[i - 1] * sin_2k_alpha[i - 1];
        cos_2k_alpha[i] = sqr(cos_2k_alpha[i - 1]) - sqr(sin_2k_alpha[i - 1]);
    }
}


template <typename Float, typename Spectrum>
std::pair<typename HairBSDF<Float, Spectrum>::BSDFSample3f, Spectrum> HairBSDF<Float, Spectrum>::sample(const BSDFContext &ctx,
                                                                                                        const SurfaceInteraction3f &si,
                                                                                                        Float sample1,
                                                                                                        const Point2f &sample2,
                                                                                                        Mask active) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

    Vector3f wi = si.wi;
    Vector3f wo(wi);
    BSDFSample3f bs = zero<BSDFSample3f>();

    Spectrum result(0.f);

    Float sin_theta_o = wo.x();
    Float cos_theta_o = safe_sqrt(1 - sqr(sin_theta_o));
    Float phi_o = std::atan2(wo.z(), wo.y());

    // Derive four random samples from _u2_
    Point2f u[2] = {demux_float(sample2[0]), demux_float(sample2[1])}; //u2

    // Determine which term $p$ to sample for hair scattering
    std::array<Float, p_max + 1> ap_pdf = compute_ap_pdf(cos_theta_o);
    int p;
    for (p = 0; p < p_max; ++p) {
        if (u[0][0] < ap_pdf[p]) break;
        u[0][0] -= ap_pdf[p];
    }


    // Rotate $\sin \thetao$ and $\cos \thetao$ to account for hair scale tilt
    Float sin_theta_op, cos_theta_op;
    tilt_scales(cos_theta_o, sin_theta_o, p, sin_theta_op, cos_theta_op);

    // Sample $M_p$ to compute $\thetai$
    u[1][0] = std::max(u[1][0], Float(1e-5));
    Float cos_theta = 1 + v[p] * std::log(u[1][0] + (1 - u[1][0]) * std::exp(-2 / v[p]));
    Float sin_theta = safe_sqrt(1 - sqr(cos_theta));
    Float cos_phi = std::cos(2 * math::Pi<ScalarFloat> * u[1][1]);
    Float sin_theta_i = -cos_theta * sin_theta_op + sin_theta * cos_phi * cos_theta_op;
    Float cos_theta_i = safe_sqrt(1 - sqr(sin_theta_i));

    std::cout << cos_theta << ", " << sin_theta << ", " << cos_phi << ", " << sin_theta_i << ", " << cos_theta_i << std::endl;

    // Sample $N_p$ to compute $\Delta\phi$

    // Compute $\gammat$ for refracted ray
    Float etap = std::sqrt(eta * eta - sqr(sin_theta_o)) / cos_theta_o;
    Float sin_gamma_t = h / etap;
    Float gamma_t = safe_asin(sin_gamma_t);
    Float dphi;
    if (p < p_max)
        dphi = warp::Phi(p, gamma_o, gamma_t) + warp::sample_trimmed_logistic(u[0][1], s, -math::Pi<ScalarFloat>, math::Pi<ScalarFloat>);
    else
        dphi = 2 * math::Pi<ScalarFloat> * u[0][1];

    // Compute _wi_ from sampled hair scattering angles
    Float phi_i = phi_o + dphi;
    wi = Vector3f(sin_theta_i, cos_theta_i * std::cos(phi_i), cos_theta_i * std::sin(phi_i)); //TODO:!!!!!!!!

    // Compute PDF for sampled hair scattering direction _wi_
    for (int p = 0; p < p_max; ++p) {
        // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for scales
        Float sin_theta_op, cos_theta_op;
        tilt_scales(cos_theta_o, sin_theta_o, p, sin_theta_op, cos_theta_op);

        // Handle out-of-range $\cos \thetao$ from scale adjustment
        cos_theta_op = std::abs(cos_theta_op);
        bs.pdf += warp::Mp(cos_theta_i, cos_theta_op, sin_theta_i, sin_theta_op, v[p]) *
                           ap_pdf[p] * warp::Np(dphi, p, s, gamma_o, gamma_t);
    }
    bs.pdf += warp::Mp(cos_theta_i, cos_theta_o, sin_theta_i, sin_theta_o, v[p_max]) *
                       ap_pdf[p_max] * (1 / (2 * math::Pi<ScalarFloat>));

    return {bs, eval(ctx, si, bs.wo, active)};

}

template <typename Float, typename Spectrum>
Spectrum HairBSDF<Float, Spectrum>::eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                        const Vector3f &wo, Mask active) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

    //Float cos_theta_i = Frame3f::cos_theta(si.wi),
    //      cos_theta_o = Frame3f::cos_theta(wo);

    //active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

    Vector3f wi = si.wi;

    // Compute hair coordinate system terms related to _wo_
    Float sin_theta_o = wo.x();
    Float cos_theta_o = safe_sqrt(1 - sqr(sin_theta_o));
    Float phi_o = std::atan2(wo.z(), wo.y());

    // Compute hair coordinate system terms related to _wi_
    Float sin_theta_i = wi.x();
    Float cos_theta_i = safe_sqrt(1 - sqr(sin_theta_i));
    Float phi_i = std::atan2(wi.z(), wi.y());

    // Compute $\cos \thetat$ for refracted ray
    Float sin_theta_t = sin_theta_o / eta;
    Float cos_theta_t = safe_sqrt(1 - sqr(sin_theta_t));

    // Compute $\gammat$ for refracted ray
    Float etap = std::sqrt(eta * eta - sqr(sin_theta_o)) / cos_theta_o;
    Float sin_gamma_t = h / etap;
    Float cos_gamma_t = safe_sqrt(1 - sqr(sin_gamma_t));
    Float gamma_t = safe_asin(sin_gamma_t);

    // Compute the transmittance _T_ of a single path through the cylinder
    Spectrum T = exp(-sigma_a * (2 * cos_gamma_t / cos_theta_t));

    // Evaluate hair BSDF
    Float phi = phi_i - phi_o;
    std::array<Spectrum, p_max + 1> ap = Ap(cos_theta_o, eta, h, T);
    Spectrum fsum(0.);
    for (int p = 0; p < p_max; ++p) {
        // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for scales
        Float sin_theta_op, cos_theta_op;
        tilt_scales(cos_theta_o, sin_theta_o, p, sin_theta_op, cos_theta_op);

        // Handle out-of-range $\cos \thetao$ from scale adjustment
        cos_theta_op = std::abs(cos_theta_op);
        fsum += warp::Mp(cos_theta_i, cos_theta_op, sin_theta_i, sin_theta_op, v[p]) * ap[p] *
                warp::Np(phi, p, s, gamma_o, gamma_t);
    }

    // Compute contribution of remaining terms after _pMax_
    fsum += warp::Mp(cos_theta_i, cos_theta_o, sin_theta_i, sin_theta_o, v[p_max]) * ap[p_max] /
                    (2.f * math::Pi<ScalarFloat>);
    if (abs_cos_theta(wi) > 0) fsum /= abs_cos_theta(wi);
    Assert(!std::isinf(fsum.y()) && !std::isnan(fsum.y()));

    return fsum;
    
}

template <typename Float, typename Spectrum>
Float HairBSDF<Float, Spectrum>::pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                    const Vector3f &wo, Mask active) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

    //Float cos_theta_i = Frame3f::cos_theta(si.wi),
    //      cos_theta_o = Frame3f::cos_theta(wo);

    Vector3f wi = si.wi;

    // Compute hair coordinate system terms related to _wo_
    Float sin_theta_o = wo.x();
    Float cos_theta_o = safe_sqrt(1 - sqr(sin_theta_o));
    Float phi_o = std::atan2(wo.z(), wo.y());

    // Compute hair coordinate system terms related to _wi_
    Float sin_theta_i = wi.x();
    Float cos_theta_i = safe_sqrt(1 - sqr(sin_theta_i));
    Float phi_i = std::atan2(wi.z(), wi.y());

    // Compute $\gammat$ for refracted ray
    Float etap = std::sqrt(eta * eta - sqr(sin_theta_o)) / cos_theta_o;
    Float sin_gamma_t = h / etap;
    Float gamma_t = safe_asin(sin_gamma_t);

    // Compute PDF for $A_p$ terms
    std::array<Float, p_max + 1> ap_pdf = compute_ap_pdf(cos_theta_o);

    // Compute PDF sum for hair scattering events
    Float phi = phi_i - phi_o;
    Float pdf = 0;
    for (int p = 0; p < p_max; ++p) {
        // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for scales
        Float sin_theta_op, cos_theta_op;
        tilt_scales(cos_theta_o, sin_theta_o, p, sin_theta_op, cos_theta_op);

        // Handle out-of-range $\cos \thetao$ from scale adjustment
        cos_theta_op = std::abs(cos_theta_op);
        pdf += warp::Mp(cos_theta_i, cos_theta_op, sin_theta_i, sin_theta_op, v[p]) *
                        ap_pdf[p] * warp::Np(phi, p, s, gamma_o, gamma_t);
    }
    pdf += warp::Mp(cos_theta_i, cos_theta_o, sin_theta_i, sin_theta_o, v[p_max]) *
                    ap_pdf[p_max] * (1 / (2 * math::Pi<ScalarFloat>));
    return pdf;
}

template <typename Float, typename Spectrum>
std::string HairBSDF<Float, Spectrum>::to_string() const {
    std::ostringstream oss;
    oss << "HairBSDF" << std::endl;
    return oss.str();
}

NAMESPACE_END(mitsuba)