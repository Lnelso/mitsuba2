#include <mitsuba/core/string.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/hair.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
HairBSDF<Float, Spectrum>::HairBSDF(const Properties &props) : Base(props) {
    //h = props.float_("h", 0.15f);

    beta_m = props.float_("beta_m", 0.3f);
    Assert(beta_m >= 0 && beta_m <= 1);

    beta_n = props.float_("beta_n", 0.3f);
    Assert(beta_n >= 0 && beta_n <= 1);

    Float alpha = props.float_("alpha", 2.f);

    if(props.has_property("absorption")){
        sigma_a = props.texture<Texture>("absorption", 1.0f);
        mode = Mode_sigma_a::Absorption;
    } else if (props.has_property("reflectance")){
        sigma_a_reflectance = props.texture<Texture>("reflectance", 1.0f);
        mode = Mode_sigma_a::Reflectance;
    } else if (props.has_property("eumelanin") || props.has_property("pheomelanin")){
        ce = props.float_("eumelanin", 0.3f);
        cp = props.float_("pheomelanin", 0.0f);
        mode = Mode_sigma_a::Concentration;
    } else{
        Log(LogLevel::Error, "A hair color need to be specified either through absorption, reflectance or eumelanin concentration");
    }

    //gamma_o = safe_asin(h);
    eta = props.float_("eta", 1.55f); //TODO: props.texture<Texture>("eta", 0.f); 

    // Compute longitudinal variance from beta_m
    static_assert( p_max >= 3, "do not handle low p_max");

    v[0] = sqr(0.726f * beta_m + 0.812f * sqr(beta_m) + 3.7f * pow<20>(beta_m));
    v[1] = .25f * v[0];
    v[2] = 4 * v[0];

    for (int p = 3; p <= p_max; ++p){
        v[p] = v[2];
    }

    // Compute azimuthal logistic scale factor from beta_n
    s = sqrt_pi_over_8 * (0.265f * beta_n + 1.194f * sqr(beta_n) + 5.372f * pow<22>(beta_n));
    Assert(!std::isnan(s));

    // Compute alpha terms for hair scales
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

    BSDFSample3f bs = zero<BSDFSample3f>();

    Float sin_theta_i, cos_theta_i, phi_i;
    get_angles(si.wi, sin_theta_i, cos_theta_i, phi_i);

    active &= Frame3f::cos_theta(si.wi) > 0.f;
    if (unlikely(none_or<false>(active)))
        return { bs, 0 };

    Float h = -1 + 2 * si.uv[1];
    Float gamma_o = safe_asin(h);

    // Derive four random samples from sample2
    Point2f u[2] = {demux_float(sample2[0]), demux_float(sample2[1])}; //u2

    // Determine which term p to sample for hair scattering
    std::array<Float, p_max + 1> ap_pdf = compute_ap_pdf(cos_theta_i, h, si, active);
    int p;
    for (p = 0; p < p_max; ++p) {
        if (u[0][0] < ap_pdf[p]) break;
        u[0][0] -= ap_pdf[p];
    }

    // Rotate sin_theta_o and cos_theta_o to account for hair scale tilt
    Float sin_theta_op, cos_theta_op;
    tilt_scales(sin_theta_i, cos_theta_i, p, sin_theta_op, cos_theta_op);

    // Sample Mp to compute theta_i
    u[1][0] = std::max(u[1][0], Float(1e-5));
    Float cos_theta = 1 + v[p] * std::log(u[1][0] + (1 - u[1][0]) * std::exp(-2 / v[p]));
    Float sin_theta = safe_sqrt(1 - sqr(cos_theta));
    Float cos_phi = std::cos(2 * math::Pi<ScalarFloat> * u[1][1]);

    Float sin_theta_o = -cos_theta * sin_theta_op + sin_theta * cos_phi * cos_theta_op;
    Float cos_theta_o = safe_sqrt(1 - sqr(sin_theta_o));

    // Sample Np to compute dphi

    // Compute gamma_t for refracted ray
    Float etap = std::sqrt(eta * eta - sqr(sin_theta_i)) / cos_theta_i;
    Float sin_gamma_t = h / etap;
    Float gamma_t = safe_asin(sin_gamma_t);
    Float dphi;
    if (p < p_max)
        dphi = warp::Phi(p, gamma_o, gamma_t) + warp::sample_trimmed_logistic(u[0][1], s, -math::Pi<Float>, math::Pi<Float>);
    else
        dphi = 2 * math::Pi<Float> * u[0][1];

    // Compute wo from sampled hair scattering angles
    Float phi_o = phi_i + dphi;
    bs.wo = -normalize(Vector3f(sin_theta_o, cos_theta_o * std::cos(phi_o), cos_theta_o * std::sin(phi_o)));

    // Compute PDF for sampled hair scattering direction wo
    /*for (int p = 0; p < p_max; ++p) {
        // Compute sin_thetao_ and cos_theta_o terms accounting for scales
        Float sin_theta_op, cos_theta_op;
        tilt_scales(sin_theta_i, cos_theta_i, p, sin_theta_op, cos_theta_op);

        // Handle out-of-range cos_theta_o from scale adjustment
        cos_theta_op = std::abs(cos_theta_op);
        bs.pdf += warp::Mp(cos_theta_o, cos_theta_op, sin_theta_o, sin_theta_op, v[p]) *
                           ap_pdf[p] * warp::Np(dphi, p, s, gamma_o, gamma_t);
    }
    bs.pdf += warp::Mp(cos_theta_o, cos_theta_i, sin_theta_o, sin_theta_i, v[p_max]) *
                       ap_pdf[p_max] * (1.0f / (2 * math::Pi<Float>));
    bs.eta = eta;*/

    bs.pdf = pdf(ctx, si, bs.wo, active);
    bs.eta = eta;

    if (bs.pdf == 0)
        return {bs, 0};

    auto value = eval(ctx, si, bs.wo, active) * Frame3f::cos_theta(bs.wo) / bs.pdf;

    for (int i=0; i<3; ++i) {
        if ((!std::isfinite(value[i]) || value[i] < 0)){
            std::cout << "h: " << h << " gamma: " << gamma_o << std::endl;
            std::cout << "pdf: " << bs.pdf << std::endl;
            std::cout << "cos_theta: " << Frame3f::cos_theta(bs.wo) << std::endl;
            Log(LogLevel::Error, "Stop.");
        }
    }

    return {bs, value};

}

template <typename Float, typename Spectrum>
Spectrum HairBSDF<Float, Spectrum>::eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                        const Vector3f &wo, Mask active) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

    active &= Frame3f::cos_theta(si.wi) > 0.f && Frame3f::cos_theta(wo) /*wo.z()*/ >= 0;
    if (unlikely(none_or<false>(active)))
        return 0.f;

    Float h = -1 +  2 * si.uv[1];
    Float gamma_o = safe_asin(h);

    // Compute hair coordinate system terms related to wi
    Float sin_theta_i, cos_theta_i, phi_i;
    get_angles(si.wi, sin_theta_i, cos_theta_i, phi_i);

    // Compute hair coordinate system terms related to wo
    Float sin_theta_o, cos_theta_o, phi_o;
    get_angles(wo, sin_theta_o, cos_theta_o, phi_o);
    
    // Compute cos_theta_t for refracted ray
    Float sin_theta_t = sin_theta_o / eta;
    Float cos_theta_t = safe_sqrt(1 - sqr(sin_theta_t));

    // Compute gamma_t for refracted ray
    Float etap = std::sqrt(eta * eta - sqr(sin_theta_o)) / cos_theta_o;
    Float sin_gamma_t = h / etap;
    Float cos_gamma_t = safe_sqrt(1 - sqr(sin_gamma_t));
    Float gamma_t = safe_asin(sin_gamma_t);

    // Compute the transmittance T of a single path through the cylinder
    Spectrum T = exp(-evaluate_sigma_a(si, active) * (2.0f * cos_gamma_t / cos_theta_t));

    // Evaluate hair BSDF
    Float phi = phi_i - phi_o;

    std::array<Spectrum, p_max + 1> ap = Ap(cos_theta_o, eta, h, T);
    Spectrum fsum(0.);
    for (int p = 0; p < p_max; ++p) {
        // Compute sin_theta_o and cos_theta_o terms accounting for scales
        Float sin_theta_op, cos_theta_op;
        tilt_scales(sin_theta_o, cos_theta_o, p, sin_theta_op, cos_theta_op);

        // Handle out-of-range cos_theta_o from scale adjustment
        cos_theta_op = std::abs(cos_theta_op);
        fsum += warp::Mp(cos_theta_i, cos_theta_op, sin_theta_i, sin_theta_op, v[p]) * ap[p] *
                warp::Np(phi, p, s, gamma_o, gamma_t);
    }
    // Compute contribution of remaining terms after p_max
    fsum += warp::Mp(cos_theta_i, cos_theta_o, sin_theta_i, sin_theta_o, v[p_max]) * ap[p_max] /
                    (2.f * math::Pi<Float>);
    if (abs_cos_theta(si.wi) > 0) fsum /= abs_cos_theta(si.wi);

    Assert(!std::isinf(fsum.y()) && !std::isnan(fsum.y()));

    return fsum;    
}

template <typename Float, typename Spectrum>
Float HairBSDF<Float, Spectrum>::pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                    const Vector3f &wo, Mask active) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

    active &= Frame3f::cos_theta(si.wi) > 0.f && Frame3f::cos_theta(wo) > 0.f;
    if (unlikely(none_or<false>(active)))
        return 0;

    Float h = -1 + 2 * si.uv[1];
    Float gamma_o = safe_asin(h);

    // Compute hair coordinate system terms related to wi
    Float sin_theta_i, cos_theta_i, phi_i;
    get_angles(si.wi, sin_theta_i, cos_theta_i, phi_i);

    // Compute hair coordinate system terms related to wo
    Float sin_theta_o, cos_theta_o, phi_o;
    get_angles(wo, sin_theta_o, cos_theta_o, phi_o);

    // Compute $\gammat$ for refracted ray
    Float etap = std::sqrt(eta * eta - sqr(sin_theta_o)) / cos_theta_o;
    Float sin_gamma_t = h / etap;
    Float gamma_t = safe_asin(sin_gamma_t);

    // Compute PDF for Ap terms
    std::array<Float, p_max + 1> ap_pdf = compute_ap_pdf(cos_theta_o, h, si, active);

    // Compute PDF sum for hair scattering events
    Float phi = phi_i - phi_o;
    Float pdf = 0;
    for (int p = 0; p < p_max; ++p) {
        // Compute sin_theta_o and cos_theta_o terms accounting for scales
        Float sin_theta_op, cos_theta_op;
        tilt_scales(sin_theta_o, cos_theta_o, p, sin_theta_op, cos_theta_op);

        // Handle out-of-range cos_theta_o from scale adjustment
        cos_theta_op = std::abs(cos_theta_op);
        pdf += warp::Mp(cos_theta_i, cos_theta_op, sin_theta_i, sin_theta_op, v[p]) *
                        ap_pdf[p] * warp::Np(phi, p, s, gamma_o, gamma_t);
    }

    pdf += warp::Mp(cos_theta_i, cos_theta_o, sin_theta_i, sin_theta_o, v[p_max]) *
                    ap_pdf[p_max] * (1.0f / (2 * math::Pi<Float>));

    return pdf;
}

template <typename Float, typename Spectrum>
std::string HairBSDF<Float, Spectrum>::to_string() const {
    std::ostringstream oss;
    oss << "HairBSDF[" << std::endl
        //<< "   h = " << h << ","
        //<< "   gamma_o = " << gamma_o << ","
        << "   eta = " << eta << ","
        << "   beta_m = " << beta_m << ","
        << "   beta_n = " << beta_n << ","
        << "   v[0] = " << v[0] << ","
        << "   s = " << s << ","
        << "]";
    return oss.str();
}

NAMESPACE_END(mitsuba)