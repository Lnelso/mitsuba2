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
    h_xml = props.float_("h_xml", -2.f);
    Assert(h_xml >= -1 && x_xml <= 1);

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
    sin_2k_alpha[0] = sin(radians(alpha));
    cos_2k_alpha[0] = safe_sqrt(1 - sqr(sin_2k_alpha[0]));
    for (int i = 1; i < 3; ++i) {
        sin_2k_alpha[i] = 2 * cos_2k_alpha[i - 1] * sin_2k_alpha[i - 1];
        cos_2k_alpha[i] = sqr(cos_2k_alpha[i - 1]) - sqr(sin_2k_alpha[i - 1]);
    }

    m_components.push_back(BSDFFlags::Glossy | BSDFFlags::SpatiallyVarying);
    m_components.push_back(BSDFFlags::Reflection | BSDFFlags::SpatiallyVarying);
    m_components.push_back(BSDFFlags::Transmission | BSDFFlags::SpatiallyVarying);

    m_flags = m_components[0] | m_components[1] | m_components[2];
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

    Float h = h_xml != -2 ? h_xml : -1 + 2 * si.uv[1];
    Float gamma_o = safe_asin(h);

    // Derive four random samples from sample2
    Point2f u[2] = {demux_float(sample2[0]), demux_float(sample2[1])}; //u2

    // Determine which term p to sample for hair scattering
    std::array<Float, p_max + 1> ap_pdf = compute_ap_pdf(cos_theta_i, h, si, active);
    /*
     * The for loop below is a convoluted way to rewrite this code with enoki.
     * int p;
     * for (p = 0; p < p_max; ++p) {
     *     if (u[0][0] < ap_pdf[p]) break;
     *     u[0][0] -= ap_pdf[p];
     * }
     */

    UInt p = 0;
    Mask broken = false;
    
    for(int i = 0; i < p_max; ++i){
        active = u[0][0] < ap_pdf[i];
        masked(u[0][0], !active) -= ap_pdf[i];
        masked(broken, active) = true;
        masked(p, !active && !broken) += 1;
    }
    
    // Rotate sin_theta_o and cos_theta_o to account for hair scale tilt
    Float sin_theta_op, cos_theta_op;

    tilt_scales(sin_theta_i, cos_theta_i, p, sin_theta_op, cos_theta_op);

    Float d = gather<Float>(v, p, true);

    // Sample Mp to compute theta_i
    u[1][0] = max(u[1][0], Float(1e-5));
    Float cos_theta = 1 + d * log(u[1][0] + (1 - u[1][0]) * exp(-2 / d));
    Float sin_theta = safe_sqrt(1 - sqr(cos_theta));
    Float cos_phi = cos(2 * Pi * u[1][1]);

    Float sin_theta_o = -cos_theta * sin_theta_op + sin_theta * cos_phi * cos_theta_op;
    Float cos_theta_o = safe_sqrt(1 - sqr(sin_theta_o));

    // Sample Np to compute dphi

    // Compute gamma_t for refracted ray
    Float etap = sqrt(eta * eta - sqr(sin_theta_i)) / cos_theta_i;
    Float sin_gamma_t = h / etap;
    Float gamma_t = safe_asin(sin_gamma_t);
    Float dphi;
    active = p < p_max;
    dphi = select(active, 
                  warp::Phi(p, gamma_o, gamma_t) +
                  warp::sample_trimmed_logistic<Float>(u[0][1], s, -Pi, Pi),
                  2 * Pi * u[0][1]);

    // Compute wo from sampled hair scattering angles
    Float phi_o = phi_i + dphi;
    bs.wo = Vector3f(sin_theta_o, cos_theta_o * cos(phi_o), cos_theta_o * sin(phi_o));
    bs.eta = eta;

    /*Float pdf = 0;
    for (int p = 0; p < p_max; ++p) {
        // Compute sin_theta_o and cos_theta_o terms accounting for scales
        Float sin_theta_op, cos_theta_op;
        tilt_scales(sin_theta_i, cos_theta_i, p, sin_theta_op, cos_theta_op);

        // Handle out-of-range cos_theta_o from scale adjustment
        cos_theta_op = abs(cos_theta_op);
        pdf += warp::Mp(cos_theta_o, cos_theta_op, sin_theta_o, sin_theta_op, v[p]) *
               ap_pdf[p] *
               warp::Np(dphi, p, s, gamma_o, gamma_t);
    }

    pdf += warp::Mp(cos_theta_o, cos_theta_i, sin_theta_o, sin_theta_i, v[p_max]) *
                    ap_pdf[p_max] * (1.0f / (2 * Pi));*/
    bs.pdf = pdf(ctx, si, bs.wo, active);

    active &= bs.pdf > 0.0f;
    return {bs, select(active, unpolarized<Spectrum>(eval(ctx, si, bs.wo, active) / bs.pdf), 0.f)};
}

template <typename Float, typename Spectrum>
Spectrum HairBSDF<Float, Spectrum>::eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                        const Vector3f &wo, Mask active) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

    Float h = h_xml != -2 ? h_xml : -1 + 2 * si.uv[1];
    Float gamma_o = safe_asin(h);

    // Compute hair coordinate system terms related to wi
    Float sin_theta_i, cos_theta_i, phi_i;
    get_angles(si.wi, sin_theta_i, cos_theta_i, phi_i);

    // Compute hair coordinate system terms related to wo
    Float sin_theta_o, cos_theta_o, phi_o;
    get_angles(wo, sin_theta_o, cos_theta_o, phi_o);

    // Compute cos_theta_t for refracted ray
    Float sin_theta_t = sin_theta_i / eta;
    Float cos_theta_t = safe_sqrt(1 - sqr(sin_theta_t));

    // Compute gamma_t for refracted ray
    Float etap = sqrt(eta * eta - sqr(sin_theta_i)) / cos_theta_i;
    Float sin_gamma_t = h / etap;
    Float cos_gamma_t = safe_sqrt(1 - sqr(sin_gamma_t));
    Float gamma_t = safe_asin(sin_gamma_t);

    // Compute the transmittance T of a single path through the cylinder
    Spectrum T = exp(-evaluate_sigma_a(si, active) * (2.0f * cos_gamma_t / cos_theta_t));
    // Evaluate hair BSDF
    Float phi = phi_o - phi_i;

    std::array<Spectrum, p_max + 1> ap = Ap(cos_theta_i, eta, h, T);
    Spectrum fsum(0.);
    for (int p = 0; p < p_max; ++p) {
        // Compute sin_theta_o and cos_theta_o terms accounting for scales
        Float sin_theta_op, cos_theta_op;
        tilt_scales(sin_theta_i, cos_theta_i, p, sin_theta_op, cos_theta_op);

        // Handle out-of-range cos_theta_o from scale adjustment
        cos_theta_op = abs(cos_theta_op);
        fsum += warp::Mp(cos_theta_o, cos_theta_op, sin_theta_o, sin_theta_op, v[p]) *
                ap[p] *
                warp::Np(phi, p, s, gamma_o, gamma_t);
    }

    // Compute contribution of remaining terms after p_max
    fsum += warp::Mp(cos_theta_o, cos_theta_i, sin_theta_o, sin_theta_i, v[p_max]) *
            ap[p_max] / (2.f * Pi);

    active &= abs_cos_theta(wo) > 0;

    return select(active, unpolarized<Spectrum>(fsum / abs_cos_theta(wo)), 0.f);
}

template <typename Float, typename Spectrum>
Float HairBSDF<Float, Spectrum>::pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                    const Vector3f &wo, Mask active) const {
    MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

    Float h = h_xml != -2 ? h_xml : -1 + 2 * si.uv[1];
    Float gamma_o = safe_asin(h);

    // Compute hair coordinate system terms related to wi
    Float sin_theta_i, cos_theta_i, phi_i;
    get_angles(si.wi, sin_theta_i, cos_theta_i, phi_i);

    // Compute hair coordinate system terms related to wo
    Float sin_theta_o, cos_theta_o, phi_o;
    get_angles(wo, sin_theta_o, cos_theta_o, phi_o);

    // Compute $\gammat$ for refracted ray
    Float etap = sqrt(eta * eta - sqr(sin_theta_i)) / cos_theta_i;
    Float sin_gamma_t = h / etap;
    Float gamma_t = safe_asin(sin_gamma_t);

    // Compute PDF for Ap terms
    std::array<Float, p_max + 1> ap_pdf = compute_ap_pdf(cos_theta_i, h, si, active);

    // Compute PDF sum for hair scattering events
    Float phi = phi_o - phi_i;

    Float pdf = 0;
    for (int p = 0; p < p_max; ++p) {
        // Compute sin_theta_o and cos_theta_o terms accounting for scales
        Float sin_theta_op, cos_theta_op;
        tilt_scales(sin_theta_i, cos_theta_i, p, sin_theta_op, cos_theta_op);

        // Handle out-of-range cos_theta_o from scale adjustment
        cos_theta_op = abs(cos_theta_op);
        pdf += warp::Mp(cos_theta_o, cos_theta_op, sin_theta_o, sin_theta_op, v[p]) *
               ap_pdf[p] *
               warp::Np(phi, p, s, gamma_o, gamma_t);
    }

    pdf += warp::Mp(cos_theta_o, cos_theta_i, sin_theta_o, sin_theta_i, v[p_max]) *
           ap_pdf[p_max] * 
           (1.0f / (2 * Pi));

    return pdf;
}

template <typename Float, typename Spectrum>
std::string HairBSDF<Float, Spectrum>::to_string() const {
    std::ostringstream oss;
    oss << "HairBSDF[" << std::endl
        << "   eta = " << eta << ","
        << "   beta_m = " << beta_m << ","
        << "   beta_n = " << beta_n << ","
        << "   v[0] = " << v[0] << ","
        << "   s = " << s << ","
        << "]";
    return oss.str();
}

NAMESPACE_END(mitsuba)