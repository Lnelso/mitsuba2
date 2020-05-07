//
// Created by Lionel Pellier on 2020-02-25.
//

#include <iostream>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/random.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/kdtree.h>
#include <mitsuba/render/sensor.h>

#include <string>
#include <fstream>


#define MTS_HAIR_USE_FANCY_CLIPPING 1

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class HairKDTree: public TShapeKDTree<BoundingBox<Point<scalar_t<Float>, 3>>, uint32_t,
                                      SurfaceAreaHeuristic3<scalar_t<Float>>,
                                      HairKDTree<Float, Spectrum>> {
public:
    MTS_IMPORT_TYPES(Shape, Mesh)
    using SurfaceAreaHeuristic3f = SurfaceAreaHeuristic3<ScalarFloat>;
    using Size                   = uint32_t;
    using Index                  = uint32_t;

    using Base = TShapeKDTree<ScalarBoundingBox3f, uint32_t, SurfaceAreaHeuristic3f, HairKDTree>;
    using typename Base::KDNode;
    using Base::ready;
    using Base::set_clip_primitives;
    using Base::set_exact_primitive_threshold;
    using Base::set_max_depth;
    using Base::set_min_max_bins;
    using Base::set_retract_bad_splits;
    using Base::set_stop_primitives;
    using Base::bbox;
    using Base::m_bbox;
    using Base::m_nodes;
    using Base::m_indices;
    using Base::m_index_count;
    using Base::m_node_count;
    using Base::build;

    using Point = typename Base::Point;
    using Vector = typename Base::Point;
    using BoundingBox = typename Base::BoundingBox;
    using Scalar = typename Base::Scalar;

    HairKDTree(const Properties &props, std::vector<Point> &vertices,
               std::vector<bool> &vertex_starts_fiber, std::vector<Float> radius_per_vertex, Float max_radius, bool cylinder)
            : Base(SurfaceAreaHeuristic3f(
                  props.float_("kd_intersection_cost", 20.f),
                  props.float_("kd_traversal_cost", 15.f),
                  props.float_("kd_empty_space_bonus", .9f))){

        if (props.has_property("kd_stop_prims"))
            set_stop_primitives(props.int_("kd_stop_prims"));

        if (props.has_property("kd_max_depth"))
            set_max_depth(props.int_("kd_max_depth"));

        if (props.has_property("kd_min_max_bins"))
            set_min_max_bins(props.int_("kd_min_max_bins"));

        if (props.has_property("kd_clip"))
            set_clip_primitives(props.bool_("kd_clip"));

        if (props.has_property("kd_retract_bad_splits"))
            set_retract_bad_splits(props.bool_("kd_retract_bad_splits"));

        if (props.has_property("kd_exact_primitive_threshold"))
            set_exact_primitive_threshold(props.int_("kd_exact_primitive_threshold"));

        m_vertices.swap(vertices);
        m_radius_per_vertex.swap(radius_per_vertex);
        m_vertex_starts_fiber.swap(vertex_starts_fiber);
        m_hair_count = 0;
        m_seg_index.reserve(m_vertices.size());
        m_cylinder = cylinder;

        for (size_t i=0; i<m_vertices.size()-1; i++) {
            m_bbox.expand(m_vertices[i]);
            if (m_vertex_starts_fiber[i])
                m_hair_count++;
            if (!m_vertex_starts_fiber[i+1])
                m_seg_index.push_back((Index) i);
        }

        m_bbox.expand(m_vertices[m_vertices.size() - 1]);
        Vector extra = (m_bbox.extents() + max_radius);
        m_bbox.min -= extra;
        m_bbox.max += extra;

        /*
        const Float eps = MTS_KD_AABB_EPSILON;

        m_bbox.min -= (m_bbox.max-m_bbox.min) * eps + VectorType(eps);
        m_bbox.max += (m_bbox.max-m_bbox.min) * eps + VectorType(eps);
         */

        m_segment_count = m_seg_index.size();

        //TODO: logging

        set_stop_primitives(1);
        set_exact_primitive_threshold(16384);
        set_clip_primitives(true);
        set_retract_bad_splits(true);

        build();

        //TODO: logging

        for (Size i=0; i<m_index_count; ++i)
            m_indices[i] = m_seg_index[m_indices[i]];
    }

    MTS_INLINE const std::vector<Point> &vertices() const {
        return m_vertices;
    }

    MTS_INLINE const std::vector<bool> &start_fiber() const {
        return m_vertex_starts_fiber;
    }

    MTS_INLINE Float radius(Index iv) const {
        return m_radius_per_vertex[iv];
    }

    MTS_INLINE Size segment_count() const {
        return m_segment_count;
    }

    MTS_INLINE Size hair_count() const {
        return m_hair_count;
    }

    MTS_INLINE Size vertex_count() const {
        return m_vertices.size();
    }

    template <bool ShadowRay>
    MTS_INLINE std::pair<Mask, Float> ray_intersect(const Ray3f &ray,
                                                    Float *cache,
                                                    Mask active) const {
        ENOKI_MARK_USED(active);
        if constexpr (!is_array_v<Float>)
            return ray_intersect_scalar<ShadowRay>(ray, cache);
        else
            return ray_intersect_packet<ShadowRay>(ray, cache, active);
    }

    template <bool ShadowRay>
    MTS_INLINE std::pair<bool, Float> ray_intersect_scalar(Ray3f ray,
                                                           Float *cache) const {
        /// Ray traversal stack entry
        struct KDStackEntry {
            // Ray distance associated with the node entry and exit point
            Float mint, maxt;
            // Pointer to the far child
            const KDNode *node;
        };

        // Allocate the node stack
        KDStackEntry stack[MTS_KD_MAXDEPTH];
        int32_t stack_index = 0;

        // True if an intersection has been found
        bool hit = false;

        // Intersect against the scene bounding box
        auto bbox_result = m_bbox.ray_intersect(ray);

        Float mint = std::max(ray.mint, std::get<1>(bbox_result));
        Float maxt = std::min(ray.maxt, std::get<2>(bbox_result));

        const KDNode *node = m_nodes.get();
        while (mint <= maxt) {
            if (likely(!node->leaf())) { // Inner node
                const Float split   = node->split();
                const uint32_t axis = node->axis();

                //Compute parametric distance along the rays to the split plane
                Float t_plane = (split - ray.o[axis]) * ray.d_rcp[axis];

                bool left_first  = (ray.o[axis] < split) || (ray.o[axis] == split && ray.d[axis] >= 0.f),
                     start_after = t_plane < mint,
                     end_before  = t_plane > maxt || t_plane < 0.f || !std::isfinite(t_plane),
                     single_node = start_after || end_before;

                //If we only need to visit one node, just pick the correct one and continue
                if (likely(single_node)) {
                    bool visit_left = end_before == left_first;
                    node = node->left() + (visit_left ? 0 : 1);
                    continue;
                }

                //Visit both child nodes in the right order
                Index node_offset = left_first ? 0 : 1;
                const KDNode *left   = node->left(),
                             *n_cur  = left + node_offset,
                             *n_next = left + (1 - node_offset);

                //Postpone visit to 'n_next'
                KDStackEntry& entry = stack[stack_index++];
                entry.mint = t_plane;
                entry.maxt = maxt;
                entry.node = n_next;

                //Visit 'n_cur' now
                node = n_cur;
                maxt = t_plane;
                continue;
            } else if (node->primitive_count() > 0) { // Arrived at a leaf node
                Index prim_start = node->primitive_offset();
                Index prim_end = prim_start + node->primitive_count();
                for (Index i = prim_start; i < prim_end; i++) {
                    Index prim_index = m_indices[i];

                    bool prim_hit;
                    Float prim_t;
                    std::tie(prim_hit, prim_t) =
                        intersect_prim(prim_index, ray, cache, true);

                    if (unlikely(prim_hit)) {
                        if (ShadowRay)
                            return { true, prim_t };

                        Assert(prim_t >= ray.mint && prim_t <= ray.maxt);
                        ray.maxt = prim_t;
                        hit = true;
                    }
                }
            }

            if (likely(stack_index > 0)) {
                --stack_index;
                KDStackEntry& entry = stack[stack_index];
                mint = entry.mint;
                maxt = std::min(entry.maxt, ray.maxt);
                node = entry.node;
            } else {
                break;
            }
        }
        return { hit, hit ? ray.maxt : math::Infinity<Float> };
    }

    template <bool ShadowRay>
    MTS_INLINE std::pair<Mask, Float> ray_intersect_packet(Ray3f ray, Float *cache, Mask active) const {
        /// Ray traversal stack entry
        struct KDStackEntry {
            // Ray distance associated with the node entry and exit point
            Float mint, maxt;
            // Is the corresponding SIMD lane enabled?
            Mask active;
            // Pointer to the far child
            const KDNode *node;
        };

        // Allocate the node stack
        KDStackEntry stack[MTS_KD_MAXDEPTH];
        int32_t stack_index = 0;

        // True if an intersection has been found
        Mask hit = false;

        const KDNode *node = m_nodes.get();

        /* Intersect against the scene bounding box */
        auto bbox_result = m_bbox.ray_intersect(ray);
        Float mint = enoki::max(ray.mint, std::get<1>(bbox_result));
        Float maxt = enoki::min(ray.maxt, std::get<2>(bbox_result));

        while (true) {
            active = active && (maxt >= mint);
            if (ShadowRay)
                active = active && !hit;

            if (likely(any(active))) {
                if (likely(!node->leaf())) { // Inner node
                    const scalar_t<Float> split = node->split();
                    const uint32_t axis = node->axis();

                    // Compute parametric distance along the rays to the split plane
                    Float t_plane          = (split - ray.o[axis]) * ray.d_rcp[axis];
                    Mask left_first        = (ray.o[axis] < split) ||
                                              (eq(ray.o[axis], split) && ray.d[axis] >= 0.f),
                         start_after       = t_plane < mint,
                         end_before        = t_plane > maxt || t_plane < 0.f || !enoki::isfinite(t_plane),
                         single_node       = start_after || end_before,
                         visit_left        = eq(end_before, left_first),
                         visit_only_left   = single_node &&  visit_left,
                         visit_only_right  = single_node && !visit_left;

                    bool all_visit_only_left  = all(visit_only_left || !active),
                         all_visit_only_right = all(visit_only_right || !active),
                         all_visit_same_node  = all_visit_only_left || all_visit_only_right;

                    /* If we only need to visit one node, just pick the correct one and continue */
                    if (all_visit_same_node) {
                        node = node->left() + (all_visit_only_left ? 0 : 1);
                        continue;
                    }

                    size_t left_votes  = count(left_first && active),
                           right_votes = count(!left_first && active);

                    bool go_left = left_votes >= right_votes;

                    Mask go_left_bcast = Mask(go_left),
                         correct_order = eq(left_first, go_left_bcast),
                         visit_both    = !single_node,
                         visit_cur     = visit_both || eq (visit_left, go_left_bcast),
                         visit_next    = visit_both || neq(visit_left, go_left_bcast);

                    /* Visit both child nodes in the right order */
                    Index node_offset = go_left ? 0 : 1;
                    const KDNode *left   = node->left(),
                                 *n_cur  = left + node_offset,
                                 *n_next = left + (1 - node_offset);

                    /* Postpone visit to 'n_next' */
                    Mask sel0 =  correct_order && visit_both,
                         sel1 = !correct_order && visit_both;
                    KDStackEntry& entry = stack[stack_index++];
                    entry.mint = select(sel0, t_plane, mint);
                    entry.maxt = select(sel1, t_plane, maxt);
                    entry.active = active && visit_next;
                    entry.node = n_next;

                    /* Visit 'n_cur' now */
                    mint = select(sel1, t_plane, mint);
                    maxt = select(sel0, t_plane, maxt);
                    active = active && visit_cur;
                    node = n_cur;
                    continue;
                } else if (node->primitive_count() > 0) { // Arrived at a leaf node
                    Index prim_start = node->primitive_offset();
                    Index prim_end = prim_start + node->primitive_count();
                    for (Index i = prim_start; i < prim_end; i++) {
                        Index prim_index = m_indices[i];

                        Mask prim_hit;
                        Float prim_t;
                        std::tie(prim_hit, prim_t) =
                            intersect_prim(prim_index, ray, cache, active);

                        if (!ShadowRay) {
                            Assert(all(!prim_hit || (prim_t >= ray.mint && prim_t <= ray.maxt)));
                            masked(ray.maxt, prim_hit) = prim_t;
                        }
                        hit |= prim_hit;
                    }
                }
            }

            if (likely(stack_index > 0)) {
                --stack_index;
                KDStackEntry& entry = stack[stack_index];
                mint = entry.mint;
                maxt = enoki::min(entry.maxt, ray.maxt);
                active = entry.active;
                node = entry.node;
            } else {
                break;
            }
        }

        return { hit, select(hit, ray.maxt, math::Infinity<Float>) };
    }

    MTS_INLINE BoundingBox bbox() const {
        return m_bbox;
    }

#if MTS_HAIR_USE_FANCY_CLIPPING == 1
    bool intersect_cyl_plane(Point plane_pt, Normal3f plane_nrml,
            Point cyl_pt, Vector cyl_d, Float radius, Point &center,
            Vector *axes, Float *lengths) const {
        
        if (abs_dot(plane_nrml, cyl_d) < math::Epsilon<Scalar>)
            return false;

        Vector B, A = cyl_d - dot(cyl_d, plane_nrml)*plane_nrml;

        Float length = norm(A);
        if (length > math::Epsilon<Scalar> && plane_nrml != cyl_d) {
            A /= length;
            B = cross(plane_nrml, A);
        } else {
            auto basis = coordinate_system(plane_nrml);
            A = basis.first;
            B = basis.second;
        }

        Vector delta = plane_pt - cyl_pt,
                delta_proj = delta - cyl_d*dot(delta, cyl_d);

        Float a_dot_d = dot(A, cyl_d);
        Float b_dot_d = dot(B, cyl_d);
        Float c0 = 1-a_dot_d*a_dot_d;
        Float c1 = 1-b_dot_d*b_dot_d;
        Float c2 = 2*dot(A, delta_proj);
        Float c3 = 2*dot(B, delta_proj);
        Float c4 = dot(delta, delta_proj) - radius*radius;

        Float lambda = (c2*c2/(4*c0) + c3*c3/(4*c1) - c4)/(c0*c1);

        Float alpha0 = -c2/(2*c0),
                beta0 = -c3/(2*c1);

        lengths[0] = std::sqrt(c1*lambda),
        lengths[1] = std::sqrt(c0*lambda);

        center = plane_pt + alpha0 * A + beta0 * B;
        axes[0] = A;
        axes[1] = B;
        return true;
    }

    BoundingBox intersect_cyl_face(int axis,
            const Point &min, const Point &max,
            const Point &cyl_pt, const Vector &cyl_d, Index iv) const {

        int axis1 = (axis + 1) % 3;
        int axis2 = (axis + 2) % 3;

        Normal3f plane_nrml(0.0f);
        plane_nrml[axis] = 1;

        Point ellipse_center;
        Vector ellipse_axes[2];
        Float ellipse_lengths[2];
        
        BoundingBox aabb;
        if (!intersect_cyl_plane(min, plane_nrml, cyl_pt, cyl_d, m_radius_per_vertex[iv] * (1 + math::Epsilon<Scalar>),
                               ellipse_center, ellipse_axes, ellipse_lengths)) {
            return aabb;
        }

        for (int i=0; i<4; ++i) {
            Point p1, p2;
            p1[axis] = p2[axis] = min[axis];
            p1[axis1] = ((i+1) & 2) ? min[axis1] : max[axis1];
            p1[axis2] = ((i+0) & 2) ? min[axis2] : max[axis2];
            p2[axis1] = ((i+2) & 2) ? min[axis1] : max[axis1];
            p2[axis2] = ((i+1) & 2) ? min[axis2] : max[axis2];

            Point2f p1l(
                    dot(p1 - ellipse_center, ellipse_axes[0]) / ellipse_lengths[0],
                    dot(p1 - ellipse_center, ellipse_axes[1]) / ellipse_lengths[1]);
            Point2f p2l(
                    dot(p2 - ellipse_center, ellipse_axes[0]) / ellipse_lengths[0],
                    dot(p2 - ellipse_center, ellipse_axes[1]) / ellipse_lengths[1]);

            Vector2f rel = p2l-p1l;
            Float A = dot(rel, rel);
            Float B = 2*dot(p1l, rel);
            Float C = dot(p1l, p1l)-1;

            auto coefs = math::solve_quadratic(A, B, C);
            Float x0 = std::get<1>(coefs);
            Float x1 = std::get<2>(coefs);
            if (std::get<0>(coefs)) {
                if (x0 >= 0 && x0 <= 1)
                    aabb.expand(p1+(p2-p1)*x0);
                if (x1 >= 0 && x1 <= 1)
                    aabb.expand(p1+(p2-p1)*x1);
            }
        }

        ellipse_axes[0] *= ellipse_lengths[0];
        ellipse_axes[1] *= ellipse_lengths[1];
        BoundingBox face_bounds(min, max);

        for (int i=0; i<2; ++i) {
            int j = (i==0) ? axis1 : axis2;
            Float alpha = ellipse_axes[0][j];
            Float beta = ellipse_axes[1][j];
            Float tmp = 1 / std::sqrt(alpha*alpha + beta*beta);
            Float cos_theta = alpha * tmp, sin_theta = beta*tmp;

            Point p1 = ellipse_center + cos_theta*ellipse_axes[0] + sin_theta*ellipse_axes[1];
            Point p2 = ellipse_center - cos_theta*ellipse_axes[0] - sin_theta*ellipse_axes[1];

            if (face_bounds.contains(p1))
                aabb.expand(p1);
            if (face_bounds.contains(p2))
                aabb.expand(p2);
        }

        return aabb;
    }

    BoundingBox bbox(Index index) const {
        Index iv = m_seg_index[index];
        Point center;
        Vector axes[2];
        Float lengths[2];

        bool success = intersect_cyl_plane(first_vertex(iv), first_miter_normal(iv),
                                         first_vertex(iv), tangent(iv), m_radius_per_vertex[iv] * (1-math::Epsilon<Scalar>), center, axes, lengths);
        Assert(success);

        BoundingBox result;
        axes[0] *= lengths[0]; axes[1] *= lengths[1];
        for (int i=0; i<3; ++i) {
            Float range = std::sqrt(axes[0][i]*axes[0][i] + axes[1][i]*axes[1][i]);
            result.min[i] = std::min(result.min[i], center[i]-range);
            result.max[i] = std::max(result.max[i], center[i]+range);
        }

        success = intersect_cyl_plane(second_vertex(iv), second_miter_normal(iv),
                                    second_vertex(iv), tangent(iv), m_radius_per_vertex[iv] * (1-math::Epsilon<Scalar>), center, axes, lengths);
        Assert(success);

        axes[0] *= lengths[0]; axes[1] *= lengths[1];
        for (int i=0; i<3; ++i) {
            Float range = std::sqrt(axes[0][i]*axes[0][i] + axes[1][i]*axes[1][i]);
            result.min[i] = std::min(result.min[i], center[i]-range);
            result.max[i] = std::max(result.max[i], center[i]+range);
        }
        return result;
    }

    BoundingBox bbox(Index index, const BoundingBox &box) const {
        BoundingBox base(bbox(index));
        base.clip(box);

        Index iv = m_seg_index[index];
        Point cyl_pt = first_vertex(iv);
        Vector cyl_d = tangent(iv);

        BoundingBox clipped_bbox;
        clipped_bbox.expand(intersect_cyl_face(0,
                                               Point(base.min.x(), base.min.y(), base.min.z()),
                                               Point(base.min.x(), base.max.y(), base.max.z()),
                                               cyl_pt, cyl_d, iv));

        clipped_bbox.expand(intersect_cyl_face(0,
                                               Point(base.max.x(), base.min.y(), base.min.z()),
                                               Point(base.max.x(), base.max.y(), base.max.z()),
                                               cyl_pt, cyl_d, iv));

        clipped_bbox.expand(intersect_cyl_face(1,
                                               Point(base.min.x(), base.min.y(), base.min.z()),
                                               Point(base.max.x(), base.min.y(), base.max.z()),
                                               cyl_pt, cyl_d, iv));

        clipped_bbox.expand(intersect_cyl_face(1,
                                               Point(base.min.x(), base.max.y(), base.min.z()),
                                               Point(base.max.x(), base.max.y(), base.max.z()),
                                               cyl_pt, cyl_d, iv));

        clipped_bbox.expand(intersect_cyl_face(2,
                                               Point(base.min.x(), base.min.y(), base.min.z()),
                                               Point(base.max.x(), base.max.y(), base.min.z()),
                                               cyl_pt, cyl_d, iv));

        clipped_bbox.expand(intersect_cyl_face(2,
                                               Point(base.min.x(), base.min.y(), base.max.z()),
                                               Point(base.max.x(), base.max.y(), base.max.z()),
                                               cyl_pt, cyl_d, iv));

        clipped_bbox.clip(base);

        return clipped_bbox;
    }
#else
    BoundingBox bbox(Index index) const {
        Index iv = m_seg_index[index];

        const Float cos0 = dot(first_miter_normal(iv), tangent(iv));
        const Float cos1 = dot(second_miter_normal(iv), tangent(iv));
        const Float max_inv_cos = 1.0f / (Float)std::min(cos0, cos1);
        const Float max = std::max(m_radius_per_vertex[iv], m_radius_per_vertex[iv+1]);
        const Vector expand_vec(max * max_inv_cos);

        const Point a = first_vertex(iv);
        const Point b = second_vertex(iv);

        BoundingBox box;
        box.expand((Point)(a - expand_vec));
        box.expand((Point)(a + expand_vec));
        box.expand((Point)(b - expand_vec));
        box.expand((Point)(b + expand_vec));
        return box;
    }

    BoundingBox bbox(Index index, const BoundingBox &box) const {
        BoundingBox cbox = bbox(index);
        cbox.clip(box);
        return cbox;
    }
#endif

    MTS_INLINE Size primitive_count() const {
        return (Size) m_seg_index.size();
    }
//http://lousodrome.net/blog/light/2017/01/03/intersection-of-a-ray-and-a-cone/
    MTS_INLINE std::pair<Mask, Float> intersect_prim(Index prim_index, const Ray3f &ray,
                                                 Float *cache, Mask active) const {
        ENOKI_MARK_USED(active);

        Vector axis = tangent(prim_index);

        Point ray_o(ray.o);
        Vector ray_d(ray.d);

        Point v1 = first_vertex(prim_index);
        Point v2 = second_vertex(prim_index);

        Vector rel_origin = ray_o - v1;
        Vector proj_origin = rel_origin - dot(axis, rel_origin) * axis;

        Float A, B, C;
        if(m_cylinder || std::abs(m_radius_per_vertex[prim_index] - m_radius_per_vertex[prim_index+1]) < math::Epsilon<Scalar>){
            Vector proj_direction = ray_d - dot(axis, ray_d) * axis;

            A = squared_norm(proj_direction);
            B = 2 * dot(proj_origin, proj_direction);
            C = squared_norm(proj_origin) - m_radius_per_vertex[prim_index]*m_radius_per_vertex[prim_index];
        } else{
            Point p_circle_v1 = v1 + m_radius_per_vertex[prim_index] * normalize(proj_origin);
            Point p_circle_v2 = v2 + m_radius_per_vertex[prim_index+1] * normalize(proj_origin);
            Vector normalized_edge = normalize(p_circle_v2 - p_circle_v1);

            Float sin_theta = norm(cross(normalized_edge, axis));
            Float cos_theta = dot(normalized_edge, axis);
            Float square_cos_theta = cos_theta * cos_theta;

            Float axis_direction = m_radius_per_vertex[prim_index] > m_radius_per_vertex[prim_index+1] ? 1 : -1;

            Point cone_top = v1 + (m_radius_per_vertex[prim_index] / sin_theta) * axis * axis_direction;
            Vector center_origin = ray_o - cone_top;

            Float d_dot_axis = dot(ray_d, axis); 
            Float center_origin_dot_axis = dot(center_origin, axis);

            A = d_dot_axis * d_dot_axis - square_cos_theta;
            B = 2 * (d_dot_axis * center_origin_dot_axis - dot(ray_d, center_origin) * square_cos_theta);
            C = center_origin_dot_axis * center_origin_dot_axis - dot(center_origin, center_origin) * square_cos_theta;
        }

        Float near_t, far_t, t;
        auto coeffs = math::solve_quadratic(A, B, C);
        near_t = std::get<1>(coeffs);
        far_t = std::get<2>(coeffs);

        if (!std::get<0>(coeffs))
            return std::make_pair(false, t);

        if (!(near_t <= ray.maxt && far_t >= ray.mint))
            return std::make_pair(false, t);

        Point point_near = ray_o + ray_d * near_t;
        Point point_far = ray_o + ray_d * far_t;

        Vector n1 = first_miter_normal(prim_index);
        Vector n2 = second_miter_normal(prim_index);
        Point3f p;

        if (dot(point_near - v1, n1) >= 0 && dot(point_near - v2, n2) <= 0 && near_t >= ray.mint) {
            p = Point3f(ray_o + ray_d * near_t);
            t = (Float) near_t;
        } else if (dot(point_far - v1, n1) >= 0 && dot(point_far - v2, n2) <= 0) {
            if (far_t > ray.maxt)
                return std::make_pair(false, t);
            p = Point3f(ray_o + ray_d * far_t);
            t = (Float) far_t;
        } else {
            return std::make_pair(false, t);
        }

        if (cache) {
            cache[1] = prim_index;
            cache[2] = p.x();
            cache[3] = p.y();
            cache[4] = p.z();
        }

        return std::make_pair(true, t);
    }

    /* Some utility functions */
    MTS_INLINE Point first_vertex(Index iv) const { return m_vertices[iv]; }
    MTS_INLINE Point second_vertex(Index iv) const { return m_vertices[iv+1]; }
    MTS_INLINE Point prev_vertex(Index iv) const { return m_vertices[iv-1]; }
    MTS_INLINE Point next_vertex(Index iv) const { return m_vertices[iv+2]; }

    MTS_INLINE bool prev_segment_exists(Index iv) const { return !m_vertex_starts_fiber[iv]; }
    MTS_INLINE bool next_segment_exists(Index iv) const { return !m_vertex_starts_fiber[iv+2]; }

    MTS_INLINE Vector tangent(Index iv) const { return normalize(second_vertex(iv) - first_vertex(iv)); }
    MTS_INLINE Vector prev_tangent(Index iv) const { return normalize(first_vertex(iv) - prev_vertex(iv)); }
    MTS_INLINE Vector next_tangent(Index iv) const { return normalize(next_vertex(iv) - second_vertex(iv)); }

    MTS_INLINE Vector first_miter_normal(Index iv) const {
        if (prev_segment_exists(iv))
            return normalize(prev_tangent(iv) + tangent(iv));
        else
            return tangent(iv);
    }

    MTS_INLINE Vector second_miter_normal(Index iv) const {
        if (next_segment_exists(iv))
            return normalize(tangent(iv) + next_tangent(iv));
        else
            return tangent(iv);
    }

    MTS_DECLARE_CLASS()
private:
    std::vector<Point> m_vertices;
    std::vector<bool> m_vertex_starts_fiber;
    std::vector<Index> m_seg_index;
    std::vector<Float> m_radius_per_vertex;
    Size m_segment_count;
    Size m_hair_count;
    bool m_cylinder;
};

MTS_IMPLEMENT_CLASS_VARIANT(HairKDTree, TShapeKDTree)

template <typename Float, typename Spectrum>
class HairShape final : public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape)
    MTS_IMPORT_TYPES(HairKDTree)

    using typename Base::ScalarSize;
    using Index = typename HairKDTree::Index;
    using ScalarIndex = typename Base::ScalarIndex;
    using PCG32 = mitsuba::PCG32<UInt32>;

    HairShape(const Properties &props) : Base(props) {
        FileResolver *fs = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(props.string("filename"));

        Float default_radius = props.float_("radius", 0.025f);

        Float angle_threshold = props.float_("angle_threshold", 1.0f) * (Float)(M_PI / 180.0);
        Float dp_thresh = std::cos(angle_threshold);

        Float reduction = props.float_("reduction", 0);
        if (reduction < 0 || reduction >= 1){
            Log(LogLevel::Error, "The 'reduction' parameter must have a value in [0, 1)!");
        } else if (reduction > 0){
            Float correction = 1.0f / (1-reduction);
            Log(LogLevel::Debug, "Reducing the amount of geometry by %.2f%%, scaling radii by %f.");
            default_radius *= correction;
        }
        Float radius = default_radius;
        std::unique_ptr<PCG32> rng = std::make_unique<PCG32>(); //TODO: Is it the correct way to do it?

        ScalarTransform4f object_to_world = props.transform("to_world");

        Log(LogLevel::Info, "Loading hair geometry from \"%s\" ..", file_path.filename().string().c_str());
        Timer* timer = new Timer();

        ref<FileStream> binary_stream = new FileStream(file_path, FileStream::ERead);
        binary_stream->set_byte_order(Stream::ELittleEndian);

        const char *binary_header = "BINARY_HAIR";
        char temp[11];

        bool binary_format = true;
        binary_stream->read(temp, 11);
        if (memcmp(temp, binary_header, 11) != 0)
            binary_format = false;

        std::vector<ScalarPoint3f> vertices;
        std::vector<Float> radius_per_vertex;
        Float max_radius = 0.f;
        std::vector<bool> vertex_starts_fiber;
        ScalarVector3f tangent(0.0f);
        size_t n_degenerate = 0, n_skipped = 0;
        ScalarPoint3f p, last_p(0.0f);
        bool ignore = false;

        if (binary_format) {
            unsigned int vertex_count;
            binary_stream->read((void *)&vertex_count, sizeof(vertex_count));
            Log(LogLevel::Info, "Loading %zd hair vertices ..", vertex_count);
            vertices.reserve(vertex_count);
            vertex_starts_fiber.reserve(vertex_count);

            bool new_fiber = true;
            size_t vertices_read = 0;

            while (vertices_read != vertex_count) {
                Float value;
                binary_stream->read((void*)&value, sizeof(value));
                if (std::isinf(value)) {
                    binary_stream->read((void*)&p.x(), sizeof(p.x()));
                    binary_stream->read((void*)&p.y(), sizeof(p.y()));
                    binary_stream->read((void*)&p.z(), sizeof(p.z()));
                    new_fiber = true;
                    if (reduction > 0)
                        ignore = rng->next_float32() < reduction;
                } else {
                    p[0] = value;
                    binary_stream->read((void*)&p.y(), sizeof(p.y()));
                    binary_stream->read((void*)&p.z(), sizeof(p.z()));
                }

                p = object_to_world * p;
                vertices_read++;

                if (ignore) {
                    ++n_skipped;
                } else if (new_fiber) {
                    vertices.push_back(p);
                    radius_per_vertex.push_back(default_radius);
                    vertex_starts_fiber.push_back(new_fiber);
                    last_p = p;
                    tangent = ScalarVector3f(0.0f);
                } else if (p != last_p) {
                    Mask is_zero = tangent == 0.0f;
                    if (all(is_zero)) {
                        vertices.push_back(p);
                        radius_per_vertex.push_back(default_radius);
                        vertex_starts_fiber.push_back(new_fiber);
                        tangent = normalize(p - last_p);
                        last_p = p;
                    } else {
                        ScalarVector3f next_tangent = normalize(p - last_p);
                        if (dot(next_tangent, tangent) > dp_thresh) {
                            tangent = normalize(p - vertices[vertices.size()-2]);
                            vertices[vertices.size()-1] = p;
                            radius_per_vertex[vertices.size()-1] = default_radius;
                            ++n_skipped;
                        } else {
                            vertices.push_back(p);
                            radius_per_vertex.push_back(default_radius);
                            vertex_starts_fiber.push_back(new_fiber);
                            tangent = next_tangent;
                        }
                        last_p = p;
                    }
                } else {
                    n_degenerate++;
                }
                new_fiber = false;
            }
        } else {
            std::string line;
            bool new_fiber = true;

            std::ifstream is(file_path);
            if (is.fail())
                Log(LogLevel::Error, "Could not open \"%s\"!", file_path.string().c_str());
            while (is.good()) {
                std::getline(is, line);
                if (line.length() > 0 && line[0] == '#') {
                    new_fiber = true;
                    continue;
                }
                std::istringstream iss(line);
                if(props.has_property("radius") || props.has_property("base")){
                    iss >> p.x() >> p.y() >> p.z();
                    radius = default_radius;
                } else{
                    iss >> p.x() >> p.y() >> p.z() >> radius;
                }
                if (!iss.fail()) {
                    radius *= norm((object_to_world * ScalarVector3f(0.f, 0.f, 1.f)));
                    if(radius >= max_radius){
                        max_radius = radius;
                    }
                    p = object_to_world * p;
                    if (ignore) {
                        // Do nothing
                        ++n_skipped;
                    } else if (new_fiber) {
                        vertices.push_back(p);
                        radius_per_vertex.push_back(radius);
                        vertex_starts_fiber.push_back(new_fiber);
                        last_p = p;
                        tangent = ScalarVector3f(0.0f);
                    } else if (p != last_p) {
                        Mask is_zero = tangent == 0.0f;
                        if (all(is_zero)) {
                            vertices.push_back(p);
                            radius_per_vertex.push_back(radius);
                            vertex_starts_fiber.push_back(new_fiber);
                            tangent = normalize(p - last_p);
                            last_p = p;
                        } else {
                            ScalarVector3f next_tangent = normalize(p - last_p);
                            if (dot(next_tangent, tangent) > dp_thresh) {
                                tangent = normalize(p - vertices[vertices.size()-2]);
                                vertices[vertices.size()-1] = p;
                                radius_per_vertex[vertices.size()-1] = radius;
                                ++n_skipped;
                            } else {
                                vertices.push_back(p);
                                radius_per_vertex.push_back(radius);
                                vertex_starts_fiber.push_back(new_fiber);
                                tangent = next_tangent;
                            }
                            last_p = p;
                        }
                    } else {
                        n_degenerate++;
                    }
                    new_fiber = false;
                } else {
                    new_fiber = true;
                    if (reduction > 0)
                        ignore = rng->next_float32() < reduction;
                }
            }
        }

        if(props.has_property("base") && props.has_property("tip")){
            Float size = radius_per_vertex.size();
            Float base = props.float_("base");
            Float incr = (base - props.float_("tip")) / size;
            radius_per_vertex[0] = base;
            for(size_t i = 1; i < size; ++i){
                radius_per_vertex[i] = base - (i+1) * incr;
            }
        }

        if (n_degenerate > 0)
            Log(LogLevel::Info, "Encountered %zd degenerate segments!", n_degenerate);

        if (n_skipped > 0)
            Log(LogLevel::Info, "Skipped %zd segments.", n_skipped);

        Log(LogLevel::Info, "Done (took %i ms)", timer->value());

        vertex_starts_fiber.push_back(true);

        m_kdtree = new HairKDTree(props, vertices, vertex_starts_fiber, radius_per_vertex, max_radius, props.has_property("radius"));
    }

    const std::vector<ScalarPoint3f> &vertices() const{
        return m_kdtree->vertices();
    }

    const std::vector<bool> &start_fiber() const{
        return m_kdtree->start_fiber();
    }

    std::pair<Mask, Float> ray_intersect(const Ray3f &ray, Float *cache, Mask active = true) const override{
        return m_kdtree->template ray_intersect<false>(ray, cache, active);
    }

    void fill_surface_interaction(const Ray3f &ray, const Float *cache, SurfaceInteraction3f &si, Mask active = true) const override{
        ENOKI_MARK_USED(active);

        si.uv = Point2f(0.f,0.f);
        si.dp_du = ScalarVector3f(0.f);
        si.dp_dv = ScalarVector3f(0.f);

        Index iv = cache[1];
        si.p[0] = cache[2];
        si.p[1] = cache[3];
        si.p[2] = cache[4];


        const Vector3f axis = m_kdtree->tangent(iv);
        si.shape = this;

        const Vector3f rel_hit_point = si.p - m_kdtree->first_vertex(iv);
        auto n = normalize(rel_hit_point - dot(axis, rel_hit_point) * axis);
        si.n = n;

        Frame3f frame = Frame3f(n);
        frame.s = axis;
        frame.t = cross(frame.n, frame.s);

        const Vector3f local = frame.to_local(rel_hit_point);
        si.p += n * (m_kdtree->radius(iv) - std::sqrt(local.y()*local.y()+local.z()*local.z()));

        si.sh_frame.n = n;
        auto uv = coordinate_system(si.sh_frame.n);
        si.dp_du = uv.first;
        si.dp_dv = uv.second;
        si.instance = this;
        si.time = ray.time;
    }

    ScalarBoundingBox3f bbox() const override{
        return m_kdtree->bbox();
    }

    ScalarBoundingBox3f bbox(ScalarIndex index) const override {
        return m_kdtree->bbox(index);
    }

    ScalarBoundingBox3f bbox(ScalarIndex index, const ScalarBoundingBox3f &clip) const override{
        return m_kdtree->bbox(index, clip);
    }

    Float surface_area() const override{
        Log(LogLevel::Error, "HairShape::getSurfaceArea(): Not implemented.");
        return -1;
    }

    ScalarSize primitive_count() const override{
        return m_kdtree->primitive_count();
    }

    ScalarSize effective_primitive_count() const override{
        return m_kdtree->primitive_count();
    }

    std::string to_string() const override{
        std::ostringstream oss;
        oss << "Hair[" << std::endl
            << "   num_vertices = " << m_kdtree->vertex_count() << ","
            << "   num_segments = " << m_kdtree->segment_count() << ","
            << "   num_hairs = " << m_kdtree->hair_count() << ","
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()

private:
    ref<HairKDTree> m_kdtree;
};

MTS_IMPLEMENT_CLASS_VARIANT(HairShape, Shape)
MTS_EXPORT_PLUGIN(HairShape, "Hair Shape")
NAMESPACE_END(mitsuba)