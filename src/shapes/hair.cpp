//
// Created by Lionel Pellier on 2020-02-25.
//

#include <iostream>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/random.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/timer.h>
//#include <mitsuba/render/hair.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/kdtree.h>
#include <string>
#include <fstream>

#define MTS_HAIR_USE_FANCY_CLIPPING 1

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class HairKDTree: ShapeKDTree<Float, Spectrum> {
public:
    MTS_IMPORT_TYPES(Shape, Mesh)
    MTS_IMPORT_BASE(ShapeKDTree, set_stop_primitives, set_exact_primitive_threshold,
                    set_clip_primitives, set_retract_bad_splits, build, m_index_count, m_indices)

    using Point = typename Base::Base::Point;
    using Vector = typename Base::Base::Point;
    using BoundingBox = typename Base::Base::BoundingBox;
    using Index = typename Base::Base::Index;
    using Size = typename Base::Base::Size;
    using Scalar = typename Base::Base::Scalar;
    using SurfaceAreaHeuristic3f = SurfaceAreaHeuristic3<ScalarFloat>;

    HairKDTree(std::vector<Point> &vertices,
               std::vector<bool> &vertex_starts_fiber, Float radius)
            : m_radius(radius) {
        m_vertices.swap(vertices);
        m_vertex_starts_fiber.swap(vertex_starts_fiber);
        m_hair_count = 0;

        m_seg_index.reserve(m_vertices.size());
        for (size_t i=0; i<m_vertices.size()-1; i++) {
            if (m_vertex_starts_fiber[i])
                m_hair_count++;
            if (!m_vertex_starts_fiber[i+1])
                m_seg_index.push_back((Index) i);
        }
        m_segment_count = m_seg_index.size();

        //TODO: logging

        set_stop_primitives(1);

        SurfaceAreaHeuristic3f::m_traversal_cost = 10;
        SurfaceAreaHeuristic3f::m_query_cost = 15;
        SurfaceAreaHeuristic3f::m_empty_space_bonus = 0.9f;

        set_exact_primitive_threshold(16384);
        set_clip_primitives(true);
        set_retract_bad_splits(true);

        build();

        //TODO: logging

        for (Size i=0; i<m_index_count; ++i)
            m_indices[i] = m_seg_index[m_indices[i]];

        std::vector<Index>().swap(m_seg_index);

    }

    MTS_INLINE const std::vector<Point> &get_vertices() const {
        return m_vertices;
    }

    MTS_INLINE const std::vector<bool> &get_start_fiber() const {
        return m_vertex_starts_fiber;
    }

    MTS_INLINE Float get_radius() const {
        return m_radius;
    }

    MTS_INLINE Size get_segment_count() const {
        return m_segment_count;
    }

    MTS_INLINE Size get_hair_count() const {
        return m_hair_count;
    }

    MTS_INLINE Size get_vertex_count() const {
        return m_vertices.size();
    }

    MTS_INLINE std::pair<Mask, Float> ray_intersect(const Ray3f &ray, Float *cache, Mask active) const {
        return Base::ray_intersect<true>(ray, cache, active);
    }

    MTS_INLINE std::pair<bool, Float> ray_intersect(const Ray3f &ray, Mask active) const {
        return Base::ray_intersect<false>(ray, NULL);
        //TODO: either should return a surfaceInteraction or HairShape should take care of it with the tuple
    }

#if MTS_HAIR_USE_FANCY_CLIPPING == 1
    bool intersect_cyl_plane(Point plane_pt, Normal3f plane_nrml,
            Point cyl_pt, Vector cyl_d, Float radius, Point &center,
            Vector *axes, Float *lengths) const {

        if (abs_dot(plane_nrml, cyl_d) < math::Epsilon<Scalar>) //TODO: absDot?
            return false;

        Assert(std::abs(plane_nrml.length()-1) < math::Epsilon<Scalar>);
        Vector B, A = cyl_d - dot(cyl_d, plane_nrml)*plane_nrml;

        Float length = A.length();
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
            const Point &cyl_pt, const Vector &cyl_d) const {
        int axis1 = (axis + 1) % 3;
        int axis2 = (axis + 2) % 3;

        Normal3f plane_nrml(0.0f);
        plane_nrml[axis] = 1;

        Point ellipse_center;
        Vector ellipse_axes[2];
        Float ellipse_lengths[2];

        BoundingBox aabb;
        if (!intersect_cyl_plane(min, plane_nrml, cyl_pt, cyl_d, m_radius * (1 + math::Epsilon<Scalar>),
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

            Point p1l(
                    dot(p1 - ellipse_center, ellipse_axes[0]) / ellipse_lengths[0],
                    dot(p1 - ellipse_center, ellipse_axes[1]) / ellipse_lengths[1]);
            Point p2l(
                    dot(p2 - ellipse_center, ellipse_axes[0]) / ellipse_lengths[0],
                    dot(p2 - ellipse_center, ellipse_axes[1]) / ellipse_lengths[1]);

            Vector rel = p2l-p1l;
            Float A = dot(rel, rel);
            Float B = 2*dot(Vector(p1l), rel);
            Float C = dot(Vector(p1l), Vector(p1l))-1;

            auto coefs = math::solve_quadratic(A, B, C);
            Float x0 = std::get<1>(coefs);
            Float x1 = std::get<2>(coefs);
            if (std::get<0>(coefs)) { //TODO: Check how to use Mask as a bool
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
                                         first_vertex(iv), tangent(iv), m_radius * (1-math::Epsilon<Scalar>), center, axes, lengths);
        Assert(success);

        BoundingBox result;
        axes[0] *= lengths[0]; axes[1] *= lengths[1];
        for (int i=0; i<3; ++i) {
            Float range = std::sqrt(axes[0][i]*axes[0][i] + axes[1][i]*axes[1][i]);
            result.min[i] = std::min(result.min[i], center[i]-range);
            result.max[i] = std::max(result.max[i], center[i]+range);
        }

        success = intersec_cyl_plane(second_vertex(iv), second_miter_normal(iv),
                                    second_vertex(iv), tangent(iv), m_radius * (1-math::Epsilon<Scalar>), center, axes, lengths);
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
                                               Point(base.min.x, base.min.y, base.min.z),
                                               Point(base.min.x, base.max.y, base.max.z),
                                               cyl_pt, cyl_d));

        clipped_bbox.expand(intersect_cyl_face(0,
                                               Point(base.max.x, base.min.y, base.min.z),
                                               Point(base.max.x, base.max.y, base.max.z),
                                               cyl_pt, cyl_d));

        clipped_bbox.expand(intersect_cyl_face(1,
                                               Point(base.min.x, base.min.y, base.min.z),
                                               Point(base.max.x, base.min.y, base.max.z),
                                               cyl_pt, cyl_d));

        clipped_bbox.expand(intersect_cyl_face(1,
                                               Point(base.min.x, base.max.y, base.min.z),
                                               Point(base.max.x, base.max.y, base.max.z),
                                               cyl_pt, cyl_d));

        clipped_bbox.expand(intersect_cyl_face(2,
                                               Point(base.min.x, base.min.y, base.min.z),
                                               Point(base.max.x, base.max.y, base.min.z),
                                               cyl_pt, cyl_d));

        clipped_bbox.expand(intersect_cyl_face(2,
                                               Point(base.min.x, base.min.y, base.max.z),
                                               Point(base.max.x, base.max.y, base.max.z),
                                               cyl_pt, cyl_d));

        clipped_bbox.clip(base);

        return clipped_bbox;
    }
#else
    BoundingBox bbox(Index index) const {
        Index iv = m_seg_index[index];

        const Float cos0 = dot(first_miter_normal(iv), tangent(iv));
        const Float cos1 = dot(second_miter_normal(iv), tangent(iv));
        const Float max_inv_cos = 1.0 / std::min(cos0, cos1);
        const Vector expandVec(m_radius * maxInvCos);

        const Point a = first_vertex(iv);
        const Point b = second_vertex(iv);

        BoundingBox box;
        box.expand(a - expand_vec);
        box.expand(a + expand_vec);
        box.expand(b - expand_vec);
        box.expand(b + expand_vec);
        return box;
    }

    BoundingBox bbox(Indexindex, const BoundingBox &box) const {
        BoundingBox cbox(bbox(index));
        cbox.clip(box);
        return cbox;
    }
#endif

    MTS_INLINE Size get_primitive_count() const {
        return (Size) m_seg_index.size();
    }

    struct IntersectionStorage {
        Index iv;
        Point p;
    };

    MTS_INLINE std::pair<Mask, Float> intersect_prim(Index prim_index, const Ray3f &ray,
                                                 Float *cache, Mask active) const {
        Vector axis = tangent_double(prim_index);

        Point ray_o(ray.o);
        Vector ray_d(ray.d);
        Point v1 = first_vertex_double(prim_index);

        Vector rel_origin = ray_o - v1;
        Vector proj_origin = rel_origin - dot(axis, rel_origin) * axis;
        Vector proj_direction = ray_d - dot(axis, ray_d) * axis;

        // Quadratic to intersect circle in projection
        const double A = proj_direction.lengthSquared(); //TODO: find equivalent for lengthSquared
        const double B = 2 * dot(proj_origin, proj_direction);
        const double C = proj_origin.lengthSquared() - m_radius*m_radius;

        double near_t, far_t;
        auto coeffs = math::solve_quadratic(A, B, C);
        near_t = std::get<1>(coeffs);
        far_t = std::get<2>(coeffs);

        if (!std::get<0>(coeffs)) //TODO: find how to get bool from Mask
            return false;

        if (!(near_t <= ray.maxt && far_t >= ray.mint))
            return false;

        Point point_near = ray_o + ray_d * near_t;
        Point point_far = ray_o + ray_d * far_t;

        Vector n1 = first_miter_normal_double(prim_index);
        Vector n2 = second_miter_normal_double(prim_index);
        Point v2 = second_vertex_double(prim_index);
        IntersectionStorage *storage = static_cast<IntersectionStorage *>(cache);
        Point p;

        if (dot(point_near - v1, n1) >= 0 &&
            dot(point_near - v2, n2) <= 0 &&
            near_t >= ray.mint) {
            p = Point(ray_o + ray_d * near_t);
            t = (Float) near_t;
        } else if (dot(point_far - v1, n1) >= 0 &&
                   dot(point_far - v2, n2) <= 0) {
            if (far_t > ray.maxt)
                return false;
            p = Point(ray_o + ray_d * far_t);
            t = (Float) far_t;
        } else {
            return false;
        }

        if (storage) {
            storage->iv = prim_index;
            storage-> p = p;
        }

        return true;
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

private:
    std::vector<Point> m_vertices;
    std::vector<bool> m_vertex_starts_fiber;
    std::vector<Index> m_seg_index;
    Float m_radius;
    Size m_segment_count;
    Size m_hair_count;
};


template <typename Float, typename Spectrum>
class HairShape final : public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape)
    MTS_IMPORT_TYPES()

    using typename Base::ScalarSize;
    using PCG32 = mitsuba::PCG32<UInt32>;

    HairShape(const Properties &props) : Base(props) {
        FileResolver *fs = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(props.string("filename"));

        Float radius = props.float_("radius", 0.025f);

        Float angle_threshold = props.float_("angle_threshold", 1.0f);
        Float dp_thresh = std::cos(angle_threshold);

        Float reduction = props.float_("reduction", 0);
        if (reduction < 0 || reduction >= 1){
            Log(LogLevel::Error, "The 'reduction' parameter must have a value in [0, 1)!");
        } else if (reduction > 0){
            Float correction = 1.0f / (1-reduction);
            Log(LogLevel::Debug, "Reducing the amount of geometry by %.2f%%, scaling radii by %f.");
            radius *= correction;
        }
        std::unique_ptr<PCG32> rng = std::make_unique<PCG32>(); //TODO: Is it the correct way to do it?

        ScalarTransform4f object_to_world = props.transform("to_world");
        radius *= norm((object_to_world * ScalarVector3f(0.f, 0.f, 1.f)));

        Log(LogLevel::Info, "Loading hair geometry from \"%s\" ..", file_path.filename().string().c_str());
        ref<Timer> timer = new Timer();

        ref<FileStream> binary_stream = new FileStream(file_path, FileStream::ERead);
        binary_stream->set_byte_order(Stream::ELittleEndian);

        const char *binary_header = "BINARY_HAIR";
        char temp[11];

        bool binary_format = true;
        binary_stream->read(temp, 11);
        if (memcmp(temp, binary_header, 11) != 0)
            binary_format = false;

        std::vector<ScalarPoint3f> vertices;
        std::vector<bool> vertex_starts_fiber;
        ScalarVector3f tangent(0.0f);
        size_t n_degenerate = 0, n_skipped = 0; //TODO: may need to convert to ScalarSize
        ScalarPoint3f p, last_p(0.0f);
        bool ignore = false;

        if (binary_format) {
            unsigned int vertex_count; //TODO: or size_t or ScalarSize?
            binary_stream->read((void *)&vertex_count, sizeof(vertex_count)); //TODO: Error prone?
            Log(LogLevel::Info, "Loading %zd hair vertices ..", vertex_count);
            vertices.reserve(vertex_count);
            vertex_starts_fiber.reserve(vertex_count);

            bool new_fiber = true;
            size_t vertices_read = 0; //TODO: scalarSize?

            while (vertices_read != vertex_count) {
                Float value;
                binary_stream->read((void*)&value, sizeof(value));
                if (std::isinf(value)) {
                    binary_stream->read((void*)&p.x, sizeof(p.x));
                    binary_stream->read((void*)&p.y, sizeof(p.y));
                    binary_stream->read((void*)&p.z, sizeof(p.z));
                    new_fiber = true;
                    if (reduction > 0)
                        ignore = rng->next_float32() < reduction; //TODO: may need a mask on the next_float32 call
                } else {
                    p.x = value;
                    binary_stream->read((void*)&p.y, sizeof(p.y));
                    binary_stream->read((void*)&p.z, sizeof(p.z));
                }

                p = object_to_world * p;
                vertices_read++;

                if (ignore) {
                    ++n_skipped;
                } else if (new_fiber) {
                    vertices.push_back(p);
                    vertex_starts_fiber.push_back(new_fiber);
                    last_p = p;
                    tangent = ScalarVector3f(0.0f);
                } else if (p != last_p) {
                    if (tangent.zero_()) { //TODO: equivalent of isZero()?
                        vertices.push_back(p);
                        vertex_starts_fiber.push_back(new_fiber);
                        tangent = normalize(p - last_p);
                        last_p = p;
                    } else {
                        ScalarVector3f next_tangent = normalize(p - last_p);
                        if (dot(next_tangent, tangent) > dp_thresh) {
                            tangent = normalize(p - vertices[vertices.size()-2]);
                            vertices[vertices.size()-1] = p;
                            ++n_skipped;
                        } else {
                            vertices.push_back(p);
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
                iss >> p.x >> p.y >> p.z;
                if (!iss.fail()) {
                    p = object_to_world * p;
                    if (ignore) {
                        // Do nothing
                        ++n_skipped;
                    } else if (new_fiber) {
                        vertices.push_back(p);
                        vertex_starts_fiber.push_back(new_fiber);
                        last_p = p;
                        tangent = ScalarVector3f(0.0f);
                    } else if (p != last_p) {
                        if (tangent.zero_()) { //TODO: equivalent of isZero()?
                            vertices.push_back(p);
                            vertex_starts_fiber.push_back(new_fiber);
                            tangent = normalize(p - last_p);
                            last_p = p;
                        } else {
                            ScalarVector3f next_tangent = normalize(p - last_p);
                            if (dot(next_tangent, tangent) > dp_thresh) {
                                tangent = normalize(p - vertices[vertices.size()-2]);
                                vertices[vertices.size()-1] = p;
                                ++n_skipped;
                            } else {
                                vertices.push_back(p);
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

        if (n_degenerate > 0)
            Log(LogLevel::Info, "Encountered %zd degenerate segments!", n_degenerate);

        if (n_skipped > 0)
            Log(LogLevel::Info, "Skipped %zd segments.", n_skipped);

        Log(LogLevel::Info, "Done (took %i ms)", timer->value());

        vertex_starts_fiber.push_back(true);

        m_kdtree = new HairKDTree<Float, Spectrum>(vertices, vertex_starts_fiber, radius); //TODO: implement HairKDTree

    }

    const std::vector<ScalarPoint3f> &get_vertices() const{
        return m_kdtree->get_vertices();
    }

    const std::vector<bool> &get_start_fiber() const{
        return m_kdtree->get_start_fiber();
    }

    std::pair<Mask, Float> ray_intersect(const Ray3f &ray, Float *cache, Mask active = true) const{
        return m_kdtree->ray_intersect(ray, cache, active);
    }

    SurfaceInteraction3f ray_intersect(const Ray3f &ray, Mask active = true) const{
        return m_kdtree->ray_intersect(ray, active);
    }

    void fill_surface_interaction(const Ray3f &ray, const Float *cache, SurfaceInteraction3f &si, Mask active = true) const{
        si.uv = Point2f(0.f,0.f);
        si.dp_du = ScalarVector3f(0.f);
        si.dp_dv = ScalarVector3f(0.f);

        const typename HairKDTree<Float, Spectrum>::IntersectionStorage *storage = static_cast<const HairKDTree<Float, Spectrum>::IntersectionStorage *>(cache);
        HairKDTree<Float, Spectrum>::Index iv = storage->iv;
        si.p = storage->p;

        const Vector axis = m_kdtree->tangent(iv); //TODO: should be ScalarVector3f ?
        si.shape = this;

        const Vector rel_hit_point = si.p - m_kdtree->first_vertex(iv);
        si.n = Normal(normalize(rel_hit_point - dot(axis, rel_hit_point) * axis));

        const Vector local = si.to_local(rel_hit_point);
        si.p += si.n * (m_kdtree->get_radius() - std::sqrt(local.y*local.y+local.z*local.z));

        si.sh_frame.n = si.n;
        auto uv = coordinate_system(si.sh_frame.n);
        si.dp_du = uv.first;
        si.dp_dv = uv.second;
        si.instance = this;
        si.time = ray.time;
    }

    /*const HairShape::TShapeKDTree<ScalarBoundingBox3f> *get_kd_tree(){ //TODO: not sure about this one
        return m_kdtree.get();
    }*/

    /*ScalarBoundingBox3f bbox(){
        return m_kdtree->get_aabb();
    }*/

    Float get_surface_area() const{
        Log(LogLevel::Error, "HairShape::getSurfaceArea(): Not implemented.");
        return -1;
    }

    ScalarSize get_primitive_count() const{
        return m_kdtree->get_hair_count();
    }

    ScalarSize get_effective_primitive_count() const{
        return m_kdtree->get_hair_count();
    }

    std::string to_string() const{
        std::ostringstream oss;
        oss << "Hair[" << std::endl
            << "   num_vertices = " << m_kdtree->get_vertex_count() << ","
            << "   num_segments = " << m_kdtree->get_segment_count() << ","
            << "   num_hairs = " << m_kdtree->get_hair_count() << ","
            << "   radius = " << m_kdtree->get_radius()
            << "]";
        return oss.str();
    }

private:
    ref<HairKDTree<Float, Spectrum>> m_kdtree;
};

MTS_IMPLEMENT_CLASS_VARIANT(HairShape, Shape)
MTS_EXPORT_PLUGIN(HairShape, "Hair intersection primitive");
NAMESPACE_END(mitsuba)