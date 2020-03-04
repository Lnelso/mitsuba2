//
// Created by Lionel Pellier on 2020-02-25.
//

#include <mitsuba/render/shape.h>

NAMESPACE_BEGIN(mitsuba)

//class HairKDTree; //TODO: check how predeclaration are done in v2

template <typename Point_, typename Float, typename Spectrum>
class MTS_EXPORT_RENDER HairShape : public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_TYPES()

    using Point = Point_;
    using Vector      = typename BoundingBox::Vector;
    using Scalar      = value_t<Vector>;

    /*using typename Base::ScalarIndex;
    using typename Base::ScalarSize;*/

    HairShape(const Properties &props);

    //Mitsuba v1
    //void serialize(Stream *stream, InstanceManager *manager) const; //TODO: find what this is about
    //ref<TriMesh> create_tri_mesh(); //TODO: find what this is about
    //End of mitsuba v1


    const std::vector<Point> &get_vertices() const;

    const std::vector<bool> &get_start_fiber() const;

    std::pair<Mask, Float> ray_intersect(const Ray3f &ray, Float *cache, Mask active = true) const;

    SurfaceInteraction3f ray_intersect(const Ray3f &ray, Mask active = true) const;

    void fill_surface_interaction(const Ray3f &ray, const Float *cache, SurfaceInteraction3f &si, Mask active = true) const;

    //const ShapeKDTree<ScalarBoundingBox3f> *get_kd_tree; //TODO: not sure about this one

    ScalarBoundingBox3f bbox();

    Float get_surface_area() const;

    Scalar get_primitive_count() const;

    Scalar get_effective_primitive_count() const;

    std::string to_string() const;


    MTS_DECLARE_CLASS()
private:
    ref<HairKDTree> m_kdtree;
};

MTS_EXTERN_CLASS_RENDER(HairShape)
NAMESPACE_END(mitsuba)
