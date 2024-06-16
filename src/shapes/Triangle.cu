/*
** RAYCASTING LIBRARY
** Triangle.cu
** Created by marton on 15/06/24.
*/

#include "shapes/Triangle.hpp"

namespace rcr {

    __device__ rcr::vec3<float> getHitPos(rcr::ray ray, float t) {
        vec3<float> temp_res, res;

        temp_res={};
        res={};

        ray.direction.multSingle(t, &temp_res);
        ray.origin.sumSingle(&temp_res, &res);
        return res;
    }

    __device__ void Triangle::move(vec3<float> p1, vec3<float> p2, vec3<float> p3) {
        p1_ = p1;
        p2_ = p2;
        p3_ = p3;
    }

    __device__ void Triangle::setID(unsigned int id) {
        id_ = id;
    }

    __device__ hitPos Triangle::hit(ray ray) const {
        vec3<float> e1, e2, h, s, q;
        float a, f, u, v, t;

        e1={}, e2={}, h={}, s={}, q={};
        f=0, f=0, u=0, v=0, t=0;

        p2_.diffSingle(&p1_, &e1);
        p3_.diffSingle(&p1_, &e2);

        ray.direction.crossSingle(&e2, &h);

        a = e1.dotSingle(&h);  // This function is blocking

        if (a > -0.00001 && a < 0.00001)
            return {false, {}};

        f = 1 / a;

        ray.origin.diffSingle(&p1_, &s);

        u = f * s.dotSingle(&h);  // This function is blocking

        if (u < 0.0 || u > 1.0)
            return {false, {}};

        s.crossSingle(&e1, &q);

        v = f * ray.direction.dotSingle(&q);  // This function is blocking

        if (v < 0.0 || u + v > 1.0)
            return {false, {}};

        t = f * e2.dotSingle(&q);

        if (t > 0.00001)
            return {true, {getHitPos(ray, t)}};

        return {false, {}};
    }

} // rcr