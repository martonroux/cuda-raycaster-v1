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

        ray.direction.mult(t, &temp_res);
        ray.origin.sum(&temp_res, &res);
        return res;
    }

    __device__ hitPos Triangle::hit(ray ray) const {
        vec3<float> e1, e2, h, s, q;
        float a, f, u, v, t;

        e1={}, e2={}, h={}, s={}, q={};
        f=0, f=0, u=0, v=0, t=0;

        p2_.diff(&p1_, &e1);
        p3_.diff(&p1_, &e2);

        ray.direction.cross(&e2, &h);

        a = e1.dot(&h);  // This function is blocking

        if (a > -0.00001 && a < 0.00001)
            return {false, {}};

        f = 1 / a;

        ray.origin.diff(&p1_, &s);

        u = f * s.dot(&h);  // This function is blocking

        if (u < 0.0 || u > 1.0)
            return {false, {}};

        s.cross(&e1, &q);

        v = f * ray.direction.dot(&q);  // This function is blocking

        if (v < 0.0 || u + v > 1.0)
            return {false, {}};

        t = f * e2.dot(&q);

        if (t > 0.00001)
            return {true, {getHitPos(ray, t)}};

        return {false, {}};
    }

} // rcr