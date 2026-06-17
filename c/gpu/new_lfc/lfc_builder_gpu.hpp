#pragma once
#include <memory>

class LFCBuilder;
class LFCGpu;
class SplinePoolGpu;

class LFCBuilderGpu
{
public:
    static std::unique_ptr<LFCGpu> build(LFCBuilder&& cpu_builder, std::shared_ptr<const SplinePoolGpu> spline_pool);
};
