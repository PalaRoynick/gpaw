#pragma once
#include <memory>

class LFCBuilder;
class LFCGpu;

class LFCBuilderGpu
{
public:
    static std::unique_ptr<LFCGpu> build(LFCBuilder&& cpu_builder);
};
