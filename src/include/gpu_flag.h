// Shared opt-in GPU switch. main() sets it from the --gpu flag before any
// net loads; every option site consults it instead of a hardcoded value.
#ifndef GPU_FLAG_H
#define GPU_FLAG_H

extern bool g_use_gpu;

#endif
