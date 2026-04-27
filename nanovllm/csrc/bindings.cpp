#include <torch/extension.h>
#include <torch/library.h>

#include <cstdint>
#include <vector>

torch::Tensor sample(torch::Tensor logits, torch::Tensor temperatures,
                     int64_t seed);
torch::Tensor rms_norm(torch::Tensor x, torch::Tensor weight, double eps);
std::vector<torch::Tensor> add_rms_norm(torch::Tensor x, torch::Tensor residual,
                                        torch::Tensor weight, double eps);
std::vector<torch::Tensor> rotary_embedding(torch::Tensor positions,
                                            torch::Tensor query,
                                            torch::Tensor key,
                                            torch::Tensor cos_sin_cache);
void store_kvcache(torch::Tensor key, torch::Tensor value,
                   torch::Tensor k_cache, torch::Tensor v_cache,
                   torch::Tensor slot_mapping);

TORCH_LIBRARY(nanovllm, m) {
  m.def("sample(Tensor logits, Tensor temperatures, int seed) -> Tensor");
  m.def("rms_norm(Tensor x, Tensor weight, float eps) -> Tensor");
  m.def(
      "add_rms_norm(Tensor x, Tensor residual, Tensor weight, float eps) -> "
      "Tensor[]");
  m.def(
      "rotary_embedding(Tensor positions, Tensor query, Tensor key, Tensor "
      "cos_sin_cache) -> Tensor[]");
  m.def(
      "store_kvcache(Tensor key, Tensor value, Tensor(a!) k_cache, Tensor(b!) "
      "v_cache, Tensor slot_mapping) -> ()");
}

TORCH_LIBRARY_IMPL(nanovllm, CUDA, m) {
  m.impl("sample", &sample);
  m.impl("rms_norm", &rms_norm);
  m.impl("add_rms_norm", &add_rms_norm);
  m.impl("rotary_embedding", &rotary_embedding);
  m.impl("store_kvcache", &store_kvcache);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}