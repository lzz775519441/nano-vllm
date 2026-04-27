#include <torch/extension.h>

#include <cstdint>
#include <vector>

torch::Tensor sample(torch::Tensor logits, torch::Tensor temperatures,
                     std::uint64_t seed);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sample", &sample, "Fused Gumbel-max sampling");
    m.def("rms_norm", &rms_norm, "Fused RMSNorm");
    m.def("add_rms_norm", &add_rms_norm, "Fused residual add + RMSNorm");
    m.def("rotary_embedding", &rotary_embedding, "Fused rotary embedding");
    m.def("store_kvcache", &store_kvcache,
          "Store key/value tensors into paged KV cache");
}
