// Offline check of next-layer expert prediction for decode (MoE offloading, cf. Eliseev & Mazur 2023, HOBBIT 2024):
// apply the router of layer application i+k to the router input of application i and compare the predicted
// experts with the ones application i+k really selects. Decodes -n tokens greedily after the prompt.
//   llama-inkling-moe-predict -m model.gguf -p "prompt" -n 256 [-cmoe --moe-cache-mib N ...]
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct probe {
    std::map<int, std::vector<float>> W;   // il -> router weights [n_expert][n_embd]
    std::map<int, std::vector<float>> b;   // il -> selection bias [n_expert]
    std::map<int, int> n_used;             // il -> experts per token
    int n_embd = 0, n_expert = 0;
    bool decode = false;                   // record only single-token decode steps
    // per decode token: il -> router input, il -> selected experts
    std::vector<std::map<int, std::vector<float>>> x;
    std::vector<std::map<int, std::vector<int>>> sel;
};

static int layer_of(const char * name, const char * prefix) {
    const size_t n = strlen(prefix);
    return strncmp(name, prefix, n) == 0 && name[n] == '-' ? atoi(name + n + 1) : -1;
}

static std::vector<float> to_f32(const ggml_tensor * t) {
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    std::vector<float> out(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), raw.data(), out.size()*sizeof(float));
    } else {
        GGML_ASSERT(ggml_is_contiguous(t));
        ggml_get_type_traits(t->type)->to_float(raw.data(), out.data(), out.size());
    }
    return out;
}

static bool cb_eval(ggml_tensor * t, bool ask, void * ud) {
    auto * p = (probe *) ud;
    const int il_logits = layer_of(t->name, "ffn_moe_logits");
    const int il_topk   = layer_of(t->name, "ffn_moe_topk");
    const int il_bias   = layer_of(t->name, "ffn_moe_probs_biased");
    if (ask) {
        return il_logits >= 0 || il_topk >= 0 || il_bias >= 0;
    }
    if (il_logits >= 0) {
        const ggml_tensor * w = t->src[0];
        const ggml_tensor * x = t->src[1];
        p->n_embd   = (int) w->ne[0];
        p->n_expert = (int) w->ne[1];
        if (!p->W.count(il_logits)) {
            p->W[il_logits] = to_f32(w);
        }
        p->decode = x->ne[1] == 1;
        if (p->decode) {
            if (il_logits == 0) {
                p->x.emplace_back();
                p->sel.emplace_back();
            }
            if (!p->x.empty() && ggml_is_contiguous(x)) {
                p->x.back()[il_logits] = to_f32(x);
            }
        }
    } else if (il_bias >= 0 && t->src[1] && !p->b.count(il_bias)) {
        p->b[il_bias] = to_f32(t->src[1]);
    } else if (il_topk >= 0 && p->decode && !p->sel.empty()) {
        std::vector<int32_t> ids(ggml_nelements(t));
        GGML_ASSERT(t->type == GGML_TYPE_I32 && ggml_is_contiguous(t));
        ggml_backend_tensor_get(t, ids.data(), 0, ids.size()*sizeof(int32_t));
        p->n_used[il_topk] = (int) t->ne[0];
        p->sel.back()[il_topk] = std::vector<int>(ids.begin(), ids.begin() + t->ne[0]);
    }
    return true;
}

// experts the router of application j selects for input x (sigmoid gating, selection bias)
static std::vector<int> predict(const probe & p, int j, const std::vector<float> & x) {
    const auto & W = p.W.at(j);
    std::vector<std::pair<float, int>> s(p.n_expert);
    for (int e = 0; e < p.n_expert; ++e) {
        double acc = 0.0;
        for (int d = 0; d < p.n_embd; ++d) {
            acc += (double) W[(size_t) e*p.n_embd + d]*x[d];
        }
        float v = 1.0f/(1.0f + expf((float) -acc));
        if (p.b.count(j)) {
            v += p.b.at(j)[e];
        }
        s[e] = {v, e};
    }
    const int k = p.n_used.at(j);
    std::partial_sort(s.begin(), s.begin() + k, s.end(), [](auto & a, auto & b) { return a.first > b.first; });
    std::vector<int> out(k);
    for (int i = 0; i < k; ++i) {
        out[i] = s[i].second;
    }
    return out;
}

static double overlap(const std::vector<int> & a, const std::vector<int> & b) {
    int n = 0;
    for (int e : a) {
        n += std::find(b.begin(), b.end(), e) != b.end();
    }
    return b.empty() ? 0.0 : (double) n/b.size();
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();

    probe p;
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &p;
    params.warmup = false;

    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to init\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size()))) {
        LOG_ERR("prompt decode failed\n");
        return 1;
    }
    const int n_gen = params.n_predict > 0 ? params.n_predict : 128;
    for (int i = 0; i < n_gen; ++i) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        llama_token tok = (llama_token) (std::max_element(logits, logits + llama_vocab_n_tokens(vocab)) - logits);
        if (llama_vocab_is_eog(vocab, tok)) {
            break;
        }
        printf("%s", common_token_to_piece(ctx, tok).c_str());
        fflush(stdout);
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) {
            LOG_ERR("decode failed\n");
            return 1;
        }
    }
    printf("\n\n");

    const int n_tok = (int) p.x.size();
    int n_layer = 0;
    for (auto & [il, _] : p.W) {
        n_layer = std::max(n_layer, il + 1);
    }
    printf("decode tokens recorded: %d, layer applications: %d, experts: %d, n_embd: %d\n", n_tok, n_layer, p.n_expert, p.n_embd);

    const int skip = 8;
    // sanity: the router of i on its own input reproduces the selection
    double sane = 0.0; int n_sane = 0;
    for (int t = skip; t < n_tok; ++t) {
        for (int i = 0; i < n_layer; i += 7) {
            if (p.x[t].count(i) && p.sel[t].count(i)) {
                sane += overlap(predict(p, i, p.x[t].at(i)), p.sel[t].at(i)); n_sane++;
            }
        }
    }
    printf("sanity (router i on input i): %.3f\n", n_sane ? sane/n_sane : 0.0);

    for (int k = 1; k <= 3; ++k) {
        double all = 0.0, fresh_hit = 0.0, fresh_n = 0.0; int n = 0;
        double grp[3] = {0, 0, 0}; int grp_n[3] = {0, 0, 0};
        for (int t = skip; t < n_tok; ++t) {
            for (int i = 0; i + k < n_layer; ++i) {
                const int j = i + k;
                if (!p.x[t].count(i) || !p.sel[t].count(j)) {
                    continue;
                }
                const auto pred = predict(p, j, p.x[t].at(i));
                const auto & act = p.sel[t].at(j);
                const double o = overlap(pred, act);
                all += o; n++;
                const int g = j < 6 ? 0 : j < 42 ? 1 : 2;
                grp[g] += o; grp_n[g]++;
                // experts of j that i did not use: the ones a cache is less likely to hold
                const auto & prev = p.sel[t].count(i) ? p.sel[t].at(i) : std::vector<int>();
                for (int e : act) {
                    if (std::find(prev.begin(), prev.end(), e) == prev.end()) {
                        fresh_n += 1;
                        fresh_hit += std::find(pred.begin(), pred.end(), e) != pred.end();
                    }
                }
            }
        }
        printf("router of i+%d on input of i: recall %.3f (first 6: %.3f, looped 6-41: %.3f, last: %.3f), new experts recall %.3f\n",
                k, n ? all/n : 0.0, grp_n[0] ? grp[0]/grp_n[0] : 0.0, grp_n[1] ? grp[1]/grp_n[1] : 0.0, grp_n[2] ? grp[2]/grp_n[2] : 0.0,
                fresh_n ? fresh_hit/fresh_n : 0.0);
    }
    // baseline: the same application in the previous token
    double prevtok = 0.0; int n_prev = 0;
    for (int t = skip + 1; t < n_tok; ++t) {
        for (int i = 0; i < n_layer; ++i) {
            if (p.sel[t].count(i) && p.sel[t - 1].count(i)) {
                prevtok += overlap(p.sel[t - 1].at(i), p.sel[t].at(i)); n_prev++;
            }
        }
    }
    printf("baseline, same application previous token: recall %.3f\n", n_prev ? prevtok/n_prev : 0.0);

    llama_backend_free();
    return 0;
}
