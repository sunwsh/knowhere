// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <cstdint>
#include <string>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/binaryset.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/version.h"
#include "simd/instruction_set.h"
#include "utils.h"

namespace {

bool
PagInstructionSetReady() {
#if defined(__x86_64__)
    auto& cpu = faiss::InstructionSet::GetInstance();
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    const bool vnni = __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) != 0 && (ecx & (1u << 11)) != 0;
    return cpu.AVX512F() && cpu.AVX512DQ() && cpu.AVX512CD() && cpu.AVX512BW() && cpu.AVX512VL() && vnni;
#else
    return false;
#endif
}

knowhere::Json
PagJson(const std::string& metric, int64_t dim, int64_t k, int projection_levels = 16, int max_search_k = 10) {
    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::METRIC_TYPE] = metric;
    json[knowhere::meta::TOPK] = k;
    json[knowhere::indexparam::HNSW_M] = 8;
    json[knowhere::indexparam::EFCONSTRUCTION] = 32;
    json[knowhere::indexparam::EF] = std::max<int64_t>(k, 16);
    json["projection_levels"] = projection_levels;
    json["max_search_k"] = max_search_k;
    return json;
}

void
RequireKnnShape(const knowhere::DataSetPtr& result, int64_t nq, int64_t k, int64_t nb) {
    REQUIRE(result != nullptr);
    REQUIRE(result->GetRows() == nq);
    REQUIRE(result->GetDim() == k);
    const int64_t* ids = result->GetIds();
    const float* distances = result->GetDistance();
    REQUIRE(ids != nullptr);
    REQUIRE(distances != nullptr);
    for (int64_t i = 0; i < nq * k; ++i) {
        REQUIRE(ids[i] >= 0);
        REQUIRE(ids[i] < nb);
    }
}

}  // namespace

TEST_CASE("PAG build and search", "[pag]") {
    if (!PagInstructionSetReady()) {
        SKIP("CPU does not provide the AVX-512 features PAG requires");
    }

    using knowhere::IndexEnum;
    using knowhere::Status;
    using knowhere::metric::COSINE;
    using knowhere::metric::IP;
    using knowhere::metric::L2;

    const int64_t nb = 128;
    const int64_t nq = 3;
    const int64_t dim = 16;
    const int64_t k = 5;
    const auto metric = GENERATE(as<std::string>{}, L2, IP, COSINE);
    const auto version = knowhere::Version::GetDefaultVersion().VersionNumber();
    auto base = GenDataSet(nb, dim, 7);
    auto query = GenDataSet(nq, dim, 11);
    const auto json = PagJson(metric, dim, k);

    auto created = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(IndexEnum::INDEX_PAG, version);
    REQUIRE(created.has_value());
    auto index = created.value();
    REQUIRE(index.Build(base, json) == Status::success);
    REQUIRE(index.Dim() == dim);
    REQUIRE(index.Count() == nb);
    REQUIRE(index.Type() == std::string(IndexEnum::INDEX_PAG));
    REQUIRE_FALSE(index.HasRawData(metric));

    auto result = index.Search(query, json, knowhere::BitsetView());
    REQUIRE(result.has_value());
    RequireKnnShape(result.value(), nq, k, nb);

    knowhere::BinarySet binset;
    REQUIRE(index.Serialize(binset) == Status::success);
    auto loaded_created = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(IndexEnum::INDEX_PAG, version);
    REQUIRE(loaded_created.has_value());
    auto loaded = loaded_created.value();
    REQUIRE(loaded.Deserialize(binset, json) == Status::success);
    auto loaded_result = loaded.Search(query, json, knowhere::BitsetView());
    REQUIRE(loaded_result.has_value());
    RequireKnnShape(loaded_result.value(), nq, k, nb);
    const int64_t* ids = result.value()->GetIds();
    const int64_t* loaded_ids = loaded_result.value()->GetIds();
    const float* distances = result.value()->GetDistance();
    const float* loaded_distances = loaded_result.value()->GetDistance();
    for (int64_t i = 0; i < nq * k; ++i) {
        REQUIRE(ids[i] == loaded_ids[i]);
        REQUIRE(distances[i] == loaded_distances[i]);
    }
}

TEST_CASE("PAG rejects oversized k and bitset filtering", "[pag]") {
    if (!PagInstructionSetReady()) {
        SKIP("CPU does not provide the AVX-512 features PAG requires");
    }

    const int64_t nb = 128;
    const int64_t dim = 16;
    const auto version = knowhere::Version::GetDefaultVersion().VersionNumber();
    auto base = GenDataSet(nb, dim, 7);
    auto query = GenDataSet(1, dim, 11);
    const auto json = PagJson(knowhere::metric::L2, dim, 5, 16, 10);
    auto created = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_PAG, version);
    REQUIRE(created.has_value());
    auto index = created.value();
    REQUIRE(index.Build(base, json) == knowhere::Status::success);

    auto too_large = PagJson(knowhere::metric::L2, dim, 11, 16, 10);
    auto oversized = index.Search(query, too_large, knowhere::BitsetView());
    REQUIRE_FALSE(oversized.has_value());
    REQUIRE(oversized.error() == knowhere::Status::out_of_range_in_json);

    uint8_t bits = 0x1;
    knowhere::BitsetView filtered(&bits, static_cast<size_t>(nb), 1);
    auto blocked = index.Search(query, json, filtered);
    REQUIRE_FALSE(blocked.has_value());
    REQUIRE(blocked.error() == knowhere::Status::not_implemented);
}

TEST_CASE("PAG train and projection_levels checks", "[pag]") {
    const int64_t nb = 32;
    const int64_t dim = 16;
    const auto version = knowhere::Version::GetDefaultVersion().VersionNumber();
    auto base = GenDataSet(nb, dim, 7);
    const auto json = PagJson(knowhere::metric::L2, dim, 5);
    auto created = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_PAG, version);
    REQUIRE(created.has_value());
    auto index = created.value();
    REQUIRE(index.Train(base, json) == knowhere::Status::not_implemented);

    auto bad_levels = PagJson(knowhere::metric::L2, dim, 5, 12, 10);
    REQUIRE(index.Build(base, bad_levels) == knowhere::Status::invalid_param_in_json);
}
