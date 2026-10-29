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

#include "index/pag/pag_config.h"

#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <utility>

#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/feature.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"
#include "pag_index.h"
#include "simd/instruction_set.h"

namespace knowhere {
namespace {

constexpr int kDefaultSearchEf = 16;

Status
CheckPagInstructionSet() {
#if defined(__x86_64__)
    auto& cpu = faiss::InstructionSet::GetInstance();
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    const bool vnni = __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) != 0 && (ecx & (1u << 11)) != 0;
    if (cpu.AVX512F() && cpu.AVX512DQ() && cpu.AVX512CD() && cpu.AVX512BW() && cpu.AVX512VL() && vnni) {
        return Status::success;
    }
#else
    (void)0;
#endif
    return Status::invalid_instruction_set;
}

pag::Metric
ToPagMetric(const std::string& metric) {
    if (IsMetricType(metric, metric::L2)) {
        return pag::Metric::L2;
    }
    if (IsMetricType(metric, metric::IP)) {
        return pag::Metric::MaximumInnerProduct;
    }
    if (IsMetricType(metric, metric::COSINE)) {
        return pag::Metric::Cosine;
    }
    throw std::invalid_argument("unsupported PAG metric");
}

bool
SameMetric(pag::Metric actual, const std::string& requested) {
    try {
        return actual == ToPagMetric(requested);
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace

template <typename DataType>
class PagIndexNode : public IndexNode {
    static_assert(std::is_same<DataType, fp32>::value, "PAG only supports fp32");

 public:
    PagIndexNode(const int32_t& version, const Object&) : IndexNode(version) {
    }

    Status
    Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool) override {
        const Status instruction = CheckPagInstructionSet();
        if (instruction != Status::success) {
            return instruction;
        }
        auto& pag_cfg = static_cast<PagConfig&>(*cfg);
        if (dataset == nullptr) {
            return Status::invalid_args;
        }
        const int64_t rows = dataset->GetRows();
        const int64_t dim = dataset->GetDim();
        const auto* tensor = static_cast<const float*>(dataset->GetTensor());
        if (tensor == nullptr || rows <= 0 || dim <= 0 ||
            rows > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
            return Status::invalid_args;
        }
        const int64_t* ids = dataset->GetIds();
        if (ids != nullptr) {
            for (int64_t i = 0; i < rows; ++i) {
                if (ids[i] != i) {
                    return Status::invalid_args;
                }
            }
        }

        pag::Metric metric;
        try {
            metric = ToPagMetric(pag_cfg.metric_type.value());
        } catch (const std::exception&) {
            return Status::invalid_metric_type;
        }

        std::filesystem::path directory;
        bool created = false;
        const auto cleanup = [&directory, &created]() -> Status {
            if (!created) {
                return Status::success;
            }
            std::error_code ec;
            std::filesystem::remove_all(directory, ec);
            created = false;
            if (ec) {
                LOG_KNOWHERE_ERROR_ << "failed to remove PAG build directory: " << ec.message();
                return Status::disk_file_error;
            }
            return Status::success;
        };

        try {
            std::error_code ec;
            const auto root = std::filesystem::temp_directory_path(ec);
            if (ec) {
                return Status::disk_file_error;
            }
            std::mt19937_64 generator{std::random_device{}()};
            std::uniform_int_distribution<uint64_t> pick;
            for (int attempt = 0; attempt < 8 && !created; ++attempt) {
                directory = root / ("pag-build-" + std::to_string(static_cast<long long>(::getpid())) + "-" +
                                    std::to_string(pick(generator)));
                created = std::filesystem::create_directory(directory, ec) && !ec;
            }
            if (!created) {
                LOG_KNOWHERE_ERROR_ << "failed to create PAG build directory";
                return Status::disk_file_error;
            }

            pag::BuildOptions options;
            options.index_path = directory.string();
            options.metric = metric;
            options.mode = pag::IndexMode::Static;
            options.max_search_k = static_cast<size_t>(pag_cfg.max_search_k.value());
            options.ef_construction = pag_cfg.efConstruction.value();
            options.target_degree = pag_cfg.M.value();
            options.projection_levels = pag_cfg.projection_levels.value();

            auto built = std::make_unique<pag::Index>();
            built->build(tensor, static_cast<size_t>(rows), static_cast<size_t>(dim), options);
            const Status removed = cleanup();
            if (removed != Status::success) {
                return removed;
            }
            index_ = std::move(built);
            return Status::success;
        } catch (const std::filesystem::filesystem_error& e) {
            LOG_KNOWHERE_ERROR_ << "PAG build directory error: " << e.what();
            const Status removed = cleanup();
            if (removed != Status::success) {
                return removed;
            }
            return Status::disk_file_error;
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "pag inner error: " << e.what();
            const Status removed = cleanup();
            if (removed != Status::success) {
                return removed;
            }
            return Status::pag_inner_error;
        }
    }

    Status
    Train(const DataSetPtr, std::shared_ptr<Config>, bool) override {
        return Status::not_implemented;
    }

    Status
    Add(const DataSetPtr, std::shared_ptr<Config>, bool) override {
        return Status::not_implemented;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext*) const override {
        if (bitset.count() > 0) {
            return expected<DataSetPtr>::Err(Status::not_implemented, "PAG does not support bitset filtering");
        }
        if (index_ == nullptr || !index_->is_loaded()) {
            return expected<DataSetPtr>::Err(Status::empty_index, "PAG index is empty");
        }
        if (dataset == nullptr) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "dataset is null");
        }
        auto& pag_cfg = static_cast<const PagConfig&>(*cfg);
        if (!SameMetric(index_->metric(), pag_cfg.metric_type.value())) {
            return expected<DataSetPtr>::Err(Status::invalid_metric_type, "metric does not match the PAG index");
        }
        const int64_t nq = dataset->GetRows();
        const int64_t dim = dataset->GetDim();
        const int64_t k = pag_cfg.k.value();
        const auto* queries = static_cast<const float*>(dataset->GetTensor());
        if (nq < 0 || k <= 0 || dim != static_cast<int64_t>(index_->dimension()) || (nq > 0 && queries == nullptr) ||
            (nq > 0 && k > std::numeric_limits<int64_t>::max() / nq)) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "invalid PAG query");
        }
        if (static_cast<size_t>(k) > index_->max_search_k()) {
            return expected<DataSetPtr>::Err(Status::out_of_range_in_json, "k exceeds PAG max_search_k");
        }
        if (nq == 0) {
            return GenResultDataSet(nq, k, new int64_t[0], new float[0]);
        }

        try {
            pag::SearchOptions options;
            options.top_k = static_cast<size_t>(k);
            options.ef_search =
                static_cast<size_t>(pag_cfg.ef.has_value() ? pag_cfg.ef.value() : std::max(k, int64_t{kDefaultSearchEf}));
            const auto batch = index_->search_batch(queries, static_cast<size_t>(nq), options);
            if (static_cast<int64_t>(batch.size()) != nq) {
                return expected<DataSetPtr>::Err(Status::pag_inner_error, "PAG returned the wrong number of rows");
            }
            auto ids = std::make_unique<int64_t[]>(static_cast<size_t>(nq * k));
            auto distances = std::make_unique<float[]>(static_cast<size_t>(nq * k));
            const bool report_distance = index_->metric() == pag::Metric::L2;
            for (int64_t row = 0; row < nq; ++row) {
                if (static_cast<int64_t>(batch[static_cast<size_t>(row)].size()) != k) {
                    return expected<DataSetPtr>::Err(Status::pag_inner_error, "PAG returned the wrong number of hits");
                }
                for (int64_t col = 0; col < k; ++col) {
                    const auto& hit = batch[static_cast<size_t>(row)][static_cast<size_t>(col)];
                    const size_t offset = static_cast<size_t>(row * k + col);
                    ids[offset] = static_cast<int64_t>(hit.id);
                    distances[offset] = report_distance ? hit.distance : hit.inner_product;
                }
            }
            return GenResultDataSet(nq, k, ids.release(), distances.release());
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "pag inner error: " << e.what();
            return expected<DataSetPtr>::Err(Status::pag_inner_error, e.what());
        }
    }

    expected<DataSetPtr>
    RangeSearch(const DataSetPtr, std::unique_ptr<Config>, const BitsetView&, milvus::OpContext*) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "PAG does not support range search");
    }

    expected<std::vector<IndexNode::IteratorPtr>>
    AnnIterator(const DataSetPtr, std::unique_ptr<Config>, const BitsetView&, bool, milvus::OpContext*) const override {
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::not_implemented,
                                                                  "PAG does not support iterators");
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr, milvus::OpContext*) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "PAG does not keep raw vectors");
    }

    bool
    HasRawData(const std::string&) const override {
        return false;
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config>) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "PAG does not support index meta");
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (index_ == nullptr || !index_->is_loaded()) {
            return Status::empty_index;
        }
        try {
            pag::MemoryIOWriter writer;
            index_->save(writer);
            auto released = writer.release();
            std::shared_ptr<uint8_t[]> data(released.first);
            binset.Append(Type(), data, static_cast<int64_t>(released.second));
            return Status::success;
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "pag inner error: " << e.what();
            return Status::pag_inner_error;
        }
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config>) override {
        const Status instruction = CheckPagInstructionSet();
        if (instruction != Status::success) {
            return instruction;
        }
        auto binary = binset.GetByName(Type());
        if (binary == nullptr || binary->data == nullptr || binary->size <= 0) {
            return Status::invalid_binary_set;
        }
        try {
            pag::MemoryIOReader reader(binary->data.get(), static_cast<size_t>(binary->size));
            auto loaded = std::make_unique<pag::Index>();
            loaded->load(reader);
            index_ = std::move(loaded);
            return Status::success;
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "pag inner error: " << e.what();
            return Status::pag_inner_error;
        }
    }

    Status
    DeserializeFromFile(const std::string&, std::shared_ptr<Config>) override {
        return Status::not_implemented;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<PagConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig&, const IndexVersion&) {
        return false;
    }

    int64_t
    Dim() const override {
        if (index_ == nullptr || !index_->is_loaded()) {
            return 0;
        }
        return static_cast<int64_t>(index_->dimension());
    }

    int64_t
    Size() const override {
        if (index_ == nullptr || !index_->is_loaded()) {
            return 0;
        }
        return static_cast<int64_t>(index_->size_bytes());
    }

    int64_t
    Count() const override {
        if (index_ == nullptr || !index_->is_loaded()) {
            return 0;
        }
        return static_cast<int64_t>(index_->count());
    }

    std::string
    Type() const override {
        return std::string(IndexEnum::INDEX_PAG);
    }

 private:
    std::unique_ptr<pag::Index> index_;
};

KNOWHERE_SIMPLE_REGISTER_GLOBAL(PAG, PagIndexNode, fp32, knowhere::feature::FLOAT32)

}  // namespace knowhere
