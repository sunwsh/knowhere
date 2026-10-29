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

#pragma once

#include <limits>
#include <string>

#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"
#include "knowhere/utils.h"

namespace knowhere {

class PagConfig : public BaseConfig {
 public:
    CFG_INT M;
    CFG_INT efConstruction;
    CFG_INT ef;
    CFG_INT projection_levels;
    CFG_INT max_search_k;

    KNOHWERE_DECLARE_CONFIG(PagConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(M)
            .description("PAG target degree")
            .set_default(16)
            .set_range(1, 2048)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(efConstruction)
            .description("PAG efConstruction")
            .set_default(200)
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(ef)
            .description("PAG ef")
            .allow_empty_without_default()
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(projection_levels)
            .description("PAG projection levels, a positive multiple of 8")
            .set_default(64)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(max_search_k)
            .description("PAG build-time search k cap")
            .set_default(100)
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_train();
    }

    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        switch (param_type) {
            case PARAM_TYPE::TRAIN: {
                const auto metric_status = CheckMetric(metric_type.value(), err_msg);
                if (metric_status != Status::success) {
                    return metric_status;
                }
                const auto levels = projection_levels.value();
                if (levels <= 0 || levels % 8 != 0) {
                    return HandleError(err_msg, "projection_levels must be a positive multiple of 8",
                                       Status::invalid_param_in_json);
                }
                break;
            }
            case PARAM_TYPE::SEARCH: {
                const auto metric_status = CheckMetric(metric_type.value(), err_msg);
                if (metric_status != Status::success) {
                    return metric_status;
                }
                if (!ef.has_value()) {
                    ef = std::max(k.value(), 16);
                } else if (k.value() > ef.value()) {
                    const std::string msg = "ef(" + std::to_string(ef.value()) + ") should be larger than k(" +
                                            std::to_string(k.value()) + ")";
                    return HandleError(err_msg, msg, Status::out_of_range_in_json);
                }
                break;
            }
            default:
                break;
        }
        return Status::success;
    }

 private:
    static Status
    CheckMetric(const std::string& metric, std::string* err_msg) {
        if (IsMetricType(metric, metric::L2) || IsMetricType(metric, metric::IP) ||
            IsMetricType(metric, metric::COSINE)) {
            return Status::success;
        }
        return HandleError(err_msg, "PAG only supports L2, IP and COSINE", Status::invalid_metric_type);
    }
};

}  // namespace knowhere
