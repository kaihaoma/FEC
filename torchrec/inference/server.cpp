/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <memory>
#include <string>

#include <folly/futures/Future.h>
#include <folly/io/IOBuf.h>
#include <torch/csrc/deploy/deploy.h>
#include <torch/torch.h>

#include <glog/logging.h>
#include <grpc++/grpc++.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>

#include "torchrec/inference/GPUExecutor.h"
#include "torchrec/inference/predictor.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using predictor::PredictionRequest;
using predictor::PredictionResponse;
using predictor::Predictor;

DEFINE_int32(n_interp_per_gpu, 1, "");
DEFINE_int32(n_gpu, 1, "");
DEFINE_string(package_path, "/tmp/model_package.zip", "");

DEFINE_int32(batching_interval, 10, "");
DEFINE_int32(queue_timeout, 500, "");

DEFINE_int32(num_mem_pinner_threads, 4, "");
DEFINE_int32(max_batch_size, 2048, "");

DEFINE_string(server_address, "0.0.0.0", "");
DEFINE_string(server_port, "50051", "");

namespace {

std::unique_ptr<torchrec::PredictionRequest> toTorchRecRequest(
    const PredictionRequest* request) {
  auto torchRecRequest = std::make_unique<torchrec::PredictionRequest>();
  torchRecRequest->batch_size = request->batch_size();

  // Client sends a request with serialized tensor to bytes.
  // Byte string is converted to folly::iobuf for torchrec request.

  {
    torchrec::FloatFeatures floatFeature;

    auto feature = request->float_features();
    auto encoded_values = feature.values();

    floatFeature.num_features = feature.num_features();
    floatFeature.values = folly::IOBuf(
        folly::IOBuf::WRAP_BUFFER,
        encoded_values.c_str(),
        encoded_values.size());

    torchRecRequest->features["float_features"] = std::move(floatFeature);
  }

  {
    torchrec::SparseFeatures sparseFeature;

    auto feature = request->id_list_features();
    auto encoded_values = feature.values();
    auto encoded_lengths = feature.lengths();

    sparseFeature.num_features = feature.num_features();
    sparseFeature.lengths = folly::IOBuf(
        folly::IOBuf::WRAP_BUFFER,
        encoded_values.c_str(),
        encoded_values.size());
    sparseFeature.values = folly::IOBuf(
        folly::IOBuf::WRAP_BUFFER,
        encoded_lengths.c_str(),
        encoded_values.size());

    torchRecRequest->features["id_list_features"] = std::move(sparseFeature);
  }

  {
    torchrec::SparseFeatures sparseFeature;

    auto feature = request->id_score_list_features();
    auto encoded_values = feature.values();
    auto encoded_lengths = feature.lengths();
    auto encoded_weights = feature.weights();

    sparseFeature.num_features = feature.num_features();
    sparseFeature.lengths = folly::IOBuf(
        folly::IOBuf::WRAP_BUFFER,
        encoded_values.c_str(),
        encoded_values.size());
    sparseFeature.values = folly::IOBuf(
        folly::IOBuf::WRAP_BUFFER,
        encoded_lengths.c_str(),
        encoded_lengths.size());
    sparseFeature.weights = folly::IOBuf(
        folly::IOBuf::WRAP_BUFFER,
        encoded_weights.c_str(),
        encoded_weights.size());

    torchRecRequest->features["id_score_list_features"] =
        std::move(sparseFeature);
  }

  {
    torchrec::FloatFeatures floatFeature;

    auto feature = request->embedding_features();
    auto encoded_values = feature.values();

    floatFeature.num_features = feature.num_features();
    floatFeature.values = folly::IOBuf(
        folly::IOBuf::WRAP_BUFFER,
        encoded_values.c_str(),
        sizeof(encoded_values.c_str()));

    torchRecRequest->features["embedding_features"] = std::move(floatFeature);
  }

  {
    torchrec::SparseFeatures sparseFeature;

    auto feature = request->unary_features();
    auto encoded_lengths = feature.lengths();
    auto encoded_values = feature.values();

    sparseFeature.num_features = feature.num_features();
    sparseFeature.lengths = folly::IOBuf(
        folly::IOBuf::WRAP_BUFFER,
        encoded_values.c_str(),
        sizeof(encoded_values.c_str()));
    sparseFeature.values = folly::IOBuf(
        folly::IOBuf::WRAP_BUFFER,
        encoded_lengths.c_str(),
        sizeof(encoded_lengths.c_str()));

    torchRecRequest->features["unary_features"] = std::move(sparseFeature);
  }

  return torchRecRequest;
}

// Convert ivalue to map<string, float[]>, TODO: find out if protobuf can
// support custom types (folly::iobuf), so we can avoid this overhead.
// Otherwise, check out fbthrift.

// Logic and data behind the server's behavior.
class PredictorServiceHandler final : public Predictor::Service {
 public:
  explicit PredictorServiceHandler(torchrec::BatchingQueue& queue)
      : queue_(queue) {}

  Status Predict(
      grpc::ServerContext* context,
      const PredictionRequest* request,
      PredictionResponse* reply) override {
    folly::Promise<std::unique_ptr<torchrec::PredictionResponse>> promise;
    auto future = promise.getSemiFuture();
    queue_.add(toTorchRecRequest(request), std::move(promise));
    auto torchRecResponse =
        std::move(future).get(); // blocking, TODO: Write async server
    auto predictions = reply->mutable_predictions();

    for (const auto& item : torchRecResponse->predictions.toGenericDict()) {
      auto tensor = item.value().toTensor();

      // if we could use the folly::iobuf in protobuf, this wouldn't be
      // necessary.
      std::stringstream stream;
      torch::save(tensor, stream);

      (*predictions)[item.key().toStringRef()] = stream.str();
    }

    return Status::OK;
  }

 private:
  torchrec::BatchingQueue& queue_;
};

} // namespace

int main(int argc, char* argv[]) {
  LOG(INFO) << "Creating GPU executors";
  // store the executors and interpreter managers
  std::vector<std::unique_ptr<torchrec::GPUExecutor>> executors;
  std::vector<torch::deploy::ReplicatedObj> models;
  std::vector<torchrec::BatchQueueCb> batchQueueCbs;
  std::unordered_map<std::string, std::string> batchingMetadataMap;

  auto manager = std::make_shared<torch::deploy::InterpreterManager>(
      FLAGS_n_gpu * FLAGS_n_interp_per_gpu);
  {
    torch::deploy::Package package = manager->loadPackage(FLAGS_package_path);
    auto I = package.acquireSession();
    auto imported = I.self.attr("import_module")({"__module_loader"});
    auto factoryType = imported.attr("MODULE_FACTORY");
    auto factory = factoryType.attr("__new__")({factoryType});
    factoryType.attr("__init__")({factory});

    // Process forward metadata.
    auto batchingMetadata =
        factory.attr("batching_metadata")(at::ArrayRef<at::IValue>())
            .toIValue();
    for (const auto& iter : batchingMetadata.toGenericDict()) {
      batchingMetadataMap[iter.key().toString()->string()] =
          iter.value().toString()->string();
    }

    // Process result metadata.
    auto resultMetadata =
        factory.attr("result_metadata")(at::ArrayRef<at::IValue>())
            .toIValue()
            .toStringRef();
    std::shared_ptr<torchrec::ResultSplitFunc> resultSplitFunc =
        torchrec::TorchRecResultSplitFuncRegistry()->Create(resultMetadata);

    LOG(INFO) << "Creating Model Shard for " << FLAGS_n_gpu << " GPUs.";
    auto dmp = factory.attr("create_predict_module")
                   .callKwargs({{"world_size", FLAGS_n_gpu}});

    for (int rank = 0; rank < FLAGS_n_gpu; rank++) {
      auto device = I.self.attr("import_module")({"torch"}).attr("device")(
          {"cuda", rank});
      auto m = dmp.attr("copy")({device.toIValue()});
      models.push_back(I.createMovable(m));
    }

    for (int rank = 0; rank < FLAGS_n_gpu; rank++) {
      auto executor = std::make_unique<torchrec::GPUExecutor>(
          manager, std::move(models[rank]), rank, FLAGS_n_gpu, resultSplitFunc);
      executors.push_back(std::move(executor));
      batchQueueCbs.push_back(
          [&, rank](std::shared_ptr<torchrec::PredictionBatch> batch) {
            executors[rank]->callback(std::move(batch));
          });
    }
  }

  torchrec::BatchingQueue queue(
      batchQueueCbs,
      torchrec::BatchingQueue::Config{
          .batchingInterval =
              std::chrono::milliseconds(FLAGS_batching_interval),
          .queueTimeout = std::chrono::milliseconds(FLAGS_queue_timeout),
          .numMemPinnerThreads = FLAGS_num_mem_pinner_threads,
          .maxBatchSize = FLAGS_max_batch_size,
          .batchingMetadata = std::move(batchingMetadataMap),
      },
      FLAGS_n_gpu);

  // create the server
  std::string server_address(FLAGS_server_address + ":" + FLAGS_server_port);
  auto service = PredictorServiceHandler(queue);

  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder builder;

  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);

  // Finally assemble the server.
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  LOG(INFO) << "Server listening on " << server_address;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();

  LOG(INFO) << "Shutting down server";
  return 0;
}
