#ifndef TORCHDATA_S3_IO_H
#define TORCHDATA_S3_IO_H

#include <aws/core/utils/StringUtils.h>
#include <aws/core/utils/threading/Executor.h>
#include <aws/s3/S3Client.h>
#include <aws/transfer/TransferManager.h>

#include <mutex>

namespace torchdata
{
   // In memory stream implementation
   class S3UnderlyingStream : public Aws::IOStream
   {
   public:
      using Base = Aws::IOStream;

      // provide a customer controlled streambuf, so as to put all transferred
      // data into this in memory buffer.
      S3UnderlyingStream(std::streambuf *buf) : Base(buf) {}

      virtual ~S3UnderlyingStream() = default;
   };

   class S3Handler
   {
   private:
      static std::shared_ptr<Aws::Client::ClientConfiguration> s3_handler_cfg_;

      std::shared_ptr<Aws::S3::S3Client> s3_client_;
      std::shared_ptr<Aws::Utils::Threading::PooledThreadExecutor> executor_;
      std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager_;

      Aws::String last_marker_;
      int max_keys_;
      std::mutex initialization_lock_;
      size_t buffer_size_;
      bool multi_part_download_;

      void InitializeS3Client();
      void InitializeS3Client(const long requestTimeoutMs, const std::string region);
      void InitializeExecutor();
      void InitializeTransferManager();

      std::shared_ptr<Aws::S3::S3Client> GetS3Client();
      std::shared_ptr<Aws::Utils::Threading::PooledThreadExecutor>
      GetExecutor();
      std::shared_ptr<Aws::Transfer::TransferManager> GetTransferManager();
      size_t GetFileSize(const std::string &bucket, const std::string &object);

   public:
      S3Handler(const long requestTimeoutMs, const std::string region);
      ~S3Handler();

      void SetMaxKeys(const int max_keys) { this->max_keys_ = max_keys; }
      void ClearMarker();

      void S3Read(const std::string &file_url, std::string *result);
      void ListFiles(const std::string &file_url,
                     std::vector<std::string> *filenames);
   };
} // namespace torchdata

#endif // TORCHDATA_S3_IO_H
