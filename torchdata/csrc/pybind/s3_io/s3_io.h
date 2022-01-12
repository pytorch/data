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
      std::shared_ptr<Aws::S3::S3Client> s3_client_;
      std::shared_ptr<Aws::Utils::Threading::PooledThreadExecutor> executor_;
      std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager_;
      std::mutex initialization_lock_;
      size_t buffer_size_;
      bool multi_part_download_;

      std::shared_ptr<Aws::S3::S3Client> InitializeS3Client();
      std::shared_ptr<Aws::Utils::Threading::PooledThreadExecutor>
      InitializeExecutor();
      std::shared_ptr<Aws::Transfer::TransferManager> InitializeTransferManager();
      size_t GetFileSize(const std::string &bucket, const std::string &object);
      size_t GetFileSize(const std::string &file_url);

   public:
      S3Handler();
      ~S3Handler();

      std::shared_ptr<Aws::S3::S3Client> GetS3Client();
      std::shared_ptr<Aws::Utils::Threading::PooledThreadExecutor>
      GetExecutor();
      std::shared_ptr<Aws::Transfer::TransferManager> GetTransferManager();

      void S3Read(const std::string &file_url, std::string *result);
      void ListFiles(const std::string &file_url,
                     std::vector<std::string> *filenames);
   };
} // namespace torchdata

#endif // TORCHDATA_S3_IO_H
