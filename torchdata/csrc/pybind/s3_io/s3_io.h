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

   class S3Init
   {
   private:
      std::shared_ptr<Aws::S3::S3Client> s3_client_;
      std::shared_ptr<Aws::Utils::Threading::PooledThreadExecutor> executor_;
      std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager_;
      size_t buffer_size_;
      bool multi_part_download_;

      size_t get_file_size(const std::string &bucket, const std::string &object);

   public:
      S3Init();

      ~S3Init();

      std::mutex initialization_lock_;

      std::shared_ptr<Aws::S3::S3Client> initializeS3Client();
      std::shared_ptr<Aws::Utils::Threading::PooledThreadExecutor>
      initializeExecutor();
      std::shared_ptr<Aws::Transfer::TransferManager> initializeTransferManager();

      void s3_read(const std::string &file_url, std::string *result);
      size_t get_file_size(const std::string &file_url);
      bool file_exists(const std::string &file_url);
      void list_files(const std::string &file_url,
                      std::vector<std::string> *filenames);
   };
} // namespace torchdata

#endif // TORCHDATA_S3_IO_H
