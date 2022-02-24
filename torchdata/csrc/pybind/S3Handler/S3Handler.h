#include "precompile.h"

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

      std::shared_ptr<std::mutex> initialization_lock_;
      std::shared_ptr<Aws::S3::S3Client> s3_client_;
      std::shared_ptr<Aws::Utils::Threading::PooledThreadExecutor> executor_;
      std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager_;

      Aws::String last_marker_;
      size_t buffer_size_;
      bool use_multi_part_download_;

      void InitializeS3Client();
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

      void SetLastMarker(const Aws::String last_marker) { this->last_marker_ = last_marker; }
      void SetBufferSize(const uint64_t buffer_size) { this->buffer_size_ = buffer_size; }
      void SetMultiPartDownload(const bool multi_part_download) { this->use_multi_part_download_ = multi_part_download; }
      void ClearMarker();

      long GetRequestTimeoutMs() const { return s3_handler_cfg_->requestTimeoutMs; }
      Aws::String GetRegion() const { return s3_handler_cfg_->region; }
      Aws::String GetLastMarker() const { return last_marker_; }
      bool GetUseMultiPartDownload() const { return use_multi_part_download_; }
      size_t GetBufferSize() const { return buffer_size_; }

      void S3Read(const std::string &file_url, std::string *result);
      void ListFiles(const std::string &file_url,
                     std::vector<std::string> *filenames);
   };
} // namespace torchdata
