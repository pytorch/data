#include "S3Handler.h"

#include <aws/core/auth/AWSAuthSigner.h>
#include <aws/core/Aws.h>
#include <aws/core/config/AWSProfileConfigLoader.h>
#include <aws/core/http/Scheme.h>
#include <aws/core/utils/FileSystemUtils.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/core/utils/logging/AWSLogging.h>
#include <aws/core/utils/logging/LogSystemInterface.h>
#include <aws/core/utils/memory/AWSMemory.h>
#include <aws/core/utils/memory/stl/AWSStreamFwd.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/core/utils/threading/Executor.h>
#include <aws/crt/auth/Sigv4Signing.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/S3Errors.h>
#include <aws/s3/model/CompletedPart.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/ListObjectsRequest.h>
#include <aws/transfer/TransferManager.h>

#include <fstream>
#include <string>

namespace torchdata
{
    namespace
    {
        static const size_t s3ReadBufferSize = 120 * 1024 * 1024;              // 16 MB
        static const uint64_t s3MultiPartDownloadChunkSize = 50 * 1024 * 1024; // 50 MB
        static const int executorPoolSize = 25;
        static const int S3DefaultMaxKeys = 1000;
        static const std::string S3DefaultMarker = "";

        std::shared_ptr<Aws::Client::ClientConfiguration> setUpS3Config(const long requestTimeoutMs, const std::string region)
        {
            std::shared_ptr<Aws::Client::ClientConfiguration> cfg =
                std::shared_ptr<Aws::Client::ClientConfiguration>(new Aws::Client::ClientConfiguration());
            Aws::String config_file;
            const char *config_file_env = getenv("AWS_CONFIG_FILE");
            if (config_file_env)
            {
                config_file = config_file_env;
            }
            else
            {
                const char *home_env = getenv("HOME");
                if (home_env)
                {
                    config_file = home_env;
                    config_file += "/.aws/config";
                }
            }
            Aws::Config::AWSConfigFileProfileConfigLoader loader(config_file);
            loader.Load();

            const char *use_https = getenv("S3_USE_HTTPS");
            if (use_https)
            {
                if (use_https[0] == '0')
                {
                    cfg->scheme = Aws::Http::Scheme::HTTP;
                }
                else
                {
                    cfg->scheme = Aws::Http::Scheme::HTTPS;
                }
            }
            const char *verify_ssl = getenv("S3_VERIFY_SSL");
            if (verify_ssl)
            {
                if (verify_ssl[0] == '0')
                {
                    cfg->verifySSL = false;
                }
                else
                {
                    cfg->verifySSL = true;
                }
            }
            const char *endpoint_url = getenv("S3_ENDPOINT_URL");
            if (endpoint_url)
            {
                cfg->endpointOverride = endpoint_url;
            }
            if (region != "")
            {
                cfg->region = region;
            }
            else
            {
                const char *env_region = getenv("AWS_REGION");
                if (env_region)
                {
                    cfg->region = env_region;
                }
            }
            if (requestTimeoutMs > -1)
            {
                cfg->requestTimeoutMs = requestTimeoutMs;
            }
            return cfg;
        }

        void ShutdownClient(std::shared_ptr<Aws::S3::S3Client> *s3_client)
        {
            if (s3_client != nullptr)
            {
                delete s3_client;
                Aws::SDKOptions options;
                Aws::ShutdownAPI(options);
            }
        }

        void ShutdownTransferManager(
            std::shared_ptr<Aws::Transfer::TransferManager> *transfer_manager)
        {
            if (transfer_manager != nullptr)
            {
                delete transfer_manager;
            }
        }

        void ShutdownExecutor(Aws::Utils::Threading::PooledThreadExecutor *executor)
        {
            if (executor != nullptr)
            {
                delete executor;
            }
        }

        void parseS3Path(const Aws::String &fname, Aws::String *bucket,
                         Aws::String *object)
        {
            if (fname.empty())
            {
                throw std::invalid_argument("The filename cannot be an empty string.");
            }

            if (fname.size() < 5 || fname.substr(0, 5) != "s3://")
            {
                throw std::invalid_argument("The filename must start with the S3 scheme.");
            }

            std::string path = fname.substr(5);

            if (path.empty())
            {
                throw std::invalid_argument("The filename cannot be an empty string.");
            }

            size_t pos = path.find_first_of('/');
            if (pos == 0)
            {
                throw std::invalid_argument("The filename does not contain a bucket name.");
            }

            *bucket = path.substr(0, pos);
            *object = path.substr(pos + 1);
            if (pos == std::string::npos)
            {
                *object = "";
            }
        }

        class S3FS
        {
        public:
            S3FS(const std::string &bucket, const std::string &object,
                 const bool multi_part_download,
                 std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager,
                 std::shared_ptr<Aws::S3::S3Client> s3_client)
                : bucket_name_(bucket),
                  object_name_(object),
                  multi_part_download_(multi_part_download),
                  transfer_manager_(transfer_manager),
                  s3_client_(s3_client) {}

            size_t read(uint64_t offset, size_t n, char *buffer)
            {
                if (multi_part_download_)
                {
                    return readS3TransferManager(offset, n, buffer);
                }
                else
                {
                    return readS3Client(offset, n, buffer);
                }
            }

            size_t readS3Client(uint64_t offset, size_t n, char *buffer)
            {
                Aws::S3::Model::GetObjectRequest getObjectRequest;

                getObjectRequest.WithBucket(this->bucket_name_.c_str())
                    .WithKey(this->object_name_.c_str());

                std::string bytes = "bytes=";
                bytes += std::to_string(offset) + "-" + std::to_string(offset + n - 1);

                getObjectRequest.SetRange(bytes.c_str());

                // When you don’t want to load the entire file into memory,
                // you can use IOStreamFactory in AmazonWebServiceRequest to pass a
                // lambda to create a string stream.
                getObjectRequest.SetResponseStreamFactory(
                    []()
                    { return Aws::New<Aws::StringStream>("S3IOAllocationTag"); });
                // get the object
                Aws::S3::Model::GetObjectOutcome getObjectOutcome = this->s3_client_->GetObject(getObjectRequest);

                if (!getObjectOutcome.IsSuccess())
                {
                    Aws::S3::S3Error error = getObjectOutcome.GetError();
                    std::cout << "ERROR: " << error.GetExceptionName() << ": "
                              << error.GetMessage() << std::endl;
                    return 0;
                }
                else
                {
                    n = getObjectOutcome.GetResult().GetContentLength();
                    // read data as a block:
                    getObjectOutcome.GetResult().GetBody().read(buffer, n);
                    return n;
                }
            }

            size_t readS3TransferManager(uint64_t offset, size_t n, char *buffer)
            {
                auto create_stream_fn = [&]() { // create stream lambda fn
                    return Aws::New<S3UnderlyingStream>(
                        "S3ReadStream",
                        Aws::New<Aws::Utils::Stream::PreallocatedStreamBuf>(
                            "S3ReadStream", reinterpret_cast<unsigned char *>(buffer),
                            n));
                }; // This buffer is what we used to initialize streambuf and is in memory

                std::shared_ptr<Aws::Transfer::TransferHandle> downloadHandle =
                    this->transfer_manager_.get()->DownloadFile(
                        this->bucket_name_.c_str(), this->object_name_.c_str(), offset,
                        n, create_stream_fn);
                downloadHandle->WaitUntilFinished();

                Aws::OFStream storeFile(object_name_.c_str(),
                                        Aws::OFStream::out | Aws::OFStream::trunc);

                if (downloadHandle->GetStatus() !=
                    Aws::Transfer::TransferStatus::COMPLETED)
                {
                    const Aws::Client::AWSError<Aws::S3::S3Errors> error = downloadHandle->GetLastError();
                    std::cout << "ERROR: " << error.GetExceptionName() << ": "
                              << error.GetMessage() << std::endl;
                    return 0;
                }
                else
                {
                    return downloadHandle->GetBytesTransferred();
                }
            }

        private:
            std::string bucket_name_;
            std::string object_name_;
            bool multi_part_download_;
            std::shared_ptr<Aws::S3::S3Client> s3_client_;
            std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager_;
        };
    } // namespace

    std::shared_ptr<Aws::Client::ClientConfiguration> S3Handler::s3_handler_cfg_;

    S3Handler::S3Handler(const long requestTimeoutMs, const std::string region)
        : s3_client_(nullptr, ShutdownClient),
          transfer_manager_(nullptr, ShutdownTransferManager),
          executor_(nullptr, ShutdownExecutor),
          initialization_lock_()
    {
        // Load reading parameters
        buffer_size_ = s3ReadBufferSize;
        const char *bufferSizeStr = getenv("S3_BUFFER_SIZE");
        if (bufferSizeStr)
        {
            buffer_size_ = std::stoull(bufferSizeStr);
        }
        multi_part_download_ = true;
        const char *multi_download_disable_char =
            getenv("S3_DISABLE_MULTI_PART_DOWNLOAD");
        if (multi_download_disable_char)
        {
            std::string multi_download_disable_str(multi_download_disable_char);
            if (multi_download_disable_str == "ON")
            {
                multi_part_download_ = false;
            }
        }

        InitializeS3Client(requestTimeoutMs, region);

        this->max_keys_ = S3DefaultMaxKeys;
        this->last_marker_ = S3DefaultMarker;
    }

    S3Handler::~S3Handler() {}

    void S3Handler::InitializeS3Client()
    {
        const long requestTimeoutMs = -1;
        const std::string region = "";
        this->InitializeS3Client(requestTimeoutMs, region);
    }

    void S3Handler::InitializeS3Client(const long requestTimeoutMs, const std::string region)
    {
        std::lock_guard<std::mutex> lock(this->initialization_lock_);
        Aws::SDKOptions options;
        Aws::InitAPI(options);

        // Set up the request
        S3Handler::s3_handler_cfg_ = setUpS3Config(requestTimeoutMs, region);
        this->s3_client_ =
            std::shared_ptr<Aws::S3::S3Client>(new Aws::S3::S3Client(
                *S3Handler::s3_handler_cfg_,
                Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
                false));
    }

    void S3Handler::InitializeExecutor()
    {
        std::lock_guard<std::mutex> lock(this->initialization_lock_);
        this->executor_ =
            Aws::MakeShared<Aws::Utils::Threading::PooledThreadExecutor>(
                "executor", executorPoolSize);
    }

    void S3Handler::InitializeTransferManager()
    {
        std::shared_ptr<Aws::S3::S3Client> s3_client = GetS3Client();
        std::lock_guard<std::mutex> lock(this->initialization_lock_);

        Aws::Transfer::TransferManagerConfiguration transfer_config(
            this->GetExecutor().get());
        transfer_config.s3Client = s3_client;
        // This buffer is what we used to initialize streambuf and is in memory
        transfer_config.bufferSize = s3MultiPartDownloadChunkSize;
        transfer_config.transferBufferMaxHeapSize =
            (executorPoolSize + 1) * s3MultiPartDownloadChunkSize;
        this->transfer_manager_ =
            Aws::Transfer::TransferManager::Create(transfer_config);
    }

    std::shared_ptr<Aws::S3::S3Client> S3Handler::GetS3Client()
    {
        if (this->s3_client_.get() == nullptr)
        {
            this->InitializeS3Client();
        }
        return this->s3_client_;
    }

    std::shared_ptr<Aws::Utils::Threading::PooledThreadExecutor>
    S3Handler::GetExecutor()
    {
        if (this->executor_.get() == nullptr)
        {
            this->InitializeExecutor();
        }
        return this->executor_;
    }

    std::shared_ptr<Aws::Transfer::TransferManager>
    S3Handler::GetTransferManager()
    {
        if (this->transfer_manager_.get() == nullptr)
        {
            this->InitializeTransferManager();
        }
        return this->transfer_manager_;
    }

    size_t S3Handler::GetFileSize(const std::string &bucket,
                                  const std::string &object)
    {
        Aws::S3::Model::HeadObjectRequest headObjectRequest;
        headObjectRequest.WithBucket(bucket.c_str()).WithKey(object.c_str());
        Aws::S3::Model::HeadObjectOutcome headObjectOutcome =
            this->GetS3Client()->HeadObject(headObjectRequest);
        if (headObjectOutcome.IsSuccess())
        {
            return headObjectOutcome.GetResult().GetContentLength();
        }
        Aws::String const &error_aws = headObjectOutcome.GetError().GetMessage();
        std::string error_str(error_aws.c_str(), error_aws.size());
        throw std::invalid_argument(error_str);
        return 0;
    }

    void S3Handler::ClearMarker() { this->last_marker_ = S3DefaultMarker; }

    void S3Handler::S3Read(const std::string &file_url, std::string *result)
    {
        std::string bucket, object;
        parseS3Path(file_url, &bucket, &object);
        S3FS s3handler(bucket, object, multi_part_download_,
                       GetTransferManager(), GetS3Client());

        uint64_t offset = 0;
        uint64_t result_size = 0;
        uint64_t file_size = this->GetFileSize(bucket, object);
        std::size_t part_count = (std::max)(
            static_cast<size_t>((file_size + buffer_size_ - 1) / buffer_size_),
            static_cast<std::size_t>(1));
        result->resize(file_size);

        for (int i = 0; i < part_count; i++)
        {

            offset = result_size;

            size_t buf_len = std::min<size_t>(buffer_size_, file_size - result_size);

            size_t read_len =
                s3handler.read(offset, buf_len, (char *)(result->data()) + offset);

            result_size += read_len;

            if (result_size == file_size)
            {
                break;
            }

            if (read_len != buf_len)
            {
                std::cout << "Result size and buffer size did not match";
                break;
            }
        }
    }

    void S3Handler::ListFiles(const std::string &file_url,
                              std::vector<std::string> *filenames)
    {
        Aws::String bucket, prefix;
        parseS3Path(file_url, &bucket, &prefix);

        Aws::S3::Model::ListObjectsRequest listObjectsRequest;
        listObjectsRequest.WithBucket(bucket)
            .WithPrefix(prefix)
            .WithMaxKeys(this->max_keys_)
            .WithMarker(this->last_marker_);
        std::cerr << "max_keys_: " << this->max_keys_ << std::endl;
        std::cerr << "last_marker_: " << this->last_marker_ << std::endl;

        Aws::S3::Model::ListObjectsOutcome listObjectsOutcome =
            this->GetS3Client()->ListObjects(listObjectsRequest);
        if (!listObjectsOutcome.IsSuccess())
        {
            Aws::String const &error_aws =
                listObjectsOutcome.GetError().GetMessage();
            throw std::invalid_argument(error_aws);
        }

        Aws::S3::Model::ListObjectsResult listObjectsResult = listObjectsOutcome.GetResult();
        Aws::Vector<Aws::S3::Model::Object> objects = listObjectsResult.GetContents();
        if (!objects.empty())
        {
            for (const Aws::S3::Model::Object &object : objects)
            {
                if (object.GetKey().back() == '/') // ignore folders
                {
                    continue;
                }
                Aws::String entry = "s3://" + bucket + "/" + object.GetKey();
                filenames->push_back(entry.c_str());
            }
            this->last_marker_ = listObjectsResult.GetContents().back().GetKey();
        }
    }
} // namespace torchdata
