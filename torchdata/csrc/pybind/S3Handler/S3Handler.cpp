#include "S3Handler.h"

namespace torchdata
{
    namespace
    {
        static const size_t S3DefaultBufferSize = 128 * 1024 * 1024;                 // 128 MB
        static const uint64_t S3DefaultMultiPartDownloadChunkSize = 5 * 1024 * 1024; // 5 MB
        static const int executorPoolSize = 25;
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
        private:
            std::string bucket_name_;
            std::string object_name_;
            bool use_multi_part_download_;
            std::shared_ptr<Aws::S3::S3Client> s3_client_;
            std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager_;

        public:
            S3FS(const std::string &bucket, const std::string &object,
                 const bool use_multi_part_download,
                 std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager,
                 std::shared_ptr<Aws::S3::S3Client> s3_client)
                : bucket_name_(bucket),
                  object_name_(object),
                  use_multi_part_download_(use_multi_part_download),
                  transfer_manager_(transfer_manager),
                  s3_client_(s3_client) {}

            size_t Read(uint64_t offset, size_t n, char *buffer)
            {
                if (use_multi_part_download_)
                {
                    return ReadTransferManager(offset, n, buffer);
                }
                else
                {
                    return ReadS3Client(offset, n, buffer);
                }
            }

            size_t ReadS3Client(uint64_t offset, size_t n, char *buffer)
            {
                Aws::S3::Model::GetObjectRequest getObjectRequest;

                getObjectRequest.WithBucket(bucket_name_.c_str())
                    .WithKey(object_name_.c_str());

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
                Aws::S3::Model::GetObjectOutcome getObjectOutcome = s3_client_->GetObject(getObjectRequest);

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

            size_t ReadTransferManager(uint64_t offset, size_t n, char *buffer)
            {
                auto create_stream_fn = [&]() { // create stream lambda fn
                    return Aws::New<S3UnderlyingStream>(
                        "S3ReadStream",
                        Aws::New<Aws::Utils::Stream::PreallocatedStreamBuf>(
                            "S3ReadStream", reinterpret_cast<unsigned char *>(buffer),
                            n));
                }; // This buffer is what we used to initialize streambuf and is in memory

                std::shared_ptr<Aws::Transfer::TransferHandle> downloadHandle =
                    transfer_manager_.get()->DownloadFile(
                        bucket_name_.c_str(), object_name_.c_str(), offset,
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
        };
    } // namespace

    std::shared_ptr<Aws::Client::ClientConfiguration> S3Handler::s3_handler_cfg_;

    S3Handler::S3Handler(const long requestTimeoutMs, const std::string region)
        : s3_client_(nullptr, ShutdownClient),
          transfer_manager_(nullptr, ShutdownTransferManager),
          executor_(nullptr, ShutdownExecutor)
    {
        initialization_lock_ = std::shared_ptr<std::mutex>(new std::mutex());

        // Load reading parameters
        buffer_size_ = S3DefaultBufferSize;
        const char *bufferSizeStr = getenv("S3_BUFFER_SIZE");
        if (bufferSizeStr)
        {
            buffer_size_ = std::stoull(bufferSizeStr);
        }
        use_multi_part_download_ = true;
        const char *use_multi_part_download_char =
            getenv("S3_MULTI_PART_DOWNLOAD");
        if (use_multi_part_download_char)
        {
            std::string use_multi_part_download_str(use_multi_part_download_char);
            if (use_multi_part_download_str == "OFF")
            {
                use_multi_part_download_ = false;
            }
        }

        Aws::SDKOptions options;
        options.loggingOptions.logLevel = Utils::Logging::LogLevel::Trace;
        Aws::InitAPI(options);
        S3Handler::s3_handler_cfg_ = setUpS3Config(requestTimeoutMs, region);
        InitializeS3Client();

        last_marker_ = S3DefaultMarker;
    }

    S3Handler::~S3Handler() {}

    void S3Handler::InitializeS3Client()
    {
        std::lock_guard<std::mutex> lock(*initialization_lock_);
        s3_client_ =
            std::shared_ptr<Aws::S3::S3Client>(
                new Aws::S3::S3Client(
                    *S3Handler::s3_handler_cfg_,
                    Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
                    false));
    }

    void S3Handler::InitializeExecutor()
    {
        executor_ =
            Aws::MakeShared<Aws::Utils::Threading::PooledThreadExecutor>(
                "executor", executorPoolSize);
    }

    void S3Handler::InitializeTransferManager()
    {
        std::shared_ptr<Aws::S3::S3Client> s3_client = GetS3Client();
        std::lock_guard<std::mutex> lock(*initialization_lock_);

        Aws::Transfer::TransferManagerConfiguration transfer_config(
            GetExecutor().get());
        transfer_config.s3Client = s3_client;
        // This buffer is what we used to initialize streambuf and is in memory
        transfer_config.bufferSize = S3DefaultMultiPartDownloadChunkSize;
        transfer_config.transferBufferMaxHeapSize =
            (executorPoolSize + 1) * S3DefaultMultiPartDownloadChunkSize;
        transfer_manager_ =
            Aws::Transfer::TransferManager::Create(transfer_config);
    }

    std::shared_ptr<Aws::S3::S3Client> S3Handler::GetS3Client()
    {
        if (s3_client_.get() == nullptr)
        {
            InitializeS3Client();
        }
        return s3_client_;
    }

    std::shared_ptr<Aws::Utils::Threading::PooledThreadExecutor>
    S3Handler::GetExecutor()
    {
        if (executor_.get() == nullptr)
        {
            InitializeExecutor();
        }
        return executor_;
    }

    std::shared_ptr<Aws::Transfer::TransferManager>
    S3Handler::GetTransferManager()
    {
        if (transfer_manager_.get() == nullptr)
        {
            InitializeTransferManager();
        }
        return transfer_manager_;
    }

    size_t S3Handler::GetFileSize(const std::string &bucket,
                                  const std::string &object)
    {
        Aws::S3::Model::HeadObjectRequest headObjectRequest;
        headObjectRequest.WithBucket(bucket.c_str()).WithKey(object.c_str());
        Aws::S3::Model::HeadObjectOutcome headObjectOutcome =
            GetS3Client()->HeadObject(headObjectRequest);
        if (headObjectOutcome.IsSuccess())
        {
            return headObjectOutcome.GetResult().GetContentLength();
        } else {
            Aws::String const &error_aws = headObjectOutcome.GetError().GetMessage();
            std::string error_str(error_aws.c_str(), error_aws.size());
            throw std::invalid_argument(error_str);
            return 0;
        }
    }

    void S3Handler::ClearMarker() { last_marker_ = S3DefaultMarker; }

    void S3Handler::S3Read(const std::string &file_url, std::string *result)
    {
        std::string bucket, object;
        parseS3Path(file_url, &bucket, &object);
        S3FS s3fs(bucket, object, use_multi_part_download_,
                  GetTransferManager(), GetS3Client());

        uint64_t offset = 0;
        uint64_t result_size = 0;
        uint64_t file_size = GetFileSize(bucket, object);
        size_t part_count = (std::max)(
            static_cast<size_t>((file_size + buffer_size_ - 1) / buffer_size_),
            static_cast<size_t>(1));
        result->resize(file_size);

        for (int i = 0; i < part_count; i++)
        {
            offset = result_size;

            size_t buf_len = std::min<size_t>(buffer_size_, file_size - result_size);

            size_t read_len =
                s3fs.Read(offset, buf_len, (char *)(result->data()) + offset);

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
            .WithMarker(last_marker_);

        Aws::S3::Model::ListObjectsOutcome listObjectsOutcome =
            GetS3Client()->ListObjects(listObjectsRequest);
        if (!listObjectsOutcome.IsSuccess())
        {
            Aws::String const &error_aws =
                listObjectsOutcome.GetError().GetMessage();
            throw std::invalid_argument(error_aws);
        }

        Aws::Vector<Aws::S3::Model::Object> objects = listObjectsOutcome.GetResult().GetContents();
        if (objects.empty())
        {
            return;
        }
        for (const Aws::S3::Model::Object &object : objects)
        {
            if (object.GetKey().back() == '/') // ignore folders
            {
                continue;
            }
            Aws::String entry = "s3://" + bucket + "/" + object.GetKey();
            filenames->push_back(entry.c_str());
        }
        last_marker_ = objects.back().GetKey();

        // extreme cases when all objects are folders
        if (filenames->size() == 0)
        {
            ListFiles(file_url, filenames);
        }
    }
} // namespace torchdata
