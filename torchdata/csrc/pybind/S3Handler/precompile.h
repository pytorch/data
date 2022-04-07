#ifndef TORCHDATA_S3_IO_H
#define TORCHDATA_S3_IO_H

#include <fstream>
#include <mutex>
#include <string>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSAuthSigner.h>
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

#endif // TORCHDATA_S3_IO_H
