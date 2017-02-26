#ifndef MATRIX_LOGGING_H_
#define MATRIX_LOGGING_H_

#include <cstdlib>
#include <sstream>

namespace snoopy {
const int INFO = 0;
const int WARNING = 1;
const int ERROR = 2;
const int FATAL = 3;

namespace internal {

string debug_info[4] = {"INFO", "WARNING", "ERROR", "FATAL"};

class LogMessage : public std::basic_ostringstream<char> {
    public:
        LogMessage(const char * fname, const int fline, const int severity):
            _fname(fname),
            _fline(fline),
            _severity(severity) {}

        ~LogMessage() {
           gen_message(); 
        }
        void gen_message() {
            fprintf(stderr, "[%s %s: %d] %s\n", debug_info[_severity].c_str(), _fname, _fline, str().c_str());
        }

    private:
        const char * _fname;
        int  _fline;
        int  _severity;
};

class LogMessageFatal : public LogMessage {
    public:
        LogMessageFatal(const char * fname, const int fline, const int severity):
            LogMessage(fname, fline, severity) {}

        ~LogMessageFatal() {
            gen_message();
            std::abort();
        }
};

#define LOG_INFO \
    ::snoopy::internal::LogMessage(__FILE__, __LINE__, snoopy::INFO)
#define LOG_WARNING \
    ::snoopy::internal::LogMessage(__FILE__, __LINE__, snoopy::WARNING)
#define LOG_ERROR \
    ::snoopy::internal::LogMessage(__FILE__, __LINE__, snoopy::ERROR)
#define LOG_FATAL \
    ::snoopy::internal::LogMessageFatal(__FILE__, __LINE__, snoopy::FATAL)


#define SN_CHECK(x) \
    if (!(x)) \
        LOG_FATAL << "Check failed: " #x << "!";

#define CHECK_EQ(a, b) SN_CHECK((a == b))
#define CHECK_NE(a, b) SN_CHECK((a != b))
#define CHECK_LT(a, b) SN_CHECK((a < b))
#define CHECK_LE(a, b) SN_CHECK((a <= b))
#define CHECK_GT(a, b) SN_CHECK((a > b))
#define CHECK_GE(a, b) SN_CHECK((a >= b))
#define CHECK_NOTNULL(a) SN_CHECK((a != NULL))

}
}

#endif

