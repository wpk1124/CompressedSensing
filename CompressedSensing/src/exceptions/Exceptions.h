#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H
#include <stdexcept>
#include <sstream>

using namespace std;

namespace CS {
namespace exception {

class UnknownTypeException : public runtime_error {
public:
	UnknownTypeException(string type) : runtime_error(type.c_str()) {}

	virtual const char* what() const throw();
private:
	static ostringstream msg;
};

class JaiCameraException : public exception {
public:
	JaiCameraException(string msg, int errorCode);
	virtual const char* what() const throw();
private:
	string functionName;
	int errorCode;
	static ostringstream msg;
};

}}

#endif /*EXCEPTIONS_H*/