//--------------------------------------------------------------------//
//   Copyright (c) 2012, Tim Warburton, David Medina                  //
//   All rights reserved.                                             //
//   License Located in:                                              //
//      https://github.com/dmed256/OCL-Library/blob/master/LICENSE    //
//--------------------------------------------------------------------//

#ifndef OCL_LIBRARY
#define OCL_LIBRARY

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <sys/stat.h>
#include <stdarg.h>

#if  defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "ocl.h"

//-----------------------------//
//          OCL_SETUP          //
//-----------------------------//
ocl_setup::ocl_setup(){
  allocs = new int[1];
  allocs[0] = 1;

  shortInfo = "";
  longInfo = "";

  findDevices();

  question1 = "Print Detailed Information?\n"\
    "\t    i j - Platform i, Device j\n"\
    "\t[Other] - Shows Detailed Information\n"\
    "Input: ";
  question2 = "    i j - Platform i, Device j\n"\
    "Input: ";
}

ocl_setup::ocl_setup(const ocl_setup& s){
  allocs = NULL;

  *this = s;
}

ocl_setup::~ocl_setup(){
  if(allocs == NULL)
    return;

  if(allocs[0] > 1)
    allocs[0]--;
  else{
    delete[] allocs;
    destructor();
  }
}

ocl_setup& ocl_setup::operator=(const ocl_setup& s){
  copyCheck(s.allocs);
  shortInfo = s.shortInfo;
  longInfo  = s.longInfo;
  question1 = s.question1;
  question2 = s.question2;
  pID = s.pID;
  dID = s.dID;
  pSize = s.pSize;
  dSize = s.dSize;
  err = s.err;
  return *this;
}

void ocl_setup::destructor(){
  delete[] pID;
  for(int i=0;i<pSize;i++)
    delete[] dID[i];
  delete[] dID;
  delete[] dSize;

}

void ocl_setup::copyCheck(int* allocs2){
  if(allocs == NULL)
    allocs = allocs2;
  else if(allocs[0] > 1){
    allocs[0]--;
    allocs = allocs2;
  }
  else{
    delete[] allocs;
    allocs = allocs2;
    destructor();
  }
  allocs[0]++;
}

void ocl_setup::findDevices(){
  const int maxP = 64;
  const int maxD = 64;
  cl_platform_id pID2[maxP];
  cl_device_id dID2[maxD];

  clGetPlatformIDs(maxP, pID2, &pSize);
  pID = new cl_platform_id[pSize];
  dID = new cl_device_id*[pSize];
  dSize = new cl_uint[pSize];

  for(int i=0;i<pSize;i++){
    pID[i] = pID2[i];

    clGetDeviceIDs(pID[i],CL_DEVICE_TYPE_ALL, maxD, dID2, dSize+i);
    dID[i] = new cl_device_id[dSize[i]];
    for(int j=0;j<dSize[i];j++)
      dID[i][j] = dID2[j];
  }
}

void ocl_setup::findDeviceInformation(){
  const int bSize = 8192;
  char buffer[bSize];
  cl_ulong buf_ulong;
  cl_uint buf_uint;
  std::stringstream str;
  cl_int err;

  std::string line = "";
  line.append(118,'-');
  line.append(1,'\n');
  std::string starLine = "\t";
  starLine.append(110,'*');
  starLine.append(1,'\n');

  str << line;
  for (int i=0;i<pSize;i++){
    str << "PLATFORM: " << i << '\n';
    ocl::printError("PLATFORM NAME",clGetPlatformInfo(pID[i], CL_PLATFORM_NAME , bSize, buffer, NULL));
    str << "\tPLATFORM_NAME       : " << buffer << '\n';

    ocl::printError("PLATFORM VERSION",clGetPlatformInfo(pID[i], CL_PLATFORM_VERSION, bSize, buffer, NULL));
    str << "\tPLATFORM_VERSION    : " << buffer << '\n';
    str << starLine;

    for(int j=0;j<dSize[i];j++){
      str << "\tDEVICE: " << j << '\n';

      ocl::printError("DEVICE_NAME",clGetDeviceInfo(dID[i][j], CL_DEVICE_NAME, bSize, buffer, NULL));
      str << "\t\tDEVICE_NAME                        : " << buffer << '\n';

      ocl::printError("DEVICE_VENDOR",clGetDeviceInfo(dID[i][j], CL_DEVICE_VENDOR, bSize, buffer, NULL));
      str << "\t\tDEVICE_VENDOR                      : " << buffer << '\n';

      ocl::printError("DEVICE_VERSION",clGetDeviceInfo(dID[i][j], CL_DEVICE_VERSION, bSize, buffer, NULL));
      str << "\t\tDEVICE_VERSION                     : " << buffer << '\n';
      str << starLine;
    }
    str << line;
  }

  shortInfo = str.str();

  str.str("");

  for(int i=0;i<pSize;i++){
    str << line;
    str << "PLATFORM            : " << i << '\n';

    ocl::printError("PLATFORM PROFILE",clGetPlatformInfo(pID[i], CL_PLATFORM_PROFILE, bSize, buffer, NULL));
    str << "PLATFORM_PROFILE    : " << buffer << '\n';

    ocl::printError("PLATFORM VERSION",clGetPlatformInfo(pID[i], CL_PLATFORM_VERSION, bSize, buffer, NULL));
    str << "PLATFORM_VERSION    : " << buffer << '\n';

    ocl::printError("PLATFORM NAME",clGetPlatformInfo(pID[i], CL_PLATFORM_NAME , bSize, buffer, NULL));
    str << "PLATFORM_NAME       : " << buffer << '\n';

    ocl::printError("PLATFORM VENDOR",clGetPlatformInfo(pID[i], CL_PLATFORM_VENDOR , bSize, buffer, NULL));
    str << "PLATFORM_VENDOR     : " << buffer << '\n';

    ocl::printError("PLATFORM EXTENSIONS",clGetPlatformInfo(pID[i], CL_PLATFORM_EXTENSIONS, bSize, buffer, NULL));
    str << "PLATFORM_EXTENSIONS : " << buffer << "\n\n";

    str << starLine;

    for(int j=0;j<dSize[i];j++){
      str << "\tDevice                             : " << j << std::endl;

      ocl::printError("DEVICE_NAME",clGetDeviceInfo(dID[i][j], CL_DEVICE_NAME, bSize, buffer, NULL));
      str << "\tDEVICE_NAME                        : " <<  buffer << std::endl;

      ocl::printError("DEVICE_VENDOR",clGetDeviceInfo(dID[i][j], CL_DEVICE_VENDOR, bSize, buffer, NULL));
      str << "\tDEVICE_VENDOR                      : " <<  buffer << std::endl;

      ocl::printError("DEVICE_VERSION",clGetDeviceInfo(dID[i][j], CL_DEVICE_VERSION, bSize, buffer, NULL));
      str << "\tDEVICE_VERSION                     : " <<  buffer << std::endl;

      ocl::printError("DRIVER_VERSION",clGetDeviceInfo(dID[i][j], CL_DRIVER_VERSION, bSize, buffer, NULL));
      str << "\tDRIVER_VERSION                     : " <<  buffer << std::endl;

      ocl::printError("DEVICE_MAX_COMPUTE_UNITS",
					  clGetDeviceInfo(dID[i][j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL));
      str << "\tDEVICE_MAX_COMPUTE_UNITS           : " << (unsigned int)buf_uint << std::endl;

      ocl::printError("DEVICE_MAX_CLOCK_FREQUENCY",
					  clGetDeviceInfo(dID[i][j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL));
      str << "\tDEVICE_MAX_CLOCK_FREQUENCY         : " <<  (unsigned int)buf_uint << std::endl;

      ocl::printError("DEVICE_GLOBAL_MEM_SIZE",
					  clGetDeviceInfo(dID[i][j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL));
      str << "\tDEVICE_GLOBAL_MEM_SIZE             : " <<  (unsigned long long)buf_ulong << std::endl;

      ocl::printError("DEVICE_MAX_WORK_GROUP_SIZE",
					  clGetDeviceInfo(dID[i][j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(buf_ulong), &buf_ulong, NULL));
      str << "\tDEVICE_MAX_WORK_GROUP_SIZE         : " <<  (unsigned long long)buf_ulong << std::endl;

      size_t workitem_dims;
      ocl::printError("GET_MAX_ITEM_DIMENSIONS",
					  clGetDeviceInfo(dID[i][j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workitem_dims), &workitem_dims, NULL));
      str << "\tCL_DEVICE_MAX_WORK_ITEM_DIMENSIONS : " <<  (unsigned int) workitem_dims << std::endl;

      size_t workitem_size[3];
      ocl::printError("GET_MAX_WORK_ITEM_SIZES",
					  clGetDeviceInfo(dID[i][j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL));
      str << "\tCL_DEVICE_MAX_WORK_ITEM_SIZES      : "
		  << (int) workitem_size[0] << " / "
		  << (int) workitem_size[1] << " / "
		  << (int) workitem_size[2] << std::endl;
      str << starLine;
    }
  }
  str << line;

  longInfo = str.str();
}

ocl_device ocl_setup::displayDevices(){
  if(shortInfo.empty())
    findDeviceInformation();

  char buffer[8192];
  int di = 1,dj = 3;
  int i,j;

  std::cout << shortInfo;

  printf(question1.c_str(),di,dj);
  gets(buffer);

  if(sscanf(buffer,"%d %d",&i,&j) == 2)
    return ocl_device(pID[i],dID[i][j]);

  std::cout << longInfo;
  printf(question2.c_str(),di,dj);
  gets(buffer);

  while(sscanf(buffer,"%d %d",&i,&j) != 2){
    std::cout << "Wrong Input, please try again\n";
    printf(question2.c_str(),di,dj);
    gets(buffer);
  }

  return ocl_device(pID[i],dID[i][j]);
}

ocl_device ocl_setup::getDevice(int p, int d){
  return ocl_device(pID[p],dID[p][d]);
}

cl_platform_id ocl_setup::getPlatformID(int p){
  return pID[p];
}

cl_device_id ocl_setup::getDeviceID(int p, int d){
  return dID[p][d];
}

//-----------------------------//
//          OCL_KERNEL          //
//-----------------------------//

ocl_kernel::ocl_kernel(){
  allocs = NULL;
  inputSize = NULL;
}

ocl_kernel::ocl_kernel(const ocl_kernel& k){
  allocs = NULL;
  inputSize = NULL;

  *this = k;
}

ocl_kernel::ocl_kernel(ocl_device* d, std::string str){
  allocs = new int[1];
  allocs[0] = 1;

  format = 1;
  setup(d,str);
}

ocl_kernel::ocl_kernel(ocl_device* d, std::string str, std::string fstr){
  allocs = new int[1];
  allocs[0] = 1;

  flags = fstr;
  format = 1;
  setup(d,str);
}

ocl_kernel::ocl_kernel(ocl_device* d, std::string str, int n){
  allocs = new int[1];
  allocs[0] = 1;

  format = n;
  setup(d,str);
}

ocl_kernel::ocl_kernel(ocl_device* d, std::string str, std::string fstr, int n){
  allocs = new int[1];
  allocs[0] = 1;

  flags = fstr;
  format = n;
  setup(d,str);
}

ocl_kernel::~ocl_kernel(){
  if(allocs == NULL)
    return;

  if(allocs[0] > 1)
    allocs[0]--;
  else{
    delete[] allocs;
    destructor();
  }
}

ocl_kernel& ocl_kernel::operator=(const ocl_kernel& k){
  copyCheck(k.allocs);
  device = k.device;
  flags = k.flags;
  kernel = k.kernel;
  program = k.program;
  name = k.name;
  function = k.function;
  inputs = k.inputs;
  format = k.format;
  events = k.events;

  if(inputSize == NULL)
    inputSize = new int[inputs];
  for(int i=0;i<inputs;i++)
    inputSize[i] = k.inputSize[i];

  inputType = k.inputType;
  dims = k.dims;
  lDims = k.lDims;
  gDims = k.gDims;

  return *this;
}

void ocl_kernel::destructor(){
  if(function.compare("")){
    delete[] inputSize;

    for(int i=0;i<events->size();i++)
      clReleaseEvent((*events)[i]);
    delete[] events;

    delete[] lDims;
    delete[] gDims;

    clReleaseKernel(kernel);
    clReleaseProgram(program);
  }
}

void ocl_kernel::copyCheck(int* allocs2){
  if(allocs == NULL)
    allocs = allocs2;
  else if(allocs[0] > 1){
    allocs[0]--;
    allocs = allocs2;
  }
  else{
    delete[] allocs;
    allocs = allocs2;
    destructor();
  }
  allocs[0]++;
}

void ocl_kernel::setup(ocl_device* d, std::string str){
  device = d;
  dims = 1;
  lDims = new size_t[3];
  gDims = new size_t[3];
  lDims[0] = gDims[0] = 0;
  lDims[1] = gDims[1] = 0;
  lDims[2] = gDims[2] = 0;

  cl_int err;

  FILE* file = fopen(str.c_str(),"r");
  if(file){
    struct stat statbuf;
    stat(str.c_str(),&statbuf);
    char* tmp = new char[statbuf.st_size+1];
    fread(tmp,statbuf.st_size+1,1,file);
    tmp[statbuf.st_size] = '\0';

    function = tmp;
    delete[] tmp;
    fclose(file);
  }
  else
    function = str;

  getKernelInformation(function);

  if(format)
    function = ocl::getFormattedKernel(function);

  const char* cFunction = function.c_str();
  const size_t cLength = function.length();
  cl_device_id dID = device->getDeviceID();

  cl_int err2;
  program = clCreateProgramWithSource(device->getContext(),1,&cFunction,&cLength,&err2);

  err = clBuildProgram(program,1,&dID,flags.c_str(),NULL,NULL);

  char* log;
  size_t logSize;

  err = clGetProgramBuildInfo(program, device->getDeviceID(), CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
  ocl::printError("OCL_Kernel ("+name+") : Building Program",err);
  if(logSize > 2){
    log = new char[logSize+1];

    err = clGetProgramBuildInfo(program, device->getDeviceID(), CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
    ocl::printError("OCL_Kernel ("+name+") : Building Program",err);
    log[logSize] = '\0';

    if(format)
      std::cout << function << std::endl;
    else
      std::cout << ocl::getFormattedKernel(function) << std::endl;

    std::cout << "OCL_Kernel (" << name << "): Build Log\n" << log;

    delete[] log;
  }

  ocl::printError("OCL_Kernel ("+name+") : Constructing Program",err2);

  kernel = clCreateKernel(program,name.c_str(),&err);
  ocl::printError("OCL_Kernel : Creating Kernel",err);

  events = new std::vector<cl_event>[1];
}

void ocl_kernel::getKernelInformation(std::string str){
  int start = str.find("__kernel",0) + 9;
  int end;

  if(start < 9){
    std::cout << "OCL_Kernel : Getting Kernel Information Error\n"
			  << "Exiting Program.\n";
    exit(1);
  }

  std::string dm = " \t\n(*";

  while(dm.find(str[start],0) == std::string::npos)
    start++;
  while(dm.find(str[start],0) != std::string::npos)
    start++;
  end = start;
  while(dm.find(str[end],0) == std::string::npos)
    end++;

  name = str.substr(start,end-start);

  start = end;
  inputs = 0;

  while(str[end] - ')'){
    if(str[end] == ',' || (dm.find(str[end]) == std::string::npos)){
      end++;
      break;
    }
    end++;
  }

  if(str[end] - ')')
    inputs++;

  while(str[end] - ')'){
    if(str[end] == ',')
      inputs++;
    end++;
  }

  int* pointer = new int[inputs]();
  inputType.resize(inputs);
  inputSize = new int[inputs];
  int pos = start+1;

  for(int i=0;i<inputs;i++){
    while(str[pos] != ',' && str[pos] != ')'){
      if(str[pos] == '*')
		pointer[i]++;
      pos++;
    }
    inputSize[i] = pos++;

    while(dm.find(str[inputSize[i]],0) != std::string::npos)
      inputSize[i]--;
    while(dm.find(str[inputSize[i]],0) == std::string::npos)
      inputSize[i]--;
    while(dm.find(str[inputSize[i]],0) != std::string::npos)
      inputSize[i]--;
  }

  for(int i=0;i<inputs;i++){
    if(pointer[i]){
      inputType[i] = ocl::type[6];
      inputSize[i] = sizeofType("cl_mem");
    }
    else{
      pos = inputSize[i];

      while(dm.find(str[pos],0) == std::string::npos){
		pos--;
      }

      inputType[i] = str.substr(pos+1,inputSize[i]-pos);
      inputSize[i] = sizeofType(inputType[i]);
    }
  }

  delete[] pointer;
}

int ocl_kernel::sizeofType(std::string type){
  int left = 0;
  int right = ocl::types-1;
  int current,check;

  while(0 <= left && right < ocl::types){
    current = (left + right)/2;
    check = type.compare(ocl::type[current]);

    if(!check)
      return ocl::typeSize[current];
    else if(0 < check)
      left = current;
    else
      right = current;

    if(current == (left+right)/2){
      std::cout << "OCL_Kernel ("+name+") : Type <" << type << "> not found in oclInfo\n"
				<< "Exiting Program.\n";
      exit(1);
    }
  }
}

void ocl_kernel::setDims(size_t lDim, size_t gDim){
  lDims[0] = lDim;
  gDims[0] = gDim;
}

void ocl_kernel::setDims(int pos, size_t lDim, size_t gDim){
  if(pos < 0 || pos > 2){
    std::cout << "OCL_Kernel ("+name+") : Dimension " << pos << " is out of bounds\n"
			  << "Exiting Program.\n";
	exit(1);
  }

  if(dims < (pos+1))
	dims = pos+1;
  lDims[pos] = lDim;
  gDims[pos] = gDim;
}

void ocl_kernel::setDims(int num, size_t* lDim, size_t* gDim){
  if(num < 0 || num > 2){
    std::cout << "OCL_Kernel ("+name+") : Dimension " << num << " is out of bounds\n"
			  << "Exiting Program.\n";
	exit(1);
  }

  dims = num;
  for(int i=0;i<num;i++){
    lDims[i] = lDim[i];
    gDims[i] = gDim[i];
  }
}

void ocl_kernel::setArgs(void* x, ...){
  va_list list;
  va_start(list,x);
  ocl::printError("OCL_Kernel ("+name+") : Setting Kernel Arguments",
				  clSetKernelArg(kernel,0,inputSize[0],x));
  for(int i=1;i<inputs;i++)
    ocl::printError("OCL_Kernel ("+name+") : Setting Kernel Arguments",
					clSetKernelArg(kernel,i,inputSize[i],(void*) va_arg(list,void*)));

  va_end(list);
}

void ocl_kernel::setArg(int pos, void* arg){
  if(pos >= inputs || pos < 0)
    ocl::printError("OCL_Kernel ("+name+") : Incorrect Kernel Argument Position",15);
  ocl::printError("OCL_Kernel ("+name+") : Setting Kernel Arguments",
				  clSetKernelArg(kernel,pos,inputSize[pos],arg));
}

std::string ocl_kernel::getArgType(int pos){
  if(pos >= inputs || pos < 0)
    ocl::printError("OCL_Kernel ("+name+") : Incorrect Kernel Argument Position",15);
  return inputType[pos];
}

void ocl_kernel::run(size_t lDim, size_t gDim){
  setDims(lDim,gDim);
  run();
}

void ocl_kernel::run(int num, size_t* lDim, size_t* gDim){
  setDims(num,lDim,gDim);
  run();
}

void ocl_kernel::run(){
  const cl_int c_dims = dims;
  const size_t c_lDims[3] = {lDims[0],lDims[1],lDims[2]};
  const size_t c_gDims[3] = {gDims[0],gDims[1],gDims[2]};
  ocl::printError("OCL_Kernel ("+name+") : Kernel Run",
				  clEnqueueNDRangeKernel(device->getCommandQueue(),kernel,c_dims,NULL,c_gDims,c_lDims,0,NULL,NULL));
}

int ocl_kernel::timedRun(size_t lDim, size_t gDim){
  setDims(lDim,gDim);
  return timedRun();
}

int ocl_kernel::timedRun(int num, size_t* lDim, size_t* gDim){
  setDims(num,lDim,gDim);
  return timedRun();
}

int ocl_kernel::timedRun(){
  const cl_int c_dims = dims;
  const size_t c_lDims[3] = {lDims[0],lDims[1],lDims[2]};
  const size_t c_gDims[3] = {gDims[0],gDims[1],gDims[2]};
  cl_event event;

  ocl::printError("OCL_Kernel ("+name+") : Kernel Timed Run",
				  clEnqueueNDRangeKernel(device->getCommandQueue(),kernel,c_dims,NULL,c_gDims,c_lDims,0,NULL,&event));

  events->push_back(event);

  return (events->size()-1);
}

float ocl_kernel::getRunTime(int i){
  cl_event event;
  cl_ulong start,end;

  if(i >= events->size())
    ocl::printError("OCL_Kernel ("+name+") : Kernel Get Run Time",15);

  event = (*events)[i];

  clWaitForEvents(1,&event);

  clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&start,NULL);
  clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&end,NULL);

  return (end-start)*1.e-6;
}

void ocl_kernel::printKernel(){
  std::cout << function << std::endl;
}

void ocl_kernel::printFormattedKernel(){
  if(format)
    std::cout << function << std::endl;
  else
    std::cout << ocl::getFormattedKernel(function) << std::endl;
}

cl_kernel ocl_kernel::getKernel(){
  return kernel;
}

cl_program ocl_kernel::getProgram(){
  return program;
}

std::string ocl_kernel::getName(){
  return name;
}

std::string ocl_kernel::getFunction(){
  return function;
}

std::string ocl_kernel::getFlags(){
  return flags;
}

int ocl_kernel::getWarpSize(){
  size_t ret;
  ocl::printError("OCL_DEVICE: Getting Warp Size",
				  clGetKernelWorkGroupInfo(kernel,device->getDeviceID(),CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
										   sizeof(size_t),&ret,NULL));
  return ret;
}

int ocl_kernel::getWavefrontSize(){
  return getWarpSize();
}

int ocl_kernel::getGroupSize(int p){
  return device->getGroupSize(p);
}

//-----------------------------//
//         OCL_DEVICE          //
//-----------------------------//
ocl_device::ocl_device(){
  allocs = NULL;
  pID = NULL;
  dID = NULL;
  groupSize = NULL;
}

ocl_device::ocl_device(const ocl_device& d){
  allocs = NULL;

  *this = d;
}

ocl_device::ocl_device(cl_platform_id p, cl_device_id d){
  allocs = new int[1];
  allocs[0] = 1;

  pID = p;
  dID = d;
  groupSize = new int[3];
  refresh();
}

ocl_device::~ocl_device(){
  if(allocs == NULL)
    return;

  if(allocs[0] > 1)
    allocs[0]--;
  else{
    delete[] allocs;
    destructor();
  }
}

ocl_device& ocl_device::operator=(const ocl_device& d){
  copyCheck(d.allocs);
  pID = d.pID;
  dID = d.dID;
  if(groupSize == NULL)
    groupSize = new int[3];

  context = d.context;
  commandQueue = d.commandQueue;
  groupSize[0] = d.groupSize[0];
  groupSize[1] = d.groupSize[1];
  groupSize[2] = d.groupSize[2];
  return *this;
}

void ocl_device::destructor(){
  if(groupSize != NULL)
    delete[] groupSize;
}

void ocl_device::copyCheck(int* allocs2){
  if(allocs == NULL)
    allocs = allocs2;
  else if(allocs[0] > 1){
    allocs[0]--;
    allocs = allocs2;
  }
  else{
    delete[] allocs;
    allocs = allocs2;
    destructor();
  }
  allocs[0]++;
}

void ocl_device::refresh(){
  context.create(&dID);
  commandQueue.create(context.getContext(),dID);
  size_t* tmp = new size_t[3];
  ocl::printError("GET_MAX_WORK_ITEM_SIZES",
				  clGetDeviceInfo(dID, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3*sizeof(size_t), tmp, NULL));
  groupSize[0] = tmp[0];
  groupSize[1] = tmp[1];
  groupSize[2] = tmp[2];
  delete[] tmp;
}

ocl_mem ocl_device::malloc(size_t s){
  cl_int err;
  cl_mem mem = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE , s, NULL, &err);
  ocl::printError("OCL_Device: Malloc",err);
  return ocl_mem(this,mem,s);
}

ocl_mem ocl_device::malloc(size_t s, cl_mem_flags f){
  cl_int err;
  cl_mem mem = clCreateBuffer(context.getContext(), f, s, NULL, &err);
  ocl::printError("OCL_Device: Malloc",err);
  return ocl_mem(this,mem,s);
}

void ocl_device::barrier(){
  ///////////////////////////////////
  //ocl::printError("OCL_Device: Malloc",clEnqueueBarrier(*commandQueue));
}

void ocl_device::finish(){
  commandQueue.finish();
}

void ocl_device::flush(){
  commandQueue.flush();
}

cl_platform_id ocl_device::getPlatformID(){
  return pID;
}

cl_device_id ocl_device::getDeviceID(){
  return dID;
}

cl_context ocl_device::getContext(){
  return context.getContext();
}

cl_command_queue ocl_device::getCommandQueue(){
  return commandQueue.getCommandQueue();
}

int ocl_device::getGroupSize(int p){
  if(p < 0 || 2 < p || groupSize == NULL)
    ocl::printError("OCL_DEVICE: Getting group size",15);
  return groupSize[p];
}

//-----------------------------//
//         OCL_CONTEXT         //
//-----------------------------//

ocl_context::ocl_context(){
  allocs = NULL;
}

ocl_context::ocl_context(const ocl_context& c){
  allocs = NULL;

  *this = c;
}

ocl_context::~ocl_context(){
  if(allocs == NULL)
    return;

  if(allocs[0] > 1)
    allocs[0]--;
  else{
    delete[] allocs;
    destructor();
  }
}

ocl_context& ocl_context::operator=(const ocl_context& c){
  copyCheck(c.allocs);
  context = c.context;
  return *this;
}

void ocl_context::destructor(){
  clReleaseContext(*context);
  delete[] context;
}

void ocl_context::copyCheck(int* allocs2){
  if(allocs == NULL)
    allocs = allocs2;
  else if(allocs[0] > 1){
    allocs[0]--;
    allocs = allocs2;
  }
  else{
    delete[] allocs;
    allocs = allocs2;
    destructor();
  }
  allocs[0]++;
}

void ocl_context::create(cl_device_id* dID){
  if(allocs == NULL){
    allocs = new int[1];
    allocs[0] = 1;
  }
  cl_int err;
  context = new cl_context[1];
  *context = clCreateContext(NULL,1,dID,NULL,NULL,&err);
  ocl::printError("OCL_Context: Creating Context",err);
}

cl_context ocl_context::getContext(){
  return *context;
}

//-----------------------------//
//      OCL_COMMANDQUEUE       //
//-----------------------------//

ocl_commandQueue::ocl_commandQueue(){
  allocs = NULL;
}

ocl_commandQueue::ocl_commandQueue(const ocl_commandQueue& cq){
  allocs = NULL;

  *this = cq;
}

ocl_commandQueue::~ocl_commandQueue(){
  if(allocs == NULL)
    return;

  if(allocs[0] > 1)
    allocs[0]--;
  else{
    delete[] allocs;
    destructor();
  }
}

ocl_commandQueue& ocl_commandQueue::operator=(const ocl_commandQueue& cq){
  copyCheck(cq.allocs);
  commandQueue = cq.commandQueue;
  return *this;
}

void ocl_commandQueue::destructor(){
  clReleaseCommandQueue(*commandQueue);
  delete[] commandQueue;
}

void ocl_commandQueue::copyCheck(int* allocs2){
  if(allocs == NULL)
    allocs = allocs2;
  else if(allocs[0] > 1){
    allocs[0]--;
    allocs = allocs2;
  }
  else{
    delete[] allocs;
    allocs = allocs2;
    destructor();
  }
  allocs[0]++;
}

void ocl_commandQueue::create(cl_context context, cl_device_id dID){
  if(allocs == NULL){
    allocs = new int[1];
    allocs[0] = 1;
  }
  cl_int err;
  commandQueue = new cl_command_queue[1];
  *commandQueue = clCreateCommandQueue(context,dID,CL_QUEUE_PROFILING_ENABLE,&err);
  ocl::printError("OCL_CommandQueue: Creating Command Queue",err);
}

void ocl_commandQueue::finish(){
  clFinish(*commandQueue);
}

void ocl_commandQueue::flush(){
  clFlush(*commandQueue);
}

cl_command_queue ocl_commandQueue::getCommandQueue(){
  return *commandQueue;
}

//-----------------------------//
//           OCL_MEM           //
//-----------------------------//

ocl_mem::ocl_mem(){
  allocs = NULL;
  device = NULL;
  memory = NULL;
  size   = -1;
}

ocl_mem::ocl_mem(const ocl_mem& m){
  allocs = NULL;

  *this = m;
}

ocl_mem::ocl_mem(ocl_device* d, cl_mem m, size_t s){
  allocs = new int[1];
  allocs[0] = 1;

  device = d;
  memory = m;
  size   = s;
}

ocl_mem::~ocl_mem(){
  if(allocs == NULL)
    return;

  if(allocs[0] > 1)
    allocs[0]--;
  else{
    delete[] allocs;
    destructor();
  }
}

ocl_mem& ocl_mem::operator=(const ocl_mem& m){
  copyCheck(m.allocs);
  device = m.device;
  memory = m.memory;
  size   = m.size;
  return *this;
}

void ocl_mem::destructor(){
  clReleaseMemObject(memory);
}

void ocl_mem::copyCheck(int* allocs2){
  if(allocs == NULL)
    allocs = allocs2;
  else if(allocs[0] > 1){
    allocs[0]--;
    allocs = allocs2;
  }
  else{
    delete[] allocs;
    allocs = allocs2;
    destructor();
  }
  allocs[0]++;
}

void ocl_mem::free(){
  if(size >= 0){
    delete[] allocs;
    clReleaseMemObject(memory);
    size = -1;
  }
}

void ocl_mem::copyTo(void* v){
  clEnqueueReadBuffer(device->getCommandQueue(), memory, CL_TRUE, 0, size, v,0,NULL,NULL);
}

void ocl_mem::copyTo(void* v, size_t offset, size_t s){
  clEnqueueReadBuffer(device->getCommandQueue(), memory, CL_TRUE, offset, s, v,0,NULL,NULL);
}

void ocl_mem::copyToNB(void* v){
  clEnqueueReadBuffer(device->getCommandQueue(), memory, CL_FALSE, 0, size, v,0,NULL,NULL);
}

void ocl_mem::copyToNB(void* v, size_t offset, size_t s){
  clEnqueueReadBuffer(device->getCommandQueue(), memory, CL_FALSE, offset, s, v,0,NULL,NULL);
}

void ocl_mem::copyFrom(void* v){
  clEnqueueWriteBuffer(device->getCommandQueue(), memory, CL_TRUE, 0, size, v, 0, NULL, NULL);
}

void ocl_mem::copyFrom(void* v, size_t offset, size_t s){
  clEnqueueWriteBuffer(device->getCommandQueue(), memory, CL_TRUE, offset, s, v,0,NULL,NULL);
}

void ocl_mem::copyFromNB(void* v){
  clEnqueueWriteBuffer(device->getCommandQueue(), memory, CL_FALSE, 0, size, v, 0, NULL, NULL);
}

void ocl_mem::copyFromNB(void* v, size_t offset, size_t s){
  clEnqueueWriteBuffer(device->getCommandQueue(), memory, CL_FALSE, offset, s, v,0,NULL,NULL);
}

cl_mem* ocl_mem::mem(){
  return &memory;
}

size_t ocl_mem::getSize(){
  return size;
}

// Opencl Variable Sizes
namespace ocl{
  ocl_setup setup;

  ocl_device displayDevices(){
    return ocl::setup.displayDevices();
  };

  ocl_device getDevice(int p, int d){
    return ocl::setup.getDevice(p,d);
  };

  cl_platform_id getPlatformID(int p){
    return ocl::setup.getPlatformID(p);
  };

  cl_device_id getDeviceID(int p, int d){
    return ocl::setup.getDeviceID(p,d);
  };

  const int types = 62;

  const std::string type[] = {
    "char"   , "char16", "char2"   , "char3"   , "char4"  , "char8"  ,
    "cl_mem" , "double", "double16", "double2" , "double3", "double4",
    "double8", "float" , "float16" , "float2"  , "float3" , "float4" ,
    "float8" , "half"  , "int"     , "int16"   , "int2"   , "int3"   ,
    "int4"   , "int8"  , "long"    , "long16"  , "long2"  , "long3"  ,
    "long4"  , "long8" , "short"   , "short16" , "short2" , "short3" ,
    "short4" , "short8", "uchar"   , "uchar16" , "uchar2" , "uchar3" ,
    "uchar4" , "uchar8", "uint"    , "uint16"  , "uint2"  , "uint3"  ,
    "uint4"  , "uint8" , "ulong"   , "ulong16" , "ulong2" , "ulong3" ,
    "ulong4" , "ulong8", "ushort"  , "ushort16", "ushort2", "ushort3",
    "ushort4", "ushort8"
  };

  const int typeSize[] = {
    1 ,    16,    2  ,    4  ,    4 ,    8 ,
    sizeof(cl_mem) ,    8 ,    128,    16 ,    32,    32,
    64,    4 ,    64 ,    8  ,    16,    16,
    32,    2 ,    4  ,    64 ,    8 ,    16,
    16,    32,    8  ,    128,    16,    32,
    32,    64,    2  ,    32 ,    4 ,    8 ,
    8 ,    16,    1  ,    16 ,    2 ,    4 ,
    4 ,    8 ,    4  ,    64 ,    8 ,    16,
    16,    32,    8  ,    128,    16,    32,
    32,    64,    2  ,    32 ,    4 ,    8 ,
    8 ,    16
  };

  const char error[65][45] = {
    "CL_SUCCESS"                        ,"CL_DEVICE_NOT_FOUND"             ,"CL_DEVICE_NOT_AVAILABLE",
    "CL_COMPILER_NOT_AVAILABLE"         ,"CL_MEM_OBJECT_ALLOCATION_FAILURE","CL_OUT_OF_RESOURCES",
    "CL_OUT_OF_HOST_MEMORY"             ,"CL_PROFILING_INFO_NOT_AVAILABLE" ,"CL_MEM_COPY_OVERLAP",
    "CL_IMAGE_FORMAT_MISMATCH"          ,"CL_IMAGE_FORMAT_NOT_SUPPORTED"   ,"CL_BUILD_PROGRAM_FAILURE",
    "CL_MAP_FAILURE"                    ,"CL_MISALIGNED_SUB_BUFFER_OFFSET" ,"CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
    "N/A"                               ,"N/A"                             ,"N/A",
    "N/A"                               ,"N/A"                             ,"N/A",
    "N/A"                               ,"N/A"                             ,"N/A",
    "N/A"                               ,"N/A"                             ,"N/A",
    "N/A"                               ,"N/A"                             ,"N/A",
    "CL_INVALID_VALUE"                  ,"CL_INVALID_DEVICE_TYPE"          ,"CL_INVALID_PLATFORM",
    "CL_INVALID_DEVICE"                 ,"CL_INVALID_CONTEXT"              ,"CL_INVALID_QUEUE_PROPERTIES",
    "CL_INVALID_COMMAND_QUEUE"          ,"CL_INVALID_HOST_PTR"             ,"CL_INVALID_MEM_OBJECT",
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR","CL_INVALID_IMAGE_SIZE"           ,"CL_INVALID_SAMPLER",
    "CL_INVALID_BINARY"                 ,"CL_INVALID_BUILD_OPTIONS"        ,"CL_INVALID_PROGRAM",
    "CL_INVALID_PROGRAM_EXECUTABLE"     ,"CL_INVALID_KERNEL_NAME"          ,"CL_INVALID_KERNEL_DEFINITION",
    "CL_INVALID_KERNEL"                 ,"CL_INVALID_ARG_INDEX"            ,"CL_INVALID_ARG_VALUE",
    "CL_INVALID_ARG_SIZE"               ,"CL_INVALID_KERNEL_ARGS"          ,"CL_INVALID_WORK_DIMENSION",
    "CL_INVALID_WORK_GROUP_SIZE"        ,"CL_INVALID_WORK_ITEM_SIZE"       ,"CL_INVALID_GLOBAL_OFFSET",
    "CL_INVALID_EVENT_WAIT_LIST"        ,"CL_INVALID_EVENT"                ,"CL_INVALID_OPERATION",
    "CL_INVALID_GL_OBJECT"              ,"CL_INVALID_BUFFER_SIZE"          ,"CL_INVALID_MIP_LEVEL",
    "CL_INVALID_GLOBAL_WORK_SIZE"       ,"CL_INVALID_PROPERTY"
  };

  std::string getFormattedKernel(ocl_kernel& k){
    return getFormattedKernel(k.getFunction());
  };

  std::string getFormattedKernel(std::string s){
    std::vector<std::string> words;
    std::vector<int> wType;
    std::stringstream ret;
    int pos = 0;

    parseKernel(s,words,wType);

    checkParsedKernel(pos,"",ret,words,wType);

    return ret.str();
  }

  void checkParsedKernel(int& pos, std::string indent, std::stringstream& ret, std::vector<std::string>& words, std::vector<int>& wType){
    int wordsSize = words.size();
    std::string tmp;
    int space = 2;
    char ctmp;
    int fc;

    while(pos < wordsSize){
      tmp = words[pos++];

      if(wType[pos-1] < 0){
	if(!tmp.compare("for")){
	  fc = 0;
	  ret << indent << "for";


	  while(pos < wordsSize && words[pos].compare("("))
	    ret << ' ' << words[pos++];

	  ret << "(";
	  space = 0;

	  while(++pos < wordsSize && fc < 3){
	    if(!words[pos].compare(";")){
	      ret << words[pos];
	      fc++;
	      space = 0;
	    }
	    else if(!words[pos].compare("(")){
	      ret << words[pos];
	      if(pos < wordsSize)
		fc--;
	      space = 0;
	    }
	    else if(!words[pos].compare(")")){
	      ret << words[pos];
	      fc++;
	    }
	    else if(wType[pos] < 0)
	      parseKernelSpaceCheck(space,ret," ",words[pos]);
	    else{
	      if(parseKernelOperatorCheck(pos,space,ret,indent,words,wType))
		return;
	    }
	  }
	}
	else if(!tmp.compare("else")){
	  ret << indent << "else";
	  if(pos < wordsSize && (!words[pos].compare("if") || !words[pos].compare("{")))
	    space = 0;
	  else{
	    ret << "\n   ";
	    space = 2;
	  }
	}
	else if(!tmp.compare("elseif")){
	  ret << indent << "else if";
	  if(pos < wordsSize && !words[pos].compare("{"))
	    space = 0;
	  else
	    space = 2;
	}
	else if(!tmp.compare("if")){
	  if(pos > 2 && words[pos-2].compare("else"))
	    ret << indent << "if";
	  else
	    ret << " if";
	  space = 0;
	  fc = 0;

	  while(pos < wordsSize && words[pos].compare("("))
	    ret << ' ' << words[pos++];
	  pos++;

	  ret << "(";

	  while(pos < wordsSize && fc < 1){
	    if(!words[pos].compare("(")){
	      ret << words[pos++];
	      fc--;
	      space = 0;
	    }
	    else if(!words[pos].compare(")")){
	      ret << words[pos++];
	      fc++;
	    }
	    else if(wType[pos] < 0)
	      parseKernelSpaceCheck(space,ret," ",words[pos++]);
	    else{
	      if(parseKernelOperatorCheck(pos,space,ret,indent,words,wType))
		return;
	      pos++;
	    }
	  }

	  if(words[pos].compare("{")){
	    ret << "\n   ";
	    space = 2;
	  }

	  if(wType[pos] > 0){
	    if(parseKernelOperatorCheck(pos,space,ret,indent,words,wType))
	      return;
	    pos++;
	  }
	}
	else
	  parseKernelSpaceCheck(space,ret,indent,tmp);
      }
      else{
	pos--;
	if(parseKernelOperatorCheck(pos,space,ret,indent,words,wType))
	  return;
	pos++;
      }
    }
  }

  int parseKernelOperatorCheck(int& pos, int& space, std::stringstream& ret, std::string indent, std::vector<std::string>& words, std::vector<int>& wType){
    char ch;
    std::string word = words[pos];

    if(word.length() > 1){
      ch = word[0];
      if(!word.compare("--") || !word.compare("++") || !word.compare("->")){
	ret << word;
	space = 0;
      }
      else if(!word.compare("//") || !word.compare("/*")){
	ret << '\n' << word << words[++pos];
	space = 2;
      }
      else
	parseKernelSpaceCheck(space,ret,indent,word);
    }
    else{
      ch = word[0];
      if(!(ch - ';')){
	ret << ";\n";
	space = 2;
      }
      else if(!(ch - '}')){
	for(int i=0;i<(indent.length()-3);i++)
	  ret << ' ';
	ret << "}\n";
	return 1;
      }
      else if(!(ch - '{')){
	ret << "{\n";
	checkParsedKernel(++pos,indent+"   ",ret,words,wType);
	space = 2;
      }
      else if(!(ch - '(') || !(ch - '[') || !(ch - ',') || !(ch - '!') || !(ch - '.')){
	if(pos > 1 && wType[pos-1] > 0)
	  ret << ' ' << ch;
	else
	  ret << ch;
	space = 0;
      }
      else if(!(ch - ')') || !(ch - ']')){
	ret << ch;
	space = 1;
      }
      else if(!(ch - '#')){
	ret << "\n#" << words[++pos];
	pos++;
      }
      else
	parseKernelSpaceCheck(space,ret,indent,word);
    }

    return 0;
  }

  void parseKernelSpaceCheck(int& space, std::stringstream& ret, std::string indent, std::string& word){
    if(space == 2){
      ret << indent << word;
      space = 1;
    }
    else if(space)
      ret << ' ' << word;
    else{
      ret << word;
      space = 1;
    }
  }

  void parseKernel(std::string s, std::vector<std::string>& words, std::vector<int>& wType){
    const std::string delim1 = "({[)}],;~*%?:^&|-+/!=<>#.";
    const std::string delim2 = "*/^^*=&&/=||--++==!=//+=<<-=>>/*<=>=->";

    int length = s.length();
    std::stringstream ret;
    int rSize = 0;
    int found = 0;
    int found1,found2;
    int offset = 0;
    int writing = 1;

    for(int i=0;i<length;i++){
      if(isspace(s[i])){
	if(writing){
	  words.push_back(s.substr(offset,i-offset));
	  wType.push_back(-1);
	  writing = 0;
	}

	i++;
	while(i < length && isspace(s[i]))
	  i++;
      }

      if(i < length){
	found1 = delim1.find(s[i]);
	if(found1 && !(s[i]-'#')){
	  offset = i+1;
	  while(++i<length && (s[i] - '\n')){}
	  words.push_back("#");
	  words.push_back(s.substr(offset,i-offset+1));
	  wType.push_back(1);
	  wType.push_back(-1);

	  writing = 0;
	}
      }
      while(i < length && (found1 != std::string::npos)){
	found = 1;
	if(writing){
	  words.push_back(s.substr(offset,i-offset));
	  wType.push_back(-1);
	  writing = 0;
	}

	if(i < length-1){
	  found2 = delim2.find(s.substr(i,2));

	  if(found2 != std::string::npos){
	    words.push_back(s.substr(i,2));
	    if(!s.substr(i,2).compare("/*")){
	      i++;
	      offset = i+1;
	      while(++i<length && (s[i] - '*') && ++i<length && (s[i] - '/')){}
	      words.push_back(s.substr(offset,i-offset+2));
	      wType.push_back(found2);
	      wType.push_back(-1);
	    }
	    if(!s.substr(i,2).compare("//")){
	      i++;
	      offset = i+1;
	      while(++i<length && (s[i] - '\n')){}
	      words.push_back(s.substr(offset,i-offset+1));
	      wType.push_back(found2);
	      wType.push_back(-1);
	    }
	    else{
	      wType.push_back(found2);
	      i++;
	    }
	  }
	  else{
	    words.push_back(s.substr(i,1));
	    wType.push_back(found1);
	  }

	  i++;
	  if(i < length)
	    found1 = delim1.find(s.substr(i,1));
	}
	else{
	  words.push_back(s.substr(i,1));
	  wType.push_back(found1);

	  i++;
	  found1 = delim1.find(s.substr(i,1));
	}
      }

      if(found){
	found = 0;
	i--;
      }
      else if(i < length && !writing){
	writing = 1;
	offset = i;
      }
    }
  };

  void printError(std::string s, int error){
    if(error){
      error = error < 0  ? error : -error;
      error = error < 65 ? error : 15;
      std::cout << s << " Error: " << error << '\n'
		<< "OpenCL Error: " << ocl::error[-error] << ".\n"
		<< "Exiting Program.\n";
      exit(1);
    }
  };

  void printSizes(){
    std::cout << "sizeof(char) = " << sizeof(cl_char) << std::endl
	      << "sizeof(char16) = " << sizeof(cl_char16) << std::endl
	      << "sizeof(char2) = " << sizeof(cl_char2) << std::endl
	      << "sizeof(char3) = " << sizeof(cl_char3) << std::endl
	      << "sizeof(char4) = " << sizeof(cl_char4) << std::endl
	      << "sizeof(char8) = " << sizeof(cl_char8) << std::endl
	      << "sizeof(cl_mem) = " << sizeof(cl_mem) << std::endl
	      << "sizeof(double) = " << sizeof(cl_double) << std::endl
	      << "sizeof(double16) = " << sizeof(cl_double16) << std::endl
	      << "sizeof(double2) = " << sizeof(cl_double2) << std::endl
	      << "sizeof(double3) = " << sizeof(cl_double3) << std::endl
	      << "sizeof(double4) = " << sizeof(cl_double4) << std::endl
	      << "sizeof(double8) = " << sizeof(cl_double8) << std::endl
	      << "sizeof(float) = " << sizeof(cl_float) << std::endl
	      << "sizeof(float16) = " << sizeof(cl_float16) << std::endl
	      << "sizeof(float2) = " << sizeof(cl_float2) << std::endl
	      << "sizeof(float3) = " << sizeof(cl_float3) << std::endl
	      << "sizeof(float4) = " << sizeof(cl_float4) << std::endl
	      << "sizeof(float8) = " << sizeof(cl_float8) << std::endl
	      << "sizeof(half) = " << sizeof(cl_half) << std::endl
	      << "sizeof(int) = " << sizeof(cl_int) << std::endl
	      << "sizeof(int16) = " << sizeof(cl_int16) << std::endl
	      << "sizeof(int2) = " << sizeof(cl_int2) << std::endl
	      << "sizeof(int3) = " << sizeof(cl_int3) << std::endl
	      << "sizeof(int4) = " << sizeof(cl_int4) << std::endl
	      << "sizeof(int8) = " << sizeof(cl_int8) << std::endl
	      << "sizeof(long) = " << sizeof(cl_long) << std::endl
	      << "sizeof(long16) = " << sizeof(cl_long16) << std::endl
	      << "sizeof(long2) = " << sizeof(cl_long2) << std::endl
	      << "sizeof(long3) = " << sizeof(cl_long3) << std::endl
	      << "sizeof(long4) = " << sizeof(cl_long4) << std::endl
	      << "sizeof(long8) = " << sizeof(cl_long8) << std::endl
	      << "sizeof(short) = " << sizeof(cl_short) << std::endl
	      << "sizeof(short16) = " << sizeof(cl_short16) << std::endl
	      << "sizeof(short2) = " << sizeof(cl_short2) << std::endl
	      << "sizeof(short3) = " << sizeof(cl_short3) << std::endl
	      << "sizeof(short4) = " << sizeof(cl_short4) << std::endl
	      << "sizeof(short8) = " << sizeof(cl_short8) << std::endl
	      << "sizeof(uchar) = " << sizeof(cl_uchar) << std::endl
	      << "sizeof(uchar16) = " << sizeof(cl_uchar16) << std::endl
	      << "sizeof(uchar2) = " << sizeof(cl_uchar2) << std::endl
	      << "sizeof(uchar3) = " << sizeof(cl_uchar3) << std::endl
	      << "sizeof(uchar4) = " << sizeof(cl_uchar4) << std::endl
	      << "sizeof(uchar8) = " << sizeof(cl_uchar8) << std::endl
	      << "sizeof(uint) = " << sizeof(cl_uint) << std::endl
	      << "sizeof(uint16) = " << sizeof(cl_uint16) << std::endl
	      << "sizeof(uint2) = " << sizeof(cl_uint2) << std::endl
	      << "sizeof(uint3) = " << sizeof(cl_uint3) << std::endl
	      << "sizeof(uint4) = " << sizeof(cl_uint4) << std::endl
	      << "sizeof(uint8) = " << sizeof(cl_uint8) << std::endl
	      << "sizeof(ulong) = " << sizeof(cl_ulong) << std::endl
	      << "sizeof(ulong16) = " << sizeof(cl_ulong16) << std::endl
	      << "sizeof(ulong2) = " << sizeof(cl_ulong2) << std::endl
	      << "sizeof(ulong3) = " << sizeof(cl_ulong3) << std::endl
	      << "sizeof(ulong4) = " << sizeof(cl_ulong4) << std::endl
	      << "sizeof(ulong8) = " << sizeof(cl_ulong8) << std::endl
	      << "sizeof(ushort) = " << sizeof(cl_ushort) << std::endl
	      << "sizeof(ushort16) = " << sizeof(cl_ushort16) << std::endl
	      << "sizeof(ushort2) = " << sizeof(cl_ushort2) << std::endl
	      << "sizeof(ushort3) = " << sizeof(cl_ushort3) << std::endl
	      << "sizeof(ushort4) = " << sizeof(cl_ushort4) << std::endl
	      << "sizeof(ushort8) = " << sizeof(cl_ushort8) << std::endl;
  };
};

#endif
