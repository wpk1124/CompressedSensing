#ifndef VIENNACL_LINALG_KERNELS_VECTOR_SOURCE_HPP_
#define VIENNACL_LINALG_KERNELS_VECTOR_SOURCE_HPP_
//Automatically generated file from auxiliary-directory, do not edit manually!
/** @file vector_source.h
 *  @brief OpenCL kernel source file, generated automatically from scripts in auxiliary/. */
namespace viennacl
{
 namespace linalg
 {
  namespace kernels
  {
const char * const vector_align1_inner_prod = 
"__kernel void inner_prod(\n"
"          __global const float * vec1,\n"
"          unsigned int start1,\n"
"          unsigned int inc1,\n"
"          unsigned int size1,\n"
"          __global const float * vec2,\n"
"          unsigned int start2,\n"
"          unsigned int inc2,\n"
"          unsigned int size2,\n"
"          __local float * tmp_buffer,\n"
"          __global float * group_buffer)\n"
"{\n"
"  unsigned int entries_per_group = get_local_size(0) * (size1-1) / get_global_size(0) + 1;\n"
"  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;\n"
"  unsigned int group_start1 = get_group_id(0) * entries_per_group * inc1 + start1;\n"
"  unsigned int group_start2 = get_group_id(0) * entries_per_group * inc2 + start2;\n"
"  \n"
"  unsigned int group_size = entries_per_group;\n"
"  if (get_group_id(0) * entries_per_group > size1)\n"
"    group_size = 0;\n"
"  else if ((get_group_id(0) + 1) * entries_per_group > size1)\n"
"    group_size = size1 - get_group_id(0) * entries_per_group;\n"
"  // compute partial results within group:\n"
"  float tmp = 0;\n"
"  for (unsigned int i = get_local_id(0); i < group_size; i += get_local_size(0))\n"
"    tmp += vec1[i*inc1 + group_start1] * vec2[i*inc2 + group_start2];\n"
"  tmp_buffer[get_local_id(0)] = tmp;\n"
"  // now run reduction:\n"
"  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2)\n"
"  {\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (get_local_id(0) < stride)\n"
"      tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0)+stride];\n"
"  }\n"
"  \n"
"  if (get_local_id(0) == 0)\n"
"    group_buffer[get_group_id(0)] = tmp_buffer[get_local_id(0)];\n"
"}\n"
; //vector_align1_inner_prod

const char * const vector_align1_avbv_cpu_gpu = 
"// generic kernel for the vector operation v1 = alpha * v2 + beta * v3, where v1, v2, v3 are not necessarily distinct vectors\n"
"__kernel void avbv_cpu_gpu(\n"
"          __global float * vec1,\n"
"          uint4 size1,\n"
"          \n"
"          float fac2,\n"
"          unsigned int options2,\n"
"          __global const float * vec2,\n"
"          uint4 size2,\n"
"          \n"
"          __global const float * fac3,\n"
"          unsigned int options3,\n"
"          __global const float * vec3,\n"
"          uint4 size3\n"
"          )\n"
"{ \n"
"  float alpha = fac2;\n"
"  if (options2 & (1 << 0))\n"
"    alpha = -alpha;\n"
"  if (options2 & (1 << 1))\n"
"    alpha = ((float)(1)) / alpha;\n"
"  float beta = fac3[0];\n"
"  if ((options3 >> 2) > 1)\n"
"  {\n"
"    for (unsigned int i=1; i<(options3 >> 2); ++i)\n"
"      beta += fac3[i];\n"
"  }\n"
"  if (options3 & (1 << 0))\n"
"    beta = -beta;\n"
"  if (options3 & (1 << 1))\n"
"    beta = ((float)(1)) / beta;\n"
"  \n"
"  for (unsigned int i = get_global_id(0); i < size1.z; i += get_global_size(0))\n"
"    vec1[i*size1.y+size1.x] = vec2[i*size2.y+size2.x] * alpha + vec3[i*size3.y+size3.x] * beta;\n"
"}\n"
; //vector_align1_avbv_cpu_gpu

const char * const vector_align1_plane_rotation = 
"////// plane rotation: (x,y) <- (\alpha x + \beta y, -\beta x + \alpha y)\n"
"__kernel void plane_rotation(\n"
"          __global float * vec1,\n"
"          unsigned int start1,\n"
"          unsigned int inc1,\n"
"          unsigned int size1,\n"
"          __global float * vec2, \n"
"          unsigned int start2,\n"
"          unsigned int inc2,\n"
"          unsigned int size2,\n"
"          float alpha,\n"
"          float beta) \n"
"{ \n"
"  float tmp1 = 0;\n"
"  float tmp2 = 0;\n"
"  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0))\n"
"  {\n"
"    tmp1 = vec1[i*inc1+start1];\n"
"    tmp2 = vec2[i*inc2+start2];\n"
"    \n"
"    vec1[i*inc1+start1] = alpha * tmp1 + beta * tmp2;\n"
"    vec2[i*inc2+start2] = alpha * tmp2 - beta * tmp1;\n"
"  }\n"
"}\n"
; //vector_align1_plane_rotation

const char * const vector_align1_element_op = 
"// generic kernel for the vector operation v1 = alpha * v2 + beta * v3, where v1, v2, v3 are not necessarily distinct vectors\n"
"__kernel void element_op(\n"
"          __global float * vec1,\n"
"          unsigned int start1,\n"
"          unsigned int inc1,          \n"
"          unsigned int size1,\n"
"          \n"
"          __global const float * vec2,\n"
"          unsigned int start2,\n"
"          unsigned int inc2,\n"
"          \n"
"          __global const float * vec3,\n"
"          unsigned int start3,\n"
"          unsigned int inc3,\n"
"          \n"
"          unsigned int is_division\n"
"          )\n"
"{ \n"
"  if (is_division)\n"
"  {\n"
"    for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0))\n"
"      vec1[i*inc1+start1] = vec2[i*inc2+start2] / vec3[i*inc3+start3];\n"
"  }\n"
"  else\n"
"  {\n"
"    for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0))\n"
"      vec1[i*inc1+start1] = vec2[i*inc2+start2] * vec3[i*inc3+start3];\n"
"  }\n"
"}\n"
; //vector_align1_element_op

const char * const vector_align1_av_gpu = 
"// generic kernel for the vector operation v1 = alpha * v2, where v1, v2 are not necessarily distinct vectors\n"
"__kernel void av_gpu(\n"
"          __global float * vec1,\n"
"          uint4 size1,\n"
"          \n"
"          __global const float * fac2,\n"
"          unsigned int options2,  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse\n"
"          __global const float * vec2,\n"
"          uint4 size2)\n"
"{ \n"
"  float alpha = fac2[0];\n"
"  if ((options2 >> 2) > 1)\n"
"  {\n"
"    for (unsigned int i=1; i<(options2 >> 2); ++i)\n"
"      alpha += fac2[i];\n"
"  }\n"
"  if (options2 & (1 << 0))\n"
"    alpha = -alpha;\n"
"  if (options2 & (1 << 1))\n"
"    alpha = ((float)(1)) / alpha;\n"
"  for (unsigned int i = get_global_id(0); i < size1.z; i += get_global_size(0))\n"
"    vec1[i*size1.y+size1.x] = vec2[i*size2.y+size2.x] * alpha;\n"
"}\n"
; //vector_align1_av_gpu

const char * const vector_align1_avbv_gpu_gpu = 
"// generic kernel for the vector operation v1 = alpha * v2 + beta * v3, where v1, v2, v3 are not necessarily distinct vectors\n"
"__kernel void avbv_gpu_gpu(\n"
"          __global float * vec1,\n"
"          uint4 size1,\n"
"          \n"
"          __global const float * fac2,\n"
"          unsigned int options2,  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse\n"
"          __global const float * vec2,\n"
"          uint4 size2,\n"
"          \n"
"          __global const float * fac3,\n"
"          unsigned int options3,  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse\n"
"          __global const float * vec3,\n"
"          uint4 size3\n"
"          )\n"
"{ \n"
"  float alpha = fac2[0];\n"
"  if ((options2 >> 2) > 1)\n"
"  {\n"
"    for (unsigned int i=1; i<(options2 >> 2); ++i)\n"
"      alpha += fac2[i];\n"
"  }\n"
"  if (options2 & (1 << 0))\n"
"    alpha = -alpha;\n"
"  if (options2 & (1 << 1))\n"
"    alpha = ((float)(1)) / alpha;\n"
"  float beta = fac3[0];\n"
"  if ((options3 >> 2) > 1)\n"
"  {\n"
"    for (unsigned int i=1; i<(options3 >> 2); ++i)\n"
"      beta += fac3[i];\n"
"  }\n"
"  if (options3 & (1 << 0))\n"
"    beta = -beta;\n"
"  if (options3 & (1 << 1))\n"
"    beta = ((float)(1)) / beta;\n"
"  \n"
"  for (unsigned int i = get_global_id(0); i < size1.z; i += get_global_size(0))\n"
"    vec1[i*size1.y+size1.x] = vec2[i*size2.y+size2.x] * alpha + vec3[i*size3.y+size3.x] * beta;\n"
"}\n"
; //vector_align1_avbv_gpu_gpu

const char * const vector_align1_swap = 
"////// swap:\n"
"__kernel void swap(\n"
"          __global float * vec1,\n"
"          unsigned int start1,\n"
"          unsigned int inc1,\n"
"          unsigned int size1,\n"
"          __global float * vec2,\n"
"          unsigned int start2,\n"
"          unsigned int inc2,\n"
"          unsigned int size2\n"
"          ) \n"
"{ \n"
"  float tmp;\n"
"  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0))\n"
"  {\n"
"    tmp = vec2[i*inc2+start2];\n"
"    vec2[i*inc2+start2] = vec1[i*inc1+start1];\n"
"    vec1[i*inc1+start1] = tmp;\n"
"  }\n"
"}\n"
" \n"
; //vector_align1_swap

const char * const vector_align1_avbv_v_gpu_gpu = 
"// generic kernel for the vector operation v1 += alpha * v2 + beta * v3, where v1, v2, v3 are not necessarily distinct vectors\n"
"__kernel void avbv_v_gpu_gpu(\n"
"          __global float * vec1,\n"
"          uint4 size1,\n"
"          \n"
"          __global const float * fac2,\n"
"          unsigned int options2,  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse\n"
"          __global const float * vec2,\n"
"          uint4 size2,\n"
"          \n"
"          __global const float * fac3,\n"
"          unsigned int options3,  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse\n"
"          __global const float * vec3,\n"
"          uint4 size3)\n"
"{ \n"
"  float alpha = fac2[0];\n"
"  if ((options2 >> 2) > 1)\n"
"  {\n"
"    for (unsigned int i=1; i<(options2 >> 2); ++i)\n"
"      alpha += fac2[i];\n"
"  }\n"
"  if (options2 & (1 << 0))\n"
"    alpha = -alpha;\n"
"  if (options2 & (1 << 1))\n"
"    alpha = ((float)(1)) / alpha;\n"
"  float beta = fac3[0];\n"
"  if ((options3 >> 2) > 1)\n"
"  {\n"
"    for (unsigned int i=1; i<(options3 >> 2); ++i)\n"
"      beta += fac3[i];\n"
"  }\n"
"  if (options3 & (1 << 0))\n"
"    beta = -beta;\n"
"  if (options3 & (1 << 1))\n"
"    beta = ((float)(1)) / beta;\n"
"  \n"
"  for (unsigned int i = get_global_id(0); i < size1.z; i += get_global_size(0))\n"
"    vec1[i*size1.y+size1.x] += vec2[i*size2.y+size2.x] * alpha + vec3[i*size3.y+size3.x] * beta;\n"
"}\n"
; //vector_align1_avbv_v_gpu_gpu

const char * const vector_align1_norm = 
"//helper:\n"
"void helper_norm_parallel_reduction( __local float * tmp_buffer )\n"
"{\n"
"  for (unsigned int stride = get_local_id(0)/2; stride > 0; stride /= 2)\n"
"  {\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (get_local_id(0) < stride)\n"
"      tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0)+stride];\n"
"  }\n"
"}\n"
"\n"
"float impl_norm(\n"
"          __global const float * vec,\n"
"          unsigned int start1,\n"
"          unsigned int inc1,\n"
"          unsigned int size1,\n"
"          unsigned int norm_selector,\n"
"          __local float * tmp_buffer)\n"
"{\n"
"  float tmp = 0;\n"
"  if (norm_selector == 1) //norm_1\n"
"  {\n"
"    for (unsigned int i = get_local_id(0); i < size1; i += get_local_size(0))\n"
"      tmp += fabs(vec[i*inc1 + start1]);\n"
"  }\n"
"  else if (norm_selector == 2) //norm_2\n"
"  {\n"
"    float vec_entry = 0;\n"
"    for (unsigned int i = get_local_id(0); i < size1; i += get_local_size(0))\n"
"    {\n"
"      vec_entry = vec[i*inc1 + start1];\n"
"      tmp += vec_entry * vec_entry;\n"
"    }\n"
"  }\n"
"  else if (norm_selector == 0) //norm_inf\n"
"  {\n"
"    for (unsigned int i = get_local_id(0); i < size1; i += get_local_size(0))\n"
"      tmp = fmax(fabs(vec[i*inc1 + start1]), tmp);\n"
"  }\n"
"  \n"
"  tmp_buffer[get_local_id(0)] = tmp;\n"
"\n"
"  if (norm_selector > 0) //norm_1 or norm_2:\n"
"  {\n"
"    for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2)\n"
"    {\n"
"      barrier(CLK_LOCAL_MEM_FENCE);\n"
"      if (get_local_id(0) < stride)\n"
"        tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0)+stride];\n"
"    }\n"
"    return tmp_buffer[0];\n"
"  }\n"
"  \n"
"  //norm_inf:\n"
"  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2)\n"
"  {\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (get_local_id(0) < stride)\n"
"      tmp_buffer[get_local_id(0)] = fmax(tmp_buffer[get_local_id(0)], tmp_buffer[get_local_id(0)+stride]);\n"
"  }\n"
"  \n"
"  return tmp_buffer[0];\n"
"};\n"
"\n"
"__kernel void norm(\n"
"          __global const float * vec,\n"
"          unsigned int start1,\n"
"          unsigned int inc1,\n"
"          unsigned int size1,\n"
"          unsigned int norm_selector,\n"
"          __local float * tmp_buffer,\n"
"          __global float * group_buffer)\n"
"{\n"
"  float tmp = impl_norm(vec,\n"
"                        (        get_group_id(0)  * size1) / get_num_groups(0) * inc1 + start1,\n"
"                        inc1,\n"
"                        (   (1 + get_group_id(0)) * size1) / get_num_groups(0) \n"
"                      - (        get_group_id(0)  * size1) / get_num_groups(0),\n"
"                        norm_selector,\n"
"                        tmp_buffer);\n"
"  \n"
"  if (get_local_id(0) == 0)\n"
"    group_buffer[get_group_id(0)] = tmp;  \n"
"}\n"
"\n"
; //vector_align1_norm

const char * const vector_align1_avbv_cpu_cpu = 
"// generic kernel for the vector operation v1 = alpha * v2 + beta * v3, where v1, v2, v3 are not necessarily distinct vectors\n"
"__kernel void avbv_cpu_cpu(\n"
"          __global float * vec1,\n"
"          uint4 size1,\n"
"          \n"
"          float fac2,\n"
"          unsigned int options2,  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse\n"
"          __global const float * vec2,\n"
"          uint4 size2,\n"
"          \n"
"          float fac3,\n"
"          unsigned int options3,  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse\n"
"          __global const float * vec3,\n"
"          uint4 size3\n"
"          )\n"
"{ \n"
"  float alpha = fac2;\n"
"  if (options2 & (1 << 0))\n"
"    alpha = -alpha;\n"
"  if (options2 & (1 << 1))\n"
"    alpha = ((float)(1)) / alpha;\n"
"  float beta = fac3;\n"
"  if (options3 & (1 << 0))\n"
"    beta = -beta;\n"
"  if (options3 & (1 << 1))\n"
"    beta = ((float)(1)) / beta;\n"
"  for (unsigned int i = get_global_id(0); i < size1.z; i += get_global_size(0))\n"
"    vec1[i*size1.y+size1.x] = vec2[i*size2.y+size2.x] * alpha + vec3[i*size3.y+size3.x] * beta;\n"
"}\n"
; //vector_align1_avbv_cpu_cpu

const char * const vector_align1_assign_cpu = 
"__kernel void assign_cpu(\n"
"          __global float * vec1,\n"
"          unsigned int start1,\n"
"          unsigned int inc1,\n"
"          unsigned int size1,\n"
"          unsigned int internal_size1,\n"
"          float alpha) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < internal_size1; i += get_global_size(0))\n"
"    vec1[i*inc1+start1] = (i < size1) ? alpha : 0;\n"
"}\n"
; //vector_align1_assign_cpu

const char * const vector_align1_avbv_v_cpu_gpu = 
"// generic kernel for the vector operation v1 += alpha * v2 + beta * v3, where v1, v2, v3 are not necessarily distinct vectors\n"
"__kernel void avbv_v_cpu_gpu(\n"
"          __global float * vec1,\n"
"          uint4 size1,\n"
"          \n"
"          float fac2,\n"
"          unsigned int options2,\n"
"          __global const float * vec2,\n"
"          uint4 size2,\n"
"          \n"
"          __global const float * fac3,\n"
"          unsigned int options3,\n"
"          __global const float * vec3,\n"
"          uint4 size3)\n"
"{ \n"
"  float alpha = fac2;\n"
"  if (options2 & (1 << 0))\n"
"    alpha = -alpha;\n"
"  if (options2 & (1 << 1))\n"
"    alpha = ((float)(1)) / alpha;\n"
"  float beta = fac3[0];\n"
"  if ((options3 >> 2) > 1)\n"
"  {\n"
"    for (unsigned int i=1; i<(options3 >> 2); ++i)\n"
"      beta += fac3[i];\n"
"  }\n"
"  if (options3 & (1 << 0))\n"
"    beta = -beta;\n"
"  if (options3 & (1 << 1))\n"
"    beta = ((float)(1)) / beta;\n"
"  \n"
"  for (unsigned int i = get_global_id(0); i < size1.z; i += get_global_size(0))\n"
"    vec1[i*size1.y+size1.x] += vec2[i*size2.y+size2.x] * alpha + vec3[i*size3.y+size3.x] * beta;\n"
"}\n"
; //vector_align1_avbv_v_cpu_gpu

const char * const vector_align1_sum = 
"// sums the array 'vec1' and writes to result. Makes use of a single work-group only. \n"
"__kernel void sum(\n"
"          __global float * vec1,\n"
"          unsigned int start1,\n"
"          unsigned int inc1,\n"
"          unsigned int size1,\n"
"          unsigned int option, //0: use fmax, 1: just sum, 2: sum and return sqrt of sum\n"
"          __local float * tmp_buffer,\n"
"          __global float * result) \n"
"{ \n"
"  float thread_sum = 0;\n"
"  for (unsigned int i = get_local_id(0); i<size1; i += get_local_size(0))\n"
"  {\n"
"    if (option > 0)\n"
"      thread_sum += vec1[i*inc1+start1];\n"
"    else\n"
"      thread_sum = fmax(thread_sum, fabs(vec1[i*inc1+start1]));\n"
"  }\n"
"  \n"
"  tmp_buffer[get_local_id(0)] = thread_sum;\n"
"  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2)\n"
"  {\n"
"    if (get_local_id(0) < stride)\n"
"    {\n"
"      if (option > 0)\n"
"        tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0) + stride];\n"
"      else\n"
"        tmp_buffer[get_local_id(0)] = fmax(tmp_buffer[get_local_id(0)], tmp_buffer[get_local_id(0) + stride]);\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"  }\n"
"  \n"
"  if (get_global_id(0) == 0)\n"
"  {\n"
"    if (option == 2)\n"
"      *result = sqrt(tmp_buffer[0]);\n"
"    else\n"
"      *result = tmp_buffer[0];\n"
"  }\n"
"}\n"
; //vector_align1_sum

const char * const vector_align1_index_norm_inf = 
"//index_norm_inf:\n"
"unsigned int float_vector1_index_norm_inf_impl(\n"
"          __global const float * vec,\n"
"          unsigned int start1,\n"
"          unsigned int inc1,\n"
"          unsigned int size1,\n"
"          __local float * float_buffer,\n"
"          __local unsigned int * index_buffer)\n"
"{\n"
"  //step 1: fill buffer:\n"
"  float cur_max = 0.0f;\n"
"  float tmp;\n"
"  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0))\n"
"  {\n"
"    tmp = fabs(vec[i*inc1+start1]);\n"
"    if (cur_max < tmp)\n"
"    {\n"
"      float_buffer[get_global_id(0)] = tmp;\n"
"      index_buffer[get_global_id(0)] = i;\n"
"      cur_max = tmp;\n"
"    }\n"
"  }\n"
"  \n"
"  //step 2: parallel reduction:\n"
"  for (unsigned int stride = get_global_size(0)/2; stride > 0; stride /= 2)\n"
"  {\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (get_global_id(0) < stride)\n"
"    {\n"
"      //find the first occurring index\n"
"      if (float_buffer[get_global_id(0)] < float_buffer[get_global_id(0)+stride])\n"
"      {\n"
"        index_buffer[get_global_id(0)] = index_buffer[get_global_id(0)+stride];\n"
"        float_buffer[get_global_id(0)] = float_buffer[get_global_id(0)+stride];\n"
"      }\n"
"      \n"
"      //index_buffer[get_global_id(0)] = float_buffer[get_global_id(0)] < float_buffer[get_global_id(0)+stride] ? index_buffer[get_global_id(0)+stride] : index_buffer[get_global_id(0)];\n"
"      //float_buffer[get_global_id(0)] = max(float_buffer[get_global_id(0)], float_buffer[get_global_id(0)+stride]);\n"
"    }\n"
"  }\n"
"  \n"
"  return index_buffer[0];\n"
"}\n"
"\n"
"__kernel void index_norm_inf(\n"
"          __global float * vec,\n"
"          unsigned int start1,\n"
"          unsigned int inc1,\n"
"          unsigned int size1,\n"
"          __local float * float_buffer,\n"
"          __local unsigned int * index_buffer,\n"
"          __global unsigned int * result) \n"
"{ \n"
"  unsigned int tmp = float_vector1_index_norm_inf_impl(vec, start1, inc1, size1, float_buffer, index_buffer);\n"
"  if (get_global_id(0) == 0) *result = tmp;\n"
"}\n"
"\n"
"\n"
; //vector_align1_index_norm_inf

const char * const vector_align1_diag_precond = 
"__kernel void diag_precond(\n"
"          __global const float * diag_A_inv, \n"
"          unsigned int start1,\n"
"          unsigned int inc1,\n"
"          unsigned int size1,\n"
"          __global float * x, \n"
"          unsigned int start2,\n"
"          unsigned int inc2,\n"
"          unsigned int size2) \n"
"{ \n"
"  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0))\n"
"    x[i*inc2+start2] *= diag_A_inv[i*inc1+start1];\n"
"}\n"
; //vector_align1_diag_precond

const char * const vector_align1_avbv_gpu_cpu = 
"// generic kernel for the vector operation v1 = alpha * v2 + beta * v3, where v1, v2, v3 are not necessarily distinct vectors\n"
"__kernel void avbv_gpu_cpu(\n"
"          __global float * vec1,\n"
"          uint4 size1,\n"
"          \n"
"          __global const float * fac2,\n"
"          unsigned int options2,  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse\n"
"          __global const float * vec2,\n"
"          uint4 size2,\n"
"          \n"
"          float fac3,\n"
"          unsigned int options3,  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse\n"
"          __global const float * vec3,\n"
"          uint4 size3\n"
"          )\n"
"{ \n"
"  float alpha = fac2[0];\n"
"  if ((options2 >> 2) > 1)\n"
"  {\n"
"    for (unsigned int i=1; i<(options2 >> 2); ++i)\n"
"      alpha += fac2[i];\n"
"  }\n"
"  if (options2 & (1 << 0))\n"
"    alpha = -alpha;\n"
"  if (options2 & (1 << 1))\n"
"    alpha = ((float)(1)) / alpha;\n"
"  float beta = fac3;\n"
"  if (options3 & (1 << 0))\n"
"    beta = -beta;\n"
"  if (options3 & (1 << 1))\n"
"    beta = ((float)(1)) / beta;\n"
"    \n"
"  for (unsigned int i = get_global_id(0); i < size1.z; i += get_global_size(0))\n"
"    vec1[i*size1.y+size1.x] = vec2[i*size2.y+size2.x] * alpha + vec3[i*size3.y+size3.x] * beta;\n"
"}\n"
; //vector_align1_avbv_gpu_cpu

const char * const vector_align1_av_cpu = 
"// generic kernel for the vector operation v1 = alpha * v2, where v1, v2 are not necessarily distinct vectors\n"
"__kernel void av_cpu(\n"
"          __global float * vec1,\n"
"          uint4 size1,\n"
"          \n"
"          float fac2,\n"
"          unsigned int options2,\n"
"          __global const float * vec2,\n"
"          uint4 size2)\n"
"{ \n"
"  float alpha = fac2;\n"
"  if (options2 & (1 << 0))\n"
"    alpha = -alpha;\n"
"  if (options2 & (1 << 1))\n"
"    alpha = ((float)(1)) / alpha;\n"
"  for (unsigned int i = get_global_id(0); i < size1.z; i += get_global_size(0))\n"
"    vec1[i*size1.y+size1.x] = vec2[i*size2.y+size2.x] * alpha;\n"
"}\n"
; //vector_align1_av_cpu

const char * const vector_align1_avbv_v_cpu_cpu = 
"// generic kernel for the vector operation v1 += alpha * v2 + beta * v3, where v1, v2, v3 are not necessarily distinct vectors\n"
"__kernel void avbv_v_cpu_cpu(\n"
"          __global float * vec1,\n"
"          uint4 size1,\n"
"          \n"
"          float fac2,\n"
"          unsigned int options2,  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse\n"
"          __global const float * vec2,\n"
"          uint4 size2,\n"
"          \n"
"          float fac3,\n"
"          unsigned int options3,  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse\n"
"          __global const float * vec3,\n"
"          uint4 size3)\n"
"{ \n"
"  float alpha = fac2;\n"
"  if (options2 & (1 << 0))\n"
"    alpha = -alpha;\n"
"  if (options2 & (1 << 1))\n"
"    alpha = ((float)(1)) / alpha;\n"
"  float beta = fac3;\n"
"  if (options3 & (1 << 0))\n"
"    beta = -beta;\n"
"  if (options3 & (1 << 1))\n"
"    beta = ((float)(1)) / beta;\n"
"  for (unsigned int i = get_global_id(0); i < size1.z; i += get_global_size(0))\n"
"    vec1[i*size1.y+size1.x] += vec2[i*size2.y+size2.x] * alpha + vec3[i*size3.y+size3.x] * beta;\n"
"}\n"
; //vector_align1_avbv_v_cpu_cpu

const char * const vector_align1_avbv_v_gpu_cpu = 
"// generic kernel for the vector operation v1 += alpha * v2 + beta * v3, where v1, v2, v3 are not necessarily distinct vectors\n"
"__kernel void avbv_v_gpu_cpu(\n"
"          __global float * vec1,\n"
"          uint4 size1,\n"
"          \n"
"          __global const float * fac2,\n"
"          unsigned int options2,  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse\n"
"          __global const float * vec2,\n"
"          uint4 size2,\n"
"          \n"
"          float fac3,\n"
"          unsigned int options3,  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse\n"
"          __global const float * vec3,\n"
"          uint4 size3)\n"
"{ \n"
"  float alpha = fac2[0];\n"
"  if ((options2 >> 2) > 1)\n"
"  {\n"
"    for (unsigned int i=1; i<(options2 >> 2); ++i)\n"
"      alpha += fac2[i];\n"
"  }\n"
"  if (options2 & (1 << 0))\n"
"    alpha = -alpha;\n"
"  if (options2 & (1 << 1))\n"
"    alpha = ((float)(1)) / alpha;\n"
"  float beta = fac3;\n"
"  if (options3 & (1 << 0))\n"
"    beta = -beta;\n"
"  if (options3 & (1 << 1))\n"
"    beta = ((float)(1)) / beta;\n"
"    \n"
"  for (unsigned int i = get_global_id(0); i < size1.z; i += get_global_size(0))\n"
"    vec1[i*size1.y+size1.x] += vec2[i*size2.y+size2.x] * alpha + vec3[i*size3.y+size3.x] * beta;\n"
"}\n"
; //vector_align1_avbv_v_gpu_cpu

  }  //namespace kernels
 }  //namespace linalg
}  //namespace viennacl
#endif

