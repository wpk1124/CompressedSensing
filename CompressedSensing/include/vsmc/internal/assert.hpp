#ifndef VSMC_INTERNAL_ASSERT_HPP
#define VSMC_INTERNAL_ASSERT_HPP

#include <vsmc/internal/config.hpp>

#include <cassert>
#include <cstdio>
#include <stdexcept>

// Runtime assertion

#if VSMC_RUNTIME_ASSERT_AS_EXCEPTION
#define VSMC_RUNTIME_ASSERT(cond, msg)                                        \
{                                                                             \
    if (!(cond)) {                                                            \
        throw vsmc::RuntimeAssert(msg);                                       \
    };                                                                        \
}
#elif defined(NDEBUG) // No Debug
#define VSMC_RUNTIME_ASSERT(cond, msg)
#else // Runtime assertion
#define VSMC_RUNTIME_ASSERT(cond, msg)                                        \
{                                                                             \
    if (!(cond)) {                                                            \
        std::fprintf(stderr,                                                  \
                "vSMC runtime assertion failed; File: %s; Line: %d\n%s\n",    \
                __FILE__, __LINE__, msg);                                     \
    };                                                                        \
    assert(cond);                                                             \
}
#endif // VSMC_RUNTIME_ASSERT_AS_EXCEPTION

// Static assertion

#if VSMC_HAS_CXX11_STATIC_ASSERT
#define VSMC_STATIC_ASSERT(cond, msg) static_assert(cond, #msg)
#else // VSMC_HAS_CXX11_STATIC_ASSERT
#ifdef _MSC_VER
#define VSMC_STATIC_ASSERT(cond, msg) \
    {vsmc::StaticAssert<bool(cond)>::msg;}
#else // _MSC_VER
#define VSMC_STATIC_ASSERT(cond, msg) \
    if (vsmc::StaticAssert<bool(cond)>::msg) {};
#endif // _MSC_VER
#endif // VSMC_HAS_CXX11_STATIC_ASSERT

namespace vsmc {

class RuntimeAssert : public std::runtime_error
{
    public :

    RuntimeAssert (const std::string &msg) : std::runtime_error(msg) {}
}; // class RuntimeAssert

template <bool> class StaticAssert {};

template <>
class StaticAssert<true>
{
    public :

    enum {
        USE_METHOD_resize_dim_WITH_A_FIXED_DIM_State_OBJECT,

        USE_StateCL_WITH_A_STATE_TYPE_OTHER_THAN_cl_float_AND_cl_double,
        USE_InitializeCL_WITH_A_STATE_TYPE_NOT_DERIVED_FROM_StateCL,
        USE_MoveCL_WITH_A_STATE_TYPE_NOT_DERIVED_FROM_StateCL,
        USE_MonitorEvalCL_WITH_A_STATE_TYPE_NOT_DERIVED_FROM_StateCL,
        USE_PathEvalCL_WITH_A_STATE_TYPE_NOT_DERIVED_FROM_StateCL,

        USE_NumericNewtonCotes_WITH_A_DEGREE_LARGER_THAN_max_degree
    };
}; // class StaticAssert

} // namespace vsmc

// Runtime assertion macros

#define VSMC_RUNTIME_ASSERT_CL_MANAGER_SETUP(func)                            \
    VSMC_RUNTIME_ASSERT((setup()),                                            \
            ("**vsmc::CLManager::"#func"** CAN ONLY BE CALLED AFTER TRUE "    \
             "**vsmc::CLManager::setup**"));

#define VSMC_RUNTIME_ASSERT_CL_MANAGER_SETUP_PLATFORM                         \
    VSMC_RUNTIME_ASSERT(setup_platform,                                       \
            ("**vsmc::CLManager::setup** FAILED TO SETUP A PLATFORM"));

#define VSMC_RUNTIME_ASSERT_CL_MANAGER_SETUP_CONTEXT                          \
    VSMC_RUNTIME_ASSERT(setup_context,                                        \
            ("**vsmc::CLManager::setup** FAILED TO SETUP A CONTEXT"));

#define VSMC_RUNTIME_ASSERT_CL_MANAGER_SETUP_DEVICE                           \
    VSMC_RUNTIME_ASSERT(setup_device,                                         \
            ("**vsmc::CLManager::setup** FAILED TO SETUP A DEVICE"));

#define VSMC_RUNTIME_ASSERT_CL_MANAGER_SETUP_COMMAND_QUEUE                    \
    VSMC_RUNTIME_ASSERT(setup_command_queue,                                  \
            ("**vsmc::CLManager::setup** FAILED TO SETUP A COMMAND_QUEUE"));

#define VSMC_RUNTIME_ASSERT_DERIVED_BASE(basename)                            \
    VSMC_RUNTIME_ASSERT((dynamic_cast<Derived *>(this)),                      \
            ("DERIVED FROM " #basename                                        \
             " WITH INCORRECT **Derived** TEMPLATE PARAMTER"));

#define VSMC_RUNTIME_ASSERT_DIM(dim)                                          \
    VSMC_RUNTIME_ASSERT((dim >= 1), ("DIMENSION IS LESS THAN 1"))

#define VSMC_RUNTIME_ASSERT_FUNCTOR(func, name, caller)                       \
    VSMC_RUNTIME_ASSERT(bool(func), "**"#caller"** INVALID "#name" OBJECT")   \

#define VSMC_RUNTIME_ASSERT_ID_NUMBER(func)                                   \
    VSMC_RUNTIME_ASSERT((id >= 0 && id < this->dim()),                        \
            ("**"#func"** INVALID ITERATION NUMBER ARGUMENT"))

#define VSMC_RUNTIME_ASSERT_INVALID_MEMCPY_IN(diff, size, func)               \
    VSMC_RUNTIME_ASSERT((std::abs(diff) > static_cast<std::ptrdiff_t>(size)), \
            ("THE DESTINATION OF **"#func"** OVERLAPPING WITH THE SOURCE"))

#define VSMC_RUNTIME_ASSERT_INVALID_MEMCPY_OUT(diff, size, func)              \
    VSMC_RUNTIME_ASSERT((std::abs(diff) > static_cast<std::ptrdiff_t>(size)), \
            ("THE SOURCE OF **"#func"** OVERLAPPING WITH THE DESTINATION"))

#define VSMC_RUNTIME_ASSERT_ITERATION_NUMBER(func)                            \
    VSMC_RUNTIME_ASSERT((iter >= 0 && iter < this->iter_size()),              \
            ("**"#func"** INVALID ITERATION NUMBER ARGUMENT"))

#define VSMC_RUNTIME_ASSERT_MATRIX_ORDER(order, func)                         \
    VSMC_RUNTIME_ASSERT((order == vsmc::RowMajor || order == vsmc::ColMajor), \
            ("**"#func"** INVALID MATRIX ORDER"))

#define VSMC_RUNTIME_ASSERT_MONITOR_NAME(iter, map, func)                     \
    VSMC_RUNTIME_ASSERT((iter != map.end()),                                  \
            ("**"#func"** INVALID MONITOR NAME"))

#define VSMC_RUNTIME_ASSERT_PARTICLE_ITERATOR_BINARY_OP                       \
    VSMC_RUNTIME_ASSERT((iter1->particle_ptr() == iter2->particle_ptr()),     \
            ("BINARY OPERATION ON TWO **ParticleIterator** BELONGING TO "     \
            "TWO DIFFERNT **Particle** OBJECT"))

#define VSMC_RUNTIME_ASSERT_RANGE(begin, end, func)                           \
    VSMC_RUNTIME_ASSERT((begin < end), ("**"#func"** INVALID RANGE"))

#define VSMC_RUNTIME_ASSERT_STATE_CL_BUILD(func)                              \
    VSMC_RUNTIME_ASSERT((build()),                                            \
            ("**StateCL::"#func"** CAN ONLY BE CALLED AFTER true "            \
             "**StateCL::build**"));

#define VSMC_RUNTIME_ASSERT_STATE_COPY_SIZE_MISMATCH(name)                    \
    VSMC_RUNTIME_ASSERT((N == static_cast<size_type>(this->size())),          \
            ("**State"#name"::copy** SIZE MISMATCH"))

#define VSMC_RUNTIME_ASSERT_STATE_COPY_SIZE_MISMATCH_MPI                      \
    VSMC_RUNTIME_ASSERT((N == global_size_),                                  \
            ("**StateMPI::copy** SIZE MISMATCH"))

#define VSMC_RUNTIME_ASSERT_STATE_MATRIX_RC_ITERATOR_BINARY_OP                \
    VSMC_RUNTIME_ASSERT((iter1->inc() == iter2->inc()),                       \
            ("BINARY OPERATION ON TWO **StateMatrixRCIteraotr** WITH"         \
            "TWO DIFFERNT INCREMENT"))

#define VSMC_RUNTIME_ASSERT_STATE_UNPACK_SIZE(pack_size, dim, name)           \
    VSMC_RUNTIME_ASSERT((pack_size >= dim),                                   \
            ("**State"#name"::state_unpack** INPUT PACK SIZE TOO SMALL"))

// Static assertion macros

#define VSMC_STATIC_ASSERT_DYNAMIC_DIM_RESIZE(Dim)                            \
    VSMC_STATIC_ASSERT((Dim == vsmc::Dynamic),                                \
            USE_METHOD_resize_dim_WITH_A_FIXED_DIM_State_OBJECT)

#define VSMC_STATIC_ASSERT_NUMERIC_NEWTON_COTES_DEGREE(degree)                \
    VSMC_STATIC_ASSERT((degree >= 1 && degree <= max_degree_),                \
            USE_NumericNewtonCotes_WITH_A_DEGREE_LARGER_THAN_max_degree)

#define VSMC_STATIC_ASSERT_NO_IMPL(member)                                    \
    VSMC_STATIC_ASSERT((vsmc::cxx11::is_same<Derived, NullType>::value),      \
            NO_IMPLEMENTATION_OF_##member##_FOUND)

#define VSMC_STATIC_ASSERT_STATE_CL_TYPE(derived, user)                       \
    VSMC_STATIC_ASSERT((vsmc::traits::IsDerivedFromStateCL<derived>::value),  \
            USE_##user##_WITH_A_STATE_TYPE_NOT_DERIVED_FROM_StateCL)

#define VSMC_STATIC_ASSERT_STATE_CL_VALUE_TYPE(type)                          \
    VSMC_STATIC_ASSERT((vsmc::cxx11::is_same<type, cl_float>::value           \
                || vsmc::cxx11::is_same<type, cl_double>::value),             \
            USE_StateCL_WITH_A_STATE_TYPE_OTHER_THAN_cl_float_AND_cl_double)

#endif // VSMC_INTERNAL_ASSERT_HPP
