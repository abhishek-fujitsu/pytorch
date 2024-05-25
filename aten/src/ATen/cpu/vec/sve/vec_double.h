#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/sve/sve_helper.h>
#if defined(CPU_CAPABILITY_SVE)
#include <sleef.h>
#endif
#include <iostream>

namespace at {
namespace vec {
// Note [CPU_CAPABILITY namespace]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This header, and all of its subheaders, will be compiled with
// different architecture flags for each supported set of vector
// intrinsics. So we need to make sure they aren't inadvertently
// linked together. We do this by declaring objects in an `inline
// namespace` which changes the name mangling, but can still be
// accessed as `at::vec`.
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_SVE)

template <> class Vectorized<double> {
private:
  vls_float64_t values;
public:
  using value_type = double;
  using size_type = int;
  static constexpr size_type size() {
    return VECTOR_WIDTH / sizeof(double);
  }
  Vectorized() {}
  Vectorized(svfloat64_t v) : values(v) {}
  Vectorized(double val) {
    values = svdup_n_f64(val);
  }
  template<typename... Args,
           typename = std::enable_if_t<(sizeof...(Args) == size())>>
  Vectorized(Args... vals) {
    __at_align__ double buffer[size()] = { vals... };
    values = svld1_f64(ptrue, buffer);
  }
  operator svfloat64_t() const {
    return values;
  }
  static Vectorized<double> blendv(const Vectorized<double>& a, const Vectorized<double>& b,
                              const Vectorized<double>& mask_) {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::blendv()"<<std::endl;
    svbool_t mask = svcmpeq_s64(ptrue, svreinterpret_s64_f64(mask_),
                               ALL_S64_TRUE_MASK);
    return svsel_f64(mask, b, a);
  }
  template<typename step_t>
  static Vectorized<double> arange(double base = 0., step_t step = static_cast<step_t>(1)) {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::arange()"<<std::endl;
    __at_align__ double buffer[size()];
    for (int64_t i = 0; i < size(); i++) {
      buffer[i] = base + i * step;
    }
    return svld1_f64(ptrue, buffer);
  }
  static Vectorized<double> set(const Vectorized<double>& a, const Vectorized<double>& b,
                           int64_t count = size()) {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::set()"<<std::endl;
    if (count == 0) {
      return a;
    } else if (count < size()) {
      return svsel_f64(svwhilelt_b64(0ull, count), b, a);
    }
    return b;
  }
  static Vectorized<double> loadu(const void* ptr, int64_t count = size()) {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::loadu()"<<std::endl;
    if (count == size())
      return svld1_f64(ptrue, reinterpret_cast<const double*>(ptr));
    svbool_t pg = svwhilelt_b64(0ull, count);
    return svld1_f64(pg, reinterpret_cast<const double*>(ptr));
  }
  void store(void* ptr, int64_t count = size()) const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::store()"<<std::endl;
    if (count == size()) {
      svst1_f64(ptrue, reinterpret_cast<double*>(ptr), values);
    } else {
      svbool_t pg = svwhilelt_b64(0ull, count);
      svst1_f64(pg, reinterpret_cast<double*>(ptr), values);
    }
  }
  const double& operator[](int idx) const  = delete;
  double& operator[](int idx) = delete;
  int64_t zero_mask() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::zero_mask()"<<std::endl;
    // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
    int64_t mask = 0;
    __at_align__ int64_t mask_array[size()];

    svbool_t svbool_mask = svcmpeq_f64(ptrue, values, ZERO_F64);
    svst1_s64(ptrue, mask_array, svsel_s64(svbool_mask,
                                          ALL_S64_TRUE_MASK,
                                          ALL_S64_FALSE_MASK));
    for (int64_t i = 0; i < size(); ++i) {
      if (mask_array[i]) mask |= (1ull << i);
    }
    return mask;
  }
  Vectorized<double> isnan() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::isnan()"<<std::endl;
    // NaN check
    svbool_t mask = svcmpuo_f64(ptrue, values, ZERO_F64);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }
  bool has_inf_nan() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::has_inf_nan()"<<std::endl;
    return svptest_any(ptrue, svcmpuo_f64(ptrue, svsub_f64_x(ptrue, values, values), ZERO_F64));
  }
  Vectorized<double> map(double (*f)(double)) const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::map()"<<std::endl;
    __at_align__ double tmp[size()];
    store(tmp);
    for (int64_t i = 0; i < size(); ++i) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> abs() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::abs()"<<std::endl;
    return svabs_f64_x(ptrue, values);
  }
  Vectorized<double> angle() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::angle()"<<std::endl;
    const auto nan_vec = svdup_n_f64(NAN);
    const auto nan_mask = svcmpuo_f64(ptrue, values, ZERO_F64);
    const auto pi = svdup_n_f64(c10::pi<double>);

    const auto neg_mask = svcmplt_f64(ptrue, values, ZERO_F64);
    auto angle = svsel_f64(neg_mask, pi, ZERO_F64);
    angle = svsel_f64(nan_mask, nan_vec, angle);
    return angle;
  }
  Vectorized<double> real() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::real()"<<std::endl;
    return *this;
  }
  Vectorized<double> imag() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::imag()"<<std::endl;
    return Vectorized<double>(0.0);
  }
  Vectorized<double> conj() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::conj()"<<std::endl;
    return *this;
  }
  Vectorized<double> acos() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::acos()"<<std::endl;
    return Vectorized<double>(Sleef_acosdx_u10sve(values));
  }
  Vectorized<double> acosh() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::acosh()"<<std::endl;
    return Vectorized<double>(Sleef_acoshdx_u10sve(values));
  }
  Vectorized<double> asin() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::asin()"<<std::endl;
    return Vectorized<double>(Sleef_asindx_u10sve(values));
  }
  Vectorized<double> atan() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::atan()"<<std::endl;
    return Vectorized<double>(Sleef_atandx_u10sve(values));
  }
  Vectorized<double> atanh() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::atanh()"<<std::endl;
    return Vectorized<double>(Sleef_atanhdx_u10sve(values));
  }
  Vectorized<double> atan2(const Vectorized<double> &b) const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::atan2()"<<std::endl;
    return Vectorized<double>(Sleef_atan2dx_u10sve(values, b));
  }
  Vectorized<double> copysign(const Vectorized<double> &sign) const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::copysign()"<<std::endl;
    return Vectorized<double>(Sleef_copysigndx_sve(values, sign));
  }
  Vectorized<double> erf() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::erf()"<<std::endl;
    return Vectorized<double>(Sleef_erfdx_u10sve(values));
  }
  Vectorized<double> erfc() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::erfc()"<<std::endl;
    return Vectorized<double>(Sleef_erfcdx_u15sve(values));
  }
  Vectorized<double> erfinv() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::erfinv()"<<std::endl;
    return map(calc_erfinv);
  }
  Vectorized<double> exp() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::exp()"<<std::endl;
    return Vectorized<double>(Sleef_expdx_u10sve(values));
  }
  Vectorized<double> exp2() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::exp2()"<<std::endl;
    return Vectorized<double>(Sleef_exp2dx_u10sve(values));
  }
  Vectorized<double> expm1() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::expm1()"<<std::endl;
    return Vectorized<double>(Sleef_expm1dx_u10sve(values));
  }
  Vectorized<double> exp_u20() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::exp_u20()"<<std::endl;
    return exp();
  }
  Vectorized<double> fmod(const Vectorized<double>& q) const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::fmod()"<<std::endl;
    return Vectorized<double>(Sleef_fmoddx_sve(values, q));
  }
  Vectorized<double> hypot(const Vectorized<double> &b) const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::hypot()"<<std::endl;
    return Vectorized<double>(Sleef_hypotdx_u05sve(values, b));
  }
  Vectorized<double> i0() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::i0()"<<std::endl;
    return map(calc_i0);
  }
  Vectorized<double> i0e() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::i0e()"<<std::endl;
    return map(calc_i0e);
  }
  Vectorized<double> digamma() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::digamma()"<<std::endl;
    return map(calc_digamma);
  }
  Vectorized<double> igamma(const Vectorized<double> &x) const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::igamma()"<<std::endl;
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> igammac(const Vectorized<double> &x) const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::igammac()"<<std::endl;
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> nextafter(const Vectorized<double> &b) const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::nextafter()"<<std::endl;
    return Vectorized<double>(Sleef_nextafterdx_sve(values, b));
  }
  Vectorized<double> log() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::log()"<<std::endl;
    return Vectorized<double>(Sleef_logdx_u10sve(values));
  }
  Vectorized<double> log2() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::log2()"<<std::endl;
    return Vectorized<double>(Sleef_log2dx_u10sve(values));
  }
  Vectorized<double> log10() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::log10()"<<std::endl;
    return Vectorized<double>(Sleef_log10dx_u10sve(values));
  }
  Vectorized<double> log1p() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::log1p()"<<std::endl;
    return Vectorized<double>(Sleef_log1pdx_u10sve(values));
  }
  Vectorized<double> frac() const;
  Vectorized<double> sin() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::sin()"<<std::endl;
    return Vectorized<double>(Sleef_sindx_u10sve(values));
  }
  Vectorized<double> sinh() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::sinh()"<<std::endl;
    return Vectorized<double>(Sleef_sinhdx_u10sve(values));
  }
  Vectorized<double> cos() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::cos()"<<std::endl;
    return Vectorized<double>(Sleef_cosdx_u10sve(values));
  }
  Vectorized<double> cosh() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::cosh()"<<std::endl;
    return Vectorized<double>(Sleef_coshdx_u10sve(values));
  }
  Vectorized<double> ceil() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::ceil()"<<std::endl;
    return svrintp_f64_x(ptrue, values);
  }
  Vectorized<double> floor() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::floor()"<<std::endl;
    return svrintm_f64_x(ptrue, values);
  }
  Vectorized<double> neg() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::neg()"<<std::endl;
    return svneg_f64_x(ptrue, values);
  }
  Vectorized<double> round() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::round()"<<std::endl;
    return svrinti_f64_x(ptrue, values);
  }
  Vectorized<double> tan() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::tan()"<<std::endl;
    return Vectorized<double>(Sleef_tandx_u10sve(values));
  }
  Vectorized<double> tanh() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::tanh()"<<std::endl;
    return Vectorized<double>(Sleef_tanhdx_u10sve(values));
  }
  Vectorized<double> trunc() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::trunc()"<<std::endl;
    return svrintz_f64_x(ptrue, values);
  }
  Vectorized<double> lgamma() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::lgamma()"<<std::endl;
    return Vectorized<double>(Sleef_lgammadx_u10sve(values));
  }
  Vectorized<double> sqrt() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::sqrt()"<<std::endl;
    return svsqrt_f64_x(ptrue, values);
  }
  Vectorized<double> reciprocal() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::reciprocal()"<<std::endl;
    return svdivr_f64_x(ptrue, values, ONE_F64);
  }
  Vectorized<double> rsqrt() const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::rsqrt()"<<std::endl;
    return svdivr_f64_x(ptrue, svsqrt_f64_x(ptrue, values), ONE_F64);
  }
  Vectorized<double> pow(const Vectorized<double> &b) const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::pow()"<<std::endl;
    return Vectorized<double>(Sleef_powdx_u10sve(values, b));
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vectorized<double> operator==(const Vectorized<double>& other) const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::operator==()"<<std::endl;
    svbool_t mask = svcmpeq_f64(ptrue, values, other);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }

  Vectorized<double> operator!=(const Vectorized<double>& other) const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::operator!=()"<<std::endl;
    svbool_t mask = svcmpne_f64(ptrue, values, other);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }

  Vectorized<double> operator<(const Vectorized<double>& other) const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::operator<()"<<std::endl;
    svbool_t mask = svcmplt_f64(ptrue, values, other);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }

  Vectorized<double> operator<=(const Vectorized<double>& other) const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::operator<=()"<<std::endl;
    svbool_t mask = svcmple_f64(ptrue, values, other);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }

  Vectorized<double> operator>(const Vectorized<double>& other) const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::operator>()"<<std::endl;
    svbool_t mask = svcmpgt_f64(ptrue, values, other);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }

  Vectorized<double> operator>=(const Vectorized<double>& other) const {
    std::cout<<"vec/sve/vec_double.h/Vectorized<double>::operator>=()"<<std::endl;
    svbool_t mask = svcmpge_f64(ptrue, values, other);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }

  Vectorized<double> eq(const Vectorized<double>& other) const;
  Vectorized<double> ne(const Vectorized<double>& other) const;
  Vectorized<double> gt(const Vectorized<double>& other) const;
  Vectorized<double> ge(const Vectorized<double>& other) const;
  Vectorized<double> lt(const Vectorized<double>& other) const;
  Vectorized<double> le(const Vectorized<double>& other) const;
};

template <>
Vectorized<double> inline operator+(const Vectorized<double>& a, const Vectorized<double>& b) {
  std::cout<<"vec/sve/vec_double.h/operator+()"<<std::endl;
  return svadd_f64_x(ptrue, a, b);
}

template <>
Vectorized<double> inline operator-(const Vectorized<double>& a, const Vectorized<double>& b) {
  std::cout<<"vec/sve/vec_double.h/operator-()"<<std::endl;
  return svsub_f64_x(ptrue, a, b);
}

template <>
Vectorized<double> inline operator*(const Vectorized<double>& a, const Vectorized<double>& b) {
  std::cout<<"vec/sve/vec_double.h/operator*()"<<std::endl;
  return svmul_f64_x(ptrue, a, b);
}

template <>
Vectorized<double> inline operator/(const Vectorized<double>& a, const Vectorized<double>& b) {
  std::cout<<"vec/sve/vec_double.h/operator/()"<<std::endl;
  return svdiv_f64_x(ptrue, a, b);
}

// frac. Implement this here so we can use subtraction
Vectorized<double> inline Vectorized<double>::frac() const {
  std::cout<<"vec/sve/vec_double.h/Vectorized<double>::frac()"<<std::endl;
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<double> inline maximum(const Vectorized<double>& a, const Vectorized<double>& b) {
  std::cout<<"vec/sve/vec_double.h/maximum()"<<std::endl;
  return svmax_f64_x(ptrue, a, b);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<double> inline minimum(const Vectorized<double>& a, const Vectorized<double>& b) {
  std::cout<<"vec/sve/vec_double.h/minimum()"<<std::endl;
  return svmin_f64_x(ptrue, a, b);
}

template <>
Vectorized<double> inline clamp(const Vectorized<double>& a, const Vectorized<double>& min, const Vectorized<double>& max) {
  std::cout<<"vec/sve/vec_double.h/clamp()"<<std::endl;
  return svmin_f64_x(ptrue, max, svmax_f64_x(ptrue, min, a));
}

template <>
Vectorized<double> inline clamp_max(const Vectorized<double>& a, const Vectorized<double>& max) {
  std::cout<<"vec/sve/vec_double.h/clamp_max()"<<std::endl;
  return svmin_f64_x(ptrue, max, a);
}

template <>
Vectorized<double> inline clamp_min(const Vectorized<double>& a, const Vectorized<double>& min) {
  std::cout<<"vec/sve/vec_double.h/clamp_min()"<<std::endl;
  return svmax_f64_x(ptrue, min, a);
}

template <>
Vectorized<double> inline operator&(const Vectorized<double>& a, const Vectorized<double>& b) {
  std::cout<<"vec/sve/vec_double.h/operator&()"<<std::endl;
  return svreinterpret_f64_s64(svand_s64_x(ptrue, svreinterpret_s64_f64(a), svreinterpret_s64_f64(b)));
}

template <>
Vectorized<double> inline operator|(const Vectorized<double>& a, const Vectorized<double>& b) {
  std::cout<<"vec/sve/vec_double.h/operator|()"<<std::endl;
  return svreinterpret_f64_s64(svorr_s64_x(ptrue, svreinterpret_s64_f64(a), svreinterpret_s64_f64(b)));
}

template <>
Vectorized<double> inline operator^(const Vectorized<double>& a, const Vectorized<double>& b) {
  std::cout<<"vec/sve/vec_double.h/operator^()"<<std::endl;
  return svreinterpret_f64_s64(sveor_s64_x(ptrue, svreinterpret_s64_f64(a), svreinterpret_s64_f64(b)));
}

Vectorized<double> inline Vectorized<double>::eq(const Vectorized<double>& other) const {
  std::cout<<"vec/sve/vec_double.h/Vectorized<double>::eq()"<<std::endl;
  return (*this == other) & Vectorized<double>(1.0);
}

Vectorized<double> inline Vectorized<double>::ne(const Vectorized<double>& other) const {
  std::cout<<"vec/sve/vec_double.h/Vectorized<double>::ne()"<<std::endl;
  return (*this != other) & Vectorized<double>(1.0);
}

Vectorized<double> inline Vectorized<double>::gt(const Vectorized<double>& other) const {
  std::cout<<"vec/sve/vec_double.h/Vectorized<double>::gt()"<<std::endl;
  return (*this > other) & Vectorized<double>(1.0);
}

Vectorized<double> inline Vectorized<double>::ge(const Vectorized<double>& other) const {
  std::cout<<"vec/sve/vec_double.h/Vectorized<double>::ge()"<<std::endl;
  return (*this >= other) & Vectorized<double>(1.0);
}

Vectorized<double> inline Vectorized<double>::lt(const Vectorized<double>& other) const {
  std::cout<<"vec/sve/vec_double.h/Vectorized<double>::lt()"<<std::endl;
  return (*this < other) & Vectorized<double>(1.0);
}

Vectorized<double> inline Vectorized<double>::le(const Vectorized<double>& other) const {
  std::cout<<"vec/sve/vec_double.h/Vectorized<double>::le()"<<std::endl;
  return (*this <= other) & Vectorized<double>(1.0);
}

template <>
inline void convert(const double* src, double* dst, int64_t n) {
  std::cout<<"vec/sve/vec_double.h/convert()"<<std::endl;
  const int64_t fraction = n % Vectorized<double>::size();
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vectorized<double>::size()) {
    svst1_f64(ptrue, dst + i, svldnt1_f64(ptrue, src + i));
  }
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += Vectorized<double>::size()) {
    svbool_t pg = svwhilelt_b64(i, n);
    svst1_f64(pg, dst + i, svldnt1_f64(pg, src + i));
  }
}

template <>
Vectorized<double> inline fmadd(const Vectorized<double>& a, const Vectorized<double>& b, const Vectorized<double>& c) {
  std::cout<<"vec/sve/vec_double.h/fmadd()"<<std::endl;
  return svmad_f64_x(ptrue, a, b, c);
}

#endif // defined(CPU_CAPABILITY_SVE)

}}}
