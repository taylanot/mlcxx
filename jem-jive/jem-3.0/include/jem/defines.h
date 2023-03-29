
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
 *
 *  Commercial License Usage
 *
 *  This file may be used under the terms of a commercial license
 *  provided with the software, or under the terms contained in a written
 *  agreement between you and DRG. For more information contact DRG at
 *  http://www.dynaflow.com.
 *
 *  GNU Lesser General Public License Usage
 *
 *  Alternatively, this file may be used under the terms of the GNU
 *  Lesser General Public License version 2.1 or version 3 as published
 *  by the Free Software Foundation and appearing in the file
 *  LICENSE.LGPLv21 and LICENSE.LGPLv3 included in the packaging of this
 *  file. Please review the following information to ensure the GNU
 *  Lesser General Public License requirements will be met:
 *  https://www.gnu.org/licenses/lgpl.html and
 *  http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */

#ifndef JEM_DEFINES_H
#define JEM_DEFINES_H

#include <jem/config.h>


//-----------------------------------------------------------------------
//   generic macros
//-----------------------------------------------------------------------


#define JEM_CONCAT_(a,b) a ## b
#define JEM_CONCAT(a,b)  JEM_CONCAT_(a,b)

#define JEM_LINT_C(i)    JEM_CONCAT(i,LL)
#define JEM_ULINT_C(i)   JEM_CONCAT(i,ULL)


//-----------------------------------------------------------------------
//   file system macros
//-----------------------------------------------------------------------


#if   defined(JEM_OS_POSIX)
#  define JEM_FS_UNIX
#elif defined(JEM_OS_WINDOWS)
#  define JEM_FS_WINDOWS
#else
#  error "Unsupported file system -- sorry"
#endif


//-----------------------------------------------------------------------
//   the macro JEM_FUNC
//-----------------------------------------------------------------------


#if   defined(__GNUC__)
#  define JEM_FUNC __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#  define JEM_FUNC __FUNCSIG__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
#  define JEM_FUNC __func__
#else
#  define JEM_FUNC "unknown function"
#endif


//-----------------------------------------------------------------------
//   the macro JEM_RESTRICT
//-----------------------------------------------------------------------


#if   defined(__GNUC__)
#  define JEM_RESTRICT __restrict__
#elif defined(__sgi)
#  define JEM_RESTRICT restrict
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
#  define JEM_RESTRICT restrict
#else
#  define JEM_RESTRICT
#endif


//-----------------------------------------------------------------------
//   some other macros
//-----------------------------------------------------------------------


#if defined(JEM_USE_THREADS) && ! defined(_REENTRANT)
#  define _REENTRANT
#endif


//-----------------------------------------------------------------------
//   include standard types
//-----------------------------------------------------------------------


#include <cstddef>


//-----------------------------------------------------------------------
//   namespace macros
//-----------------------------------------------------------------------


#define JEM_BEGIN_PACKAGE_BASE       namespace jem {
#define JEM_END_PACKAGE_BASE         }

#define JEM_BEGIN_PACKAGE( name )    namespace jem { namespace name {
#define JEM_END_PACKAGE( name )      } }

#define JEM_BEGIN_INTERNAL           namespace internal {
#define JEM_END_INTERNAL             }

#define JEM_BEGIN_NAMESPACE( name )  namespace name {
#define JEM_END_NAMESPACE( name )    }


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   some forward declarations and typedefs
//-----------------------------------------------------------------------


template <class T> class  IdxType;
template <class T> class  TypeTraits;
template <class T> class  ObjectTraits;

typedef unsigned char     byte;
typedef long long int     lint;
typedef unsigned
        long long int     ulint;

typedef
  IdxType<JEM_IDX_T>      idx_t;
typedef
  IdxType<JEM_UIDX_T>     uidx_t;


//-----------------------------------------------------------------------
//   class IdxType
//-----------------------------------------------------------------------


template <class T>

  class IdxType

{
 public:

  typedef T               ValueType;

                          IdxType       ()     = default;

  template <class U>
    explicit constexpr
    inline                IdxType

    ( U                     rhs )                noexcept;

  constexpr inline        IdxType

    ( T                     rhs )                noexcept;

  constexpr               IdxType

    ( const IdxType&        rhs )              = default;

  template <class U>
    explicit constexpr
    inline                IdxType

    ( const IdxType<U>&     rhs )                noexcept;

  template <class U>
    explicit inline       operator U    () const;
  constexpr  inline       operator T    () const noexcept;
  explicit constexpr
    inline                operator bool () const noexcept;

  inline IdxType&         operator ++   ()       noexcept;
  inline IdxType          operator ++   ( int )  noexcept;
  inline IdxType&         operator --   ()       noexcept;
  inline IdxType          operator --   ( int )  noexcept;


 public:

  T                       value;


 private:

  static void             castError_    ();

};


//-----------------------------------------------------------------------
//   literal operators
//-----------------------------------------------------------------------


inline namespace literals
{

constexpr inline idx_t    operator"" _idx

  ( unsigned long long      arg );

constexpr inline uidx_t   operator"" _uidx

  ( unsigned long long      arg );

constexpr inline lint     operator"" _lint

  ( unsigned long long      arg );

constexpr inline ulint    operator"" _ulint

  ( unsigned long long      arg );

}


//#######################################################################
//  implementation
//#######################################################################

//=======================================================================
//   class IdxType
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class T>
  template <class U>

  constexpr inline IdxType<T>::IdxType ( U rhs ) noexcept :

  value ( (T) rhs )

{}


template <class T>

  constexpr inline IdxType<T>::IdxType ( T rhs ) noexcept :

  value ( rhs )

{}


template <class T>
  template <class U>

  constexpr inline
  IdxType<T>::IdxType ( const IdxType<U>& rhs ) noexcept :

  value ( (T) rhs.value )

{}


//-----------------------------------------------------------------------
//   cast operators
//-----------------------------------------------------------------------


template <class T>
  template <class U>

  inline IdxType<T>::operator U () const

{
  if ( sizeof(U) >= sizeof(T) )
  {
    return U ( value );
  }
  else
  {
    U  tmp = (U) value;

    if ( (T) tmp != value )
    {
      castError_ ();
    }

    return tmp;
  }
}


template <class T>

  constexpr inline IdxType<T>::operator T () const noexcept

{
  return value;
}


template <class T>

  constexpr inline IdxType<T>::operator bool () const noexcept

{
  return (value ? true : false);
}


//-----------------------------------------------------------------------
//   increment/decrement operators
//-----------------------------------------------------------------------


template <class T>

  inline IdxType<T>& IdxType<T>::operator ++ () noexcept

{
  value++;

  return *this;
}


template <class T>

  inline IdxType<T> IdxType<T>::operator ++ ( int ) noexcept

{
  IdxType  i = *this;

  value++;

  return i;
}


template <class T>

  inline IdxType<T>& IdxType<T>::operator -- () noexcept

{
  value--;

  return *this;
}


template <class T>

  inline IdxType<T> IdxType<T>::operator -- ( int ) noexcept

{
  IdxType  i = *this;

  value--;

  return i;
}


//-----------------------------------------------------------------------
//   equality & comparison operators
//-----------------------------------------------------------------------

// Note that the compiler will use the built-in comparison operators
// in combination with the cast operator in the case that an IdxType
// is compared with an integral type.

#undef  JEM_DEFINE_COMP_OP
#define JEM_DEFINE_COMP_OP( op )                             \
                                                             \
template <class T>                                           \
  constexpr inline                                           \
  bool operator op ( IdxType<T> lhs, IdxType<T> rhs )        \
{                                                            \
  return (lhs.value op rhs.value);                           \
}

JEM_DEFINE_COMP_OP( == )
JEM_DEFINE_COMP_OP( != )
JEM_DEFINE_COMP_OP( >  )
JEM_DEFINE_COMP_OP( >= )
JEM_DEFINE_COMP_OP( <  )
JEM_DEFINE_COMP_OP( <= )

#undef JEM_DEFINE_COMP_OP


//-----------------------------------------------------------------------
//   unary operators
//-----------------------------------------------------------------------


#undef  JEM_DEFINE_UNARY_OP
#define JEM_DEFINE_UNARY_OP( op )                            \
                                                             \
template <class T>                                           \
  constexpr inline                                           \
  IdxType<T> operator op ( IdxType<T> lhs )                  \
{                                                            \
  return IdxType<T> ( op lhs.value );                        \
}

JEM_DEFINE_UNARY_OP( ! )
JEM_DEFINE_UNARY_OP( ~ )

#undef JEM_DEFINE_UNARY_OP


constexpr inline idx_t operator - ( idx_t lhs )
{
  return idx_t ( -lhs.value );
}


//-----------------------------------------------------------------------
//   binary operators
//-----------------------------------------------------------------------

// Note that the compiler will use the built-in binary operators
// in combination with the cast operator in the case that an IdxType
// is combined with an integral type.

#undef  JEM_DEFINE_BINARY_OP
#define JEM_DEFINE_BINARY_OP( op )                           \
                                                             \
template <class T>                                           \
  constexpr inline                                           \
  IdxType<T> operator op ( IdxType<T> lhs,                   \
                           IdxType<T> rhs )                  \
{                                                            \
  return IdxType<T> ( lhs.value op rhs.value );              \
}

JEM_DEFINE_BINARY_OP( + )
JEM_DEFINE_BINARY_OP( - )
JEM_DEFINE_BINARY_OP( * )
JEM_DEFINE_BINARY_OP( / )
JEM_DEFINE_BINARY_OP( % )
JEM_DEFINE_BINARY_OP( ^ )
JEM_DEFINE_BINARY_OP( | )
JEM_DEFINE_BINARY_OP( & )
JEM_DEFINE_BINARY_OP( && )
JEM_DEFINE_BINARY_OP( || )
JEM_DEFINE_BINARY_OP( >> )
JEM_DEFINE_BINARY_OP( << )

#undef JEM_DEFINE_BINARY_OP


//-----------------------------------------------------------------------
//   computed assignment operators
//-----------------------------------------------------------------------


#undef  JEM_DEFINE_ASSIGN_OP
#define JEM_DEFINE_ASSIGN_OP( op )                           \
                                                             \
template <class T>                                           \
  inline IdxType<T>& operator op ( IdxType<T>& lhs,          \
                                   IdxType<T>  rhs )         \
{                                                            \
  lhs.value op rhs.value;                                    \
  return lhs;                                                \
}                                                            \
                                                             \
inline idx_t&        operator op ( idx_t&      lhs,          \
                                   JEM_IDX_T   rhs )         \
{                                                            \
  lhs.value op rhs;                                          \
  return lhs;                                                \
}                                                            \
                                                             \
inline uidx_t&       operator op ( uidx_t&     lhs,          \
                                   JEM_UIDX_T  rhs )         \
{                                                            \
  lhs.value op rhs;                                          \
  return lhs;                                                \
}

JEM_DEFINE_ASSIGN_OP( += )
JEM_DEFINE_ASSIGN_OP( -= )
JEM_DEFINE_ASSIGN_OP( *= )
JEM_DEFINE_ASSIGN_OP( /= )
JEM_DEFINE_ASSIGN_OP( %= )
JEM_DEFINE_ASSIGN_OP( ^= )
JEM_DEFINE_ASSIGN_OP( |= )
JEM_DEFINE_ASSIGN_OP( &= )
JEM_DEFINE_ASSIGN_OP( >>= )
JEM_DEFINE_ASSIGN_OP( <<= )

#undef JEM_DEFINE_ASSIGN_OP


//-----------------------------------------------------------------------
//   specializations
//-----------------------------------------------------------------------


template <>
  void IdxType<JEM_IDX_T> ::castError_ ();


template <>
  void IdxType<JEM_UIDX_T>::castError_ ();


//-----------------------------------------------------------------------
//   literal operators
//-----------------------------------------------------------------------


inline namespace literals
{

constexpr inline idx_t    operator"" _idx

  ( unsigned long long      arg )

{
  return idx_t ( (JEM_IDX_T) arg );
}


constexpr inline uidx_t   operator"" _uidx

  ( unsigned long long      arg )

{
  return uidx_t ( (JEM_UIDX_T) arg );
}


constexpr inline lint     operator"" _lint

  ( unsigned long long      arg )

{
  return (lint) arg;
}


constexpr inline ulint    operator"" _ulint

  ( unsigned long long      arg )

{
  return (ulint) arg;
}

}


JEM_END_PACKAGE_BASE

#endif
