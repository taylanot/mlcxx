
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

#ifndef JEM_BASE_LIMITS_H
#define JEM_BASE_LIMITS_H

#include <limits>
#include <jem/defines.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class Limits & specializations
//-----------------------------------------------------------------------


template <class T>

  class Limits

{
 public:

  static constexpr T      MIN_VALUE  =

    std::numeric_limits<T>::min ();

  static constexpr T      MAX_VALUE  =

    std::numeric_limits<T>::max ();

};


template <>

  class Limits<idx_t>

{
 public:

  static constexpr idx_t  MIN_VALUE  =

    std::numeric_limits<JEM_IDX_T>::min ();

  static constexpr idx_t  MAX_VALUE  =

    std::numeric_limits<JEM_IDX_T>::max ();

};


template <>

  class Limits<uidx_t>

{
 public:

  static constexpr uidx_t MIN_VALUE  =

    std::numeric_limits<JEM_UIDX_T>::min ();

  static constexpr uidx_t MAX_VALUE  =

    std::numeric_limits<JEM_UIDX_T>::max ();

};


template <>

  class Limits<float>

{
 public:

  static constexpr float  MIN_VALUE  =

    std::numeric_limits<float>::min     ();

  static constexpr float  MAX_VALUE  =

    std::numeric_limits<float>::max     ();

  static constexpr float  TINY_VALUE = 1.0e-30F;

  static constexpr float  EPSILON    =

    std::numeric_limits<float>::epsilon ();

};


template <>

  class Limits<double>

{
 public:

  static constexpr double MIN_VALUE  =

    std::numeric_limits<double>::min     ();

  static constexpr double MAX_VALUE  =

    std::numeric_limits<double>::max     ();

  static constexpr double TINY_VALUE = 1.0e-300;

  static constexpr double EPSILON    =

    std::numeric_limits<double>::epsilon ();

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class T>

  constexpr inline T    minOf () noexcept

{
  return Limits<T>::MIN_VALUE;
}


template <class T>

  constexpr inline T    minOf ( const T& ) noexcept

{
  return Limits<T>::MIN_VALUE;
}


template <class T>

  constexpr inline T    maxOf () noexcept

{
  return Limits<T>::MAX_VALUE;
}


template <class T>

  constexpr inline T    maxOf ( const T& ) noexcept

{
  return Limits<T>::MAX_VALUE;
}


constexpr inline bool   isTiny ( float x ) noexcept
{
  return ( x < Limits<float>::TINY_VALUE &&
          -x < Limits<float>::TINY_VALUE );
}


constexpr inline bool   isTiny ( double x ) noexcept
{
  return ( x < Limits<double>::TINY_VALUE &&
          -x < Limits<double>::TINY_VALUE );
}


JEM_END_PACKAGE_BASE

#endif
