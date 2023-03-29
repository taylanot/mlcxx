
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

#ifndef JEM_BASE_PROMOTE_H
#define JEM_BASE_PROMOTE_H

#include <jem/defines.h>


JEM_BEGIN_PACKAGE_BASE

JEM_BEGIN_INTERNAL


//-----------------------------------------------------------------------
//   class TypeRank & specializations
//-----------------------------------------------------------------------


template <class T>

  class TypeRank

{
 public:

  static const int  RANK = 99 + (int) sizeof(T);

};

template <>

  class TypeRank<bool>

{
 public:

  static const int  RANK = 1;

};

template <>

  class TypeRank<int>

{
 public:

  static const int  RANK = TypeRank<bool>::RANK + 1;

};

template <>

  class TypeRank<unsigned int>

{
 public:

  static const int  RANK = TypeRank<int>::RANK + 1;

};

template <>

  class TypeRank<lint>

{
 public:

  static const int  RANK = TypeRank<unsigned int>::RANK + 1;

};

template <>

  class TypeRank<ulint>

{
 public:

  static const int  RANK = TypeRank<lint>::RANK + 1;

};

template <>

  class TypeRank<float>

{
 public:

  static const int  RANK = TypeRank<ulint>::RANK + 1;

};

template <>

  class TypeRank<double>

{
 public:

  static const int  RANK = TypeRank<float>::RANK + 1;

};

#ifdef JEM_STD_COMPLEX

template <class T>

  class TypeRank< std::complex<T> >

{
 public:

  static const int  RANK = 20 + TypeRank<T>::RANK;

};

#endif // JEM_STD_COMPLEX


//-----------------------------------------------------------------------
//   class SelectType & specializations
//-----------------------------------------------------------------------


template <bool SelectFirst, class T, class U>

  class SelectType

{};

template <class T, class U>

  class SelectType<true,T,U>

{
 public:

  typedef T         ResultType;

};

template <class T, class U>

  class SelectType<false,T,U>

{
 public:

  typedef U         ResultType;

};


JEM_END_INTERNAL


//-----------------------------------------------------------------------
//   class PromoteType & specializations
//-----------------------------------------------------------------------


template <class T>

  class PromoteType

{
 public:

  typedef T         ResultType;

};

template <>

  class PromoteType<bool>

{
 public:

  typedef bool      ResultType;

};

template <>

  class PromoteType<char>

{
 public:

  typedef int       ResultType;

};

template <>

  class PromoteType<unsigned char>

{
 public:

  typedef int       ResultType;

};

template <>

  class PromoteType<short>

{
 public:

  typedef int       ResultType;

};

template <>

  class PromoteType<unsigned short>

{
 public:

  typedef int       ResultType;

};


//-----------------------------------------------------------------------
//   class PromoteTypes & specializations
//-----------------------------------------------------------------------


template <class T, class U>

  class PromoteTypes

{
 private:

  typedef typename
    PromoteType<T>::ResultType  FirstType;
  typedef typename
    PromoteType<U>::ResultType  SecndType;

  static const int  FIRST_RANK = internal::TypeRank<FirstType>::RANK;
  static const int  SECND_RANK = internal::TypeRank<SecndType>::RANK;


 public:

  typedef typename  internal::SelectType
    <(FIRST_RANK > SECND_RANK),
     FirstType, SecndType> :: ResultType  ResultType;

};

template <class T>

  class PromoteTypes<T,T>

{
 public:

  typedef typename
    PromoteType<T>::ResultType  ResultType;

};


JEM_END_PACKAGE_BASE

#endif
