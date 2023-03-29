
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

#ifndef JEM_NUMERIC_ALGEBRA_MATMUL_H
#define JEM_NUMERIC_ALGEBRA_MATMUL_H

#include <jem/base/array/Array.h>


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   matmul
//-----------------------------------------------------------------------


template <class T>

  void                    matmul

  ( const Array<T,1>&       r,
    const Array<T,2>&       a,
    const Array<T,1>&       b );

template <class T>

  void                    matmul

  ( const Array<T,1>&       r,
    const Array<T,1>&       a,
    const Array<T,2>&       b );

template <class T>

  void                    matmul

  ( const Array<T,2>&       r,
    const Array<T,1>&       a,
    const Array<T,1>&       b );

template <class T>

  void                    matmul

  ( const Array<T,2>&       r,
    const Array<T,2>&       a,
    const Array<T,2>&       b );


void                      matmul

  ( const Array<double,1>&  r,
    const Array<double,2>&  a,
    const Array<double,1>&  b );

void                      matmul

  ( const Array<double,1>&  r,
    const Array<double,1>&  a,
    const Array<double,2>&  b );

void                      matmul

  ( const Array<double,2>&  r,
    const Array<double,1>&  a,
    const Array<double,1>&  b );

void                      matmul

  ( const Array<double,2>&  r,
    const Array<double,2>&  a,
    const Array<double,2>&  b );


template <class T, int M, int N, int P>

  inline void             matmul

  ( Tuple<T,M,P>&           r,
    const Tuple<T,M,N>&     a,
    const Tuple<T,N,P>&     b );

template <class T, int M, int N>

  inline void             matmul

  ( Tuple<T,M,1>&           r,
    const Tuple<T,M,N>&     a,
    const Tuple<T,N,1>&     b );

template <class T, int M, int N>

  inline void             matmul

  ( Tuple<T,N,1>&           r,
    const Tuple<T,M,1>&     a,
    const Tuple<T,M,N>&     b );

template <class T, int M, int N>

  inline void             matmul

  ( Tuple<T,M,N>&           r,
    const Tuple<T,M,1>&     a,
    const Tuple<T,N,1>&     b );


template <class T, int M, int P>

  void                    matmul

  ( Tuple<T,M,P>&           r,
    const Array<T,2>&       a,
    const Array<T,2>&       b );

template <class T, int M, int N>

  void                    matmul

  ( const Array<T,2>&       r,
    const Tuple<T,M,N>&     a,
    const Array<T,2>&       b );

template <class T, int N, int P>

  void                    matmul

  ( const Array<T,2>&       r,
    const Array<T,2>&       a,
    const Tuple<T,N,P>&     b );


template <class T>

  inline Array<T,1>       matmul

  ( const Array<T,2>&       a,
    const Array<T,1>&       b );

template <class T>

  inline Array<T,1>       matmul

  ( const Array<T,1>&       a,
    const Array<T,2>&       b );

template <class T>

  inline Array<T,2>       matmul

  ( const Array<T,1>&       a,
    const Array<T,1>&       b );

template <class T>

  inline Array<T,2>       matmul

  ( const Array<T,2>&       a,
    const Array<T,2>&       b );

template <class T, int M, int N, int P>

  inline Tuple<T,M,P>     matmul

  ( const Tuple<T,M,N>&     a,
    const Tuple<T,N,P>&     b );

template <class T, int M, int N>

  inline Tuple<T,M,1>     matmul

  ( const Tuple<T,M,N>&     a,
    const Tuple<T,N,1>&     b );

template <class T, int M, int N>

  inline Tuple<T,N,1>     matmul

  ( const Tuple<T,M,1>&     a,
    const Tuple<T,M,N>&     b );

template <class T, int M, int N>

  inline Tuple<T,M,N>     matmul

  ( const Tuple<T,M,1>&     a,
    const Tuple<T,N,1>&     b );

template <class T, int M, int N>

  inline Array<T,2>       matmul

  ( const Tuple<T,M,N>&     a,
    const Array<T,2>&       b );

template <class T, int N, int P>

  inline Array<T,2>       matmul

  ( const Array<T,2>&       a,
    const Tuple<T,N,P>&     b );





//#######################################################################
//   Implementation
//#######################################################################


JEM_END_PACKAGE( numeric )

#include <jem/numeric/algebra/matmul.tcc>

#endif
