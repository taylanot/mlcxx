
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

#ifndef JEM_NUMERIC_ALGEBRA_UTILITIES_H
#define JEM_NUMERIC_ALGEBRA_UTILITIES_H

#include <jem/base/array/utilities.h>
#include <jem/base/array/Array.h>


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   public functions
//-----------------------------------------------------------------------


template <class T, class E1, class E2>

  inline Tuple<T,3>       crossProduct

  ( const Tuple<T,3,1,E1>&  a,
    const Tuple<T,3,1,E2>&  b );

template <class T, class E1, class E2>

  inline T                crossProduct

  ( const Tuple<T,2,1,E1>&  a,
    const Tuple<T,2,1,E2>&  b );

template <class T, int N, class E1, class E2>

  inline T                dotProduct

  ( const Tuple<T,N,1,E1>&  a,
    const Tuple<T,N,1,E2>&  b );

template <class T, class E1, class E2>

  inline T                dotProduct

  ( const Array<T,1,E1>&    a,
    const Array<T,1,E2>&    b );

template <class T, int N, class E>

  inline T                norm2

  ( const Tuple<T,N,1,E>&   v );

template <class T, class E>

  inline T                norm2

  ( const Array<T,1,E>&     v );

template <class T, int N>

  inline void             axpy

  ( const Array<T,N>&       r,
    T                       t,
    const Array<T,N>&       b );

template <class T, int N>

  inline void             axpy

  ( const Array<T,N>&       r,
    const Array<T,N>&       a,
    T                       t,
    const Array<T,N>&       b );




//#######################################################################
//   Implementation
//#######################################################################

JEM_END_PACKAGE( numeric )

#include <jem/numeric/algebra/utilities.tcc>

#ifdef JEM_STD_COMPLEX
#  include <jem/numeric/algebra/zutilities.h>
#endif

#endif
