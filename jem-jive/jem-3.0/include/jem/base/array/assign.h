
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

#ifndef JEM_BASE_ARRAY_ASSIGN_H
#define JEM_BASE_ARRAY_ASSIGN_H

#include <jem/base/array/base.h>


JEM_BEGIN_PACKAGE_BASE
JEM_BEGIN_NAMESPACE( array )


//-----------------------------------------------------------------------
//   assign functions
//-----------------------------------------------------------------------


template <class T, class U, int N, class E1, class E2, class Op>

  inline void             assign

  ( const Array<T,N,E1>&    lhs,
    const Array<U,N,E2>&    rhs,
    const Op&               op );

template <class T, class U, int N, class E1, class E2, class Op>

  inline void             assign

  ( const Array<T,N,E1>&    lhs,
    const Array<U,N,E2>&    rhs,
    const Tuple<idx_t,N>&   sh,
    const Op&               op );

template <class T, class U, int N, class E, class Op>

  inline void             assign

  ( const Array<T,N,E>&     lhs,
    const U&                rhs,
    const Op&               op );

template <class T, class U, int N, class E, class Op>

  inline void             assign

  ( const Array<T,N,E>&     lhs,
    const U&                rhs,
    const Tuple<idx_t,N>&   sh,
    const Op&               op );

template <class T, class U, int N, class E, class Op>

  inline void             assignArray

  ( T*                      lhs,
    const Array<U,N,E>&     rhs,
    idx_t                   len,
    const Op&               op );

template <class T, class U, int N, class E, class Op>

  inline void             assignArray

  ( T*                      lhs,
    const Tuple<idx_t,N>&   lst,
    const Array<U,N,E>&     rhs,
    const Tuple<idx_t,N>&   sh,
    const Op&               op );

template <class T, class U, int N, class E1, class E2, class Op>

  void                    assignArray

  ( const Array<T,N,E1>&    lhs,
    const Array<U,N,E2>&    rhs,
    idx_t                   len,
    const Op&               op );

template <class T, class U, int N, class E1, class E2, class Op>

  inline void             assignArray

  ( const Array<T,N,E1>&    lhs,
    const Array<U,N,E2>&    rhs,
    const Tuple<idx_t,N>&   sh,
    const Op&               op );

template <class T, class U, class E1, class E2, class Op>

  void                    assignArray

  ( const Array<T,1,E1>&    lhs,
    const Array<U,1,E2>&    rhs,
    const Tuple<idx_t,1>&   sh,
    const Op&               op );

template <class T, class U, int N, class E, class Op>

  void                    assignScalar

  ( const Array<T,N,E>&     lhs,
    const U&                rhs,
    idx_t                   len,
    const Op&               op );

template <class T, class U, int N, class E, class Op>

  void                    assignScalar

  ( const Array<T,N,E>&     lhs,
    const U&                rhs,
    const Tuple<idx_t,N>&   sh,
    const Op&               op );

template <class T, class U, class E, class Op>

  void                    assignScalar

  ( const Array<T,1,E>&     lhs,
    const U&                rhs,
    const Tuple<idx_t,1>&   sh,
    const Op&               op );


JEM_END_NAMESPACE( array )
JEM_END_PACKAGE_BASE

#include <jem/base/array/assign.tcc>

#endif
