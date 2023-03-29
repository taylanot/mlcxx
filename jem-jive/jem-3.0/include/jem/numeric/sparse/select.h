
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

#ifndef JEM_NUMERIC_SPARSE_SELECT_H
#define JEM_NUMERIC_SPARSE_SELECT_H

#include <jem/numeric/sparse/SparseMatrix.h>


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   class SparseSelector
//-----------------------------------------------------------------------


class SparseSelector
{
 public:

  static SparseStruct         select

    ( const SparseStruct&       src,
      const Array<idx_t>&       rowMap,
      const Array<idx_t>&       colMap );

  template <class T>

    static SparseMatrix<T>    select

    ( const SparseMatrix<T>&    src,
      const Array<idx_t>&       rowMap,
      const Array<idx_t>&       colMap );

  static Array<idx_t>         makeMap

    ( idx_t                     size,
      const String&             kind,
      const Array<idx_t>&       map );

  static Array<idx_t>         makeMap

    ( idx_t                     size,
      const String&             kind,
      const Array<bool>&        mask );

  static Array<idx_t>         makeMap

    ( idx_t                     size,
      const String&             kind,
      const Slice&              slice );

  static Array<idx_t>         makeMap

    ( idx_t                     size,
      const String&             kind,
      const SliceAll&           slice );

  static Array<idx_t>         makeMap

    ( idx_t                     size,
      const String&             kind,
      const SliceFromTo&        slice );

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class R, class C>

  SparseStructure             select

  ( const SparseStruct&         src,
    const R&                    rows,
    const C&                    cols );

template <class T, class R, class C>

  SparseMatrix<T>             select

  ( const SparseMatrix<T>&      src,
    const R&                    rows,
    const C&                    cols );




//#######################################################################
//   Implementation
//#######################################################################

JEM_END_PACKAGE( numeric )

#include <jem/numeric/sparse/select.tcc>

#endif
