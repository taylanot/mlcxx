
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

#ifndef JEM_NUMERIC_SPARSE_REORDER_H
#define JEM_NUMERIC_SPARSE_REORDER_H

#include <jem/base/array/Array.h>


JEM_BEGIN_PACKAGE( numeric )


class SparseStructure;


//-----------------------------------------------------------------------
//   class Reorder
//-----------------------------------------------------------------------


class Reorder
{
 public:

  typedef void              (*Algorithm)

    ( const Array<idx_t>&       iperm,
      const SparseStructure&    mstruc );


  static void                 nd

    ( const Array<idx_t>&       iperm,
      const SparseStructure&    mstruc );

  static void                 rcm

    ( const Array<idx_t>&       iperm,
      const SparseStructure&    mstruc );

  static void                 colamd

    ( const Array<idx_t>&       jperm,
      const SparseStructure&    mstruc );

  static void                 automa

    ( const Array<idx_t>&       jperm,
      const SparseStructure&    mstruc );

  static void                 superReorder

    ( const Array<idx_t>&       iperm,
      const SparseStructure&    mstruc,
      Algorithm                 reorder );

  static void                 mergeSuperNodes

    ( Array<idx_t>&             superNodes,
      SparseStructure&          superStruc,
      const SparseStructure&    inputStruc );

  static void                 expandSuperPerm

    ( const Array<idx_t>&       nodePerm,
      const Array<idx_t>&       superPerm,
      const Array<idx_t>&       superNodes );


 private:

  static idx_t                calcFill_

    ( const SparseStructure&    mstruct );

};


JEM_END_PACKAGE( numeric )

#endif
