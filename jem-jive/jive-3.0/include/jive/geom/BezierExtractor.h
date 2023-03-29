
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */

#ifndef JIVE_GEOM_BEZIEREXTRACTOR_H
#define JIVE_GEOM_BEZIEREXTRACTOR_H

#include <jive/defines.h>
#include <jive/Array.h>
#include <jive/geom/import.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class BezierExtractor
//-----------------------------------------------------------------------


class BezierExtractor
{
 public:

  typedef BezierExtractor   Self;

                            BezierExtractor   ();

  explicit                  BezierExtractor

    ( idx_t                   m );

                            BezierExtractor

    ( idx_t                   m,
      idx_t                   n );

  Self&                     operator =

    ( const Self&             rhs );

  void                      setIdentity

    ( idx_t                   n );

  inline bool               isContiguous      () const;
  inline idx_t              nonZeroCount      () const;

  void                      configure

    ( const Properties&       props );

  void                      getConfig

    ( const Properties&       conf )             const;


 public:

  jem::Tuple<idx_t,2>       shape;
  IdxVector                 irows;
  IdxVector                 jcols;
  Vector                    values;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


void                        matmul

  ( const Vector&             lhs,
    const BezierExtractor&    bez,
    const Vector&             rhs );

void                        matmul

  ( const Matrix&             lhs,
    const BezierExtractor&    bez,
    const Matrix&             rhs );

void                        decode

  ( DataInput&                in,
    BezierExtractor&          bez );

void                        encode

  ( DataOutput&               out,
    const BezierExtractor&    bez );



//#######################################################################
//  implementation
//#######################################################################

//-----------------------------------------------------------------------
//   isContiguous
//-----------------------------------------------------------------------


inline bool BezierExtractor::isContiguous () const
{
  return (irows.stride() +
          jcols.stride() + values.stride() == 3_idx);
}


//-----------------------------------------------------------------------
//   nonZeroCount
//-----------------------------------------------------------------------


inline idx_t BezierExtractor::nonZeroCount () const
{
  return irows.size ();
}


JIVE_END_PACKAGE( geom )

#endif
