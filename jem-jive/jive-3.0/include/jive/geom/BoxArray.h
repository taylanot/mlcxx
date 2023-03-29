
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

#ifndef JIVE_GEOM_BOXARRAY_H
#define JIVE_GEOM_BOXARRAY_H

#include <jem/util/Flex.h>
#include <jive/Array.h>
#include <jive/geom/import.h>


JIVE_BEGIN_PACKAGE( geom )


class MaskMatrix;


//-----------------------------------------------------------------------
//   class BoxArray
//-----------------------------------------------------------------------


class BoxArray
{
 public:

  typedef BoxArray          Self;


  explicit                  BoxArray

    ( idx_t                   rank );

  inline idx_t              size          () const;
  inline idx_t              rank          () const;
  inline idx_t              rank2         () const;
  inline void               clear         ();

  inline void               reserve

    ( idx_t                   boxCount );

  inline void               trimToSize    ();

  inline void               checkIndex

    ( idx_t                   ibox )         const;

  void                      checkBoxes

    ( const String&           context )      const;

  void                      getEnclosingBox

    ( const Vector&           box )          const;

  idx_t                     addBox

    ( const Vector&           box,
      idx_t                   color = 0 );

  idx_t                     addBoxes

    ( const Matrix&           boxes,
      idx_t                   color = 0 );

  idx_t                     addBoxes

    ( const Matrix&           boxes,
      const IdxVector&        colors );

  void                      reorderBoxes

    ( const Reordering&       reord );

  void                      getBox

    ( const Vector&           box,
      idx_t                   ibox )         const;

  void                      getBoxes

    ( const Matrix&           boxes )        const;

  void                      getSomeBoxes

    ( const Matrix&           boxes,
      const IdxVector&        iboxes )       const;

  inline idx_t              getColor

    ( idx_t                   ibox   )       const;

  void                      setColors

    ( const IdxVector&        colors );

  void                      setSomeColors

    ( const IdxVector&        iboxes,
      const IdxVector&        colors );

  void                      updateBoxes

    ( const Matrix&           boxes );

  void                      updateSomeBoxes

    ( const IdxVector&        iboxes,
      const Matrix&           boxes );

  idx_t                     findChangedBoxes

    ( const IdxVector&        iboxes,
      const Matrix&           boxes )        const;

  idx_t                     findBoxNeighbors

    ( idx_t*                  iboxes,
      const Vector&           box,
      const idx_t*            ilist,
      idx_t                   count )        const;

  idx_t                     findBoxNeighbors

    ( idx_t*                  iboxes,
      const Vector&           box,
      idx_t                   color,
      const MaskMatrix&       mask,
      const idx_t*            ilist,
      idx_t                   count )        const;

  void                      readFrom

    ( DataInput&              in );

  void                      writeTo

    ( DataOutput&             out )          const;


 private:

  void                      updateBoxes_

    ( idx_t                   from,
      const Matrix&           boxes );

  void                      updateBounds_ ();

  void                      indexError_

    ( idx_t                   ibox,
      idx_t                   ubound )       const;


 private:

  idx_t                     rank_;
  idx_t                     rank2_;

  jem::util::Flex<double>   boxes_;
  jem::util::Flex<idx_t>    colors_;
  jem::util::Flex<double>   bounds_;

  bool                      validBounds_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class Input>

  inline void               decode

  ( Input&                    in,
    BoxArray&                 ba );

template <class Output>

  inline void               encode

  ( Output&                   out,
    const BoxArray&           ba );





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class BoxArray
//=======================================================================

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t BoxArray::size () const
{
  return colors_.size ();
}


//-----------------------------------------------------------------------
//   rank & rank2
//-----------------------------------------------------------------------


inline idx_t BoxArray::rank () const
{
  return rank_;
}


inline idx_t BoxArray::rank2 () const
{
  return rank2_;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


inline void BoxArray::clear ()
{
  colors_.clear ();
  boxes_ .clear ();

  validBounds_ = false;
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


inline void BoxArray::reserve ( idx_t n )
{
  colors_.reserve ( n );
  boxes_ .reserve ( n * rank2_ );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


inline void BoxArray::trimToSize ()
{
  colors_.trimToSize ();
  boxes_ .trimToSize ();
}


//-----------------------------------------------------------------------
//   checkIndex
//-----------------------------------------------------------------------


inline void BoxArray::checkIndex ( idx_t ibox ) const
{
  if ( ibox < 0 || ibox >= colors_.size() )
  {
    indexError_ ( ibox, colors_.size() );
  }
}


//-----------------------------------------------------------------------
//   getColor
//-----------------------------------------------------------------------


inline idx_t BoxArray::getColor ( idx_t ibox ) const
{
  return colors_[ibox];
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input>

  inline void         decode

  ( Input&              in,
    BoxArray&           ba )

{
  ba.readFrom ( in );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output>

  inline void         encode

  ( Output&             out,
    const BoxArray&     ba )

{
  ba.writeTo ( out );
}


JIVE_END_PACKAGE( geom )

#endif
