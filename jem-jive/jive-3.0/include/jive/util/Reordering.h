
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

#ifndef JIVE_UTIL_REORDERING_H
#define JIVE_UTIL_REORDERING_H

#include <jem/base/assert.h>
#include <jive/Array.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class Reordering
//-----------------------------------------------------------------------


class Reordering
{
 public:

                          Reordering  ();

  explicit                Reordering

    ( const IdxVector&      fmap );

                          Reordering

    ( const IdxVector&      rmap,
      idx_t                 nsize );

                          Reordering

    ( const Reordering&     rhs );

  Reordering&             operator =

    ( const Reordering&     rhs );

  inline idx_t            start       () const;
  inline idx_t            oldSize     () const;
  inline idx_t            newSize     () const;
  inline bool             isSimple    () const;

  inline idx_t            oldIndex

    ( idx_t                 inew )       const;

  inline idx_t            newIndex

    ( idx_t                 iold )       const;

  inline IdxVector        forwardMap  () const;
  inline IdxVector        reverseMap  () const;
  inline const idx_t*     fmapPtr     () const;
  inline const idx_t*     rmapPtr     () const;


 public:

  IdxVector               fmap_;
  IdxVector               rmap_;

  idx_t                   start_;
  idx_t                   newSize_;
  bool                    simple_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class T, int N>

  void                    reorder

  ( jem::Array<T,N>&        data,
    const Reordering&       reord,
    T                       fill = T() );

template <class T, int N>

  void                    reorder

  ( jem::Array<T,N>&        data,
    int                     idim,
    const Reordering&       reord,
    T                       fill = T() );

Reordering                makeEraseReordering

  ( const IdxVector&        imask,
    idx_t                   osize );





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Reordering
//=======================================================================

//-----------------------------------------------------------------------
//   start
//-----------------------------------------------------------------------


inline idx_t Reordering::start () const
{
  return start_;
}


//-----------------------------------------------------------------------
//   oldSize
//-----------------------------------------------------------------------


inline idx_t Reordering::oldSize () const
{
  return fmap_.size ();
}


//-----------------------------------------------------------------------
//   newSize
//-----------------------------------------------------------------------


inline idx_t Reordering::newSize () const
{
  return newSize_;
}


//-----------------------------------------------------------------------
//   isSimple
//-----------------------------------------------------------------------


inline bool Reordering::isSimple () const
{
  return simple_;
}


//-----------------------------------------------------------------------
//   oldIndex
//-----------------------------------------------------------------------


inline idx_t Reordering::oldIndex ( idx_t inew ) const
{
  return rmap_[inew];
}


//-----------------------------------------------------------------------
//   newIndex
//-----------------------------------------------------------------------


inline idx_t Reordering::newIndex ( idx_t iold ) const
{
  return fmap_[iold];
}


//-----------------------------------------------------------------------
//   forwardMap
//-----------------------------------------------------------------------


inline IdxVector Reordering::forwardMap () const
{
  return fmap_;
}


//-----------------------------------------------------------------------
//   reverseMap
//-----------------------------------------------------------------------


inline IdxVector Reordering::reverseMap () const
{
  return rmap_;
}


//-----------------------------------------------------------------------
//   fmapPtr
//-----------------------------------------------------------------------


inline const idx_t* Reordering::fmapPtr () const
{
  return fmap_.addr ();
}


//-----------------------------------------------------------------------
//   rmapPtr
//-----------------------------------------------------------------------


inline const idx_t* Reordering::rmapPtr () const
{
  return rmap_.addr ();
}


JIVE_END_PACKAGE( util )

#include <jive/util/Reordering.tcc>

#endif
