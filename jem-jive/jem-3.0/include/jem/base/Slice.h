
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

#ifndef JEM_BASE_SLICE_H
#define JEM_BASE_SLICE_H

#include <jem/base/limits.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class Slice
//-----------------------------------------------------------------------


class Slice
{
 public:

  static constexpr
    idx_t               MAX_INDEX = Limits<idx_t>::MAX_VALUE;


  constexpr inline      Slice       () noexcept;

  constexpr inline      Slice

    ( idx_t               first,
      idx_t               last,
      idx_t               stride = 1 ) noexcept;

  constexpr inline      Slice

    ( const Slice&        rhs )        noexcept;

  constexpr
    inline idx_t        first       () const noexcept;
  constexpr
    inline idx_t        last        () const noexcept;
  constexpr
    inline idx_t        last

    ( idx_t               ubound )     const noexcept;

  constexpr
    inline idx_t        stride      () const noexcept;


 protected:

  idx_t                 first_;
  idx_t                 last_;
  idx_t                 stride_;

};


//-----------------------------------------------------------------------
//   class SliceFromTo
//-----------------------------------------------------------------------


class SliceFromTo : public Slice
{
 public:

  constexpr inline      SliceFromTo

    ( idx_t               first,
      idx_t               last ) noexcept;

};


//-----------------------------------------------------------------------
//   class SliceTo
//-----------------------------------------------------------------------


class SliceTo : public SliceFromTo
{
 public:

  constexpr inline      SliceTo

    ( idx_t               last ) noexcept;

};


//-----------------------------------------------------------------------
//   class SliceFrom
//-----------------------------------------------------------------------


class SliceFrom : public Slice
{
 public:

  constexpr inline      SliceFrom

    ( idx_t               first ) noexcept;

};


//-----------------------------------------------------------------------
//   class SliceAll
//-----------------------------------------------------------------------


class SliceAll : public SliceFrom
{
 public:

  constexpr inline      SliceAll  () noexcept;

};


extern const SliceAll   ALL;


//-----------------------------------------------------------------------
//   class Begin
//-----------------------------------------------------------------------


class Begin {};

extern const Begin      BEGIN;


//-----------------------------------------------------------------------
//   class End
//-----------------------------------------------------------------------


class End {};

extern const End        END;


//-----------------------------------------------------------------------
//   Related functions
//-----------------------------------------------------------------------


constexpr
  inline SliceFromTo    slice

  ( idx_t                 first,
    idx_t                 last )   noexcept;

constexpr
  inline Slice          slice

  ( idx_t                 first,
    idx_t                 last,
    idx_t                 stride ) noexcept;

constexpr
  inline SliceTo        slice

  ( Begin                 first,
    idx_t                 last )   noexcept;

constexpr
  inline Slice          slice

  ( Begin                 first,
    idx_t                 last,
    idx_t                 stride ) noexcept;

constexpr
  inline SliceFrom      slice

  ( idx_t                 first,
    End                   last )   noexcept;

constexpr
  inline Slice          slice

  ( idx_t                 first,
    End                   last,
    idx_t                 stride ) noexcept;

inline const SliceAll&  slice

  ( Begin                 first,
    End                   last )   noexcept;

constexpr
  inline Slice          slice

  ( Begin                 first,
    End                   last,
    idx_t                 stride ) noexcept;

inline const SliceAll&  slice

  ( const SliceAll&       all )    noexcept;

inline Slice            slice

  ( const SliceAll&       all,
    idx_t                 stride ) noexcept;





//#######################################################################
//   implementation
//#######################################################################

//=======================================================================
//   class Slice
//=======================================================================

//-----------------------------------------------------------------------
//   contructors
//-----------------------------------------------------------------------


constexpr inline Slice::Slice () noexcept :

  first_  ( 0 ),
  last_   ( 0 ),
  stride_ ( 0 )

{}


constexpr inline Slice::Slice

( idx_t  first,
  idx_t  last,
  idx_t  stride ) noexcept :

  first_  (  first ),
  last_   (   last ),
  stride_ ( stride )

{}


constexpr inline Slice::Slice ( const Slice& rhs ) noexcept :

  first_  ( rhs.first_  ),
  last_   ( rhs.last_   ),
  stride_ ( rhs.stride_ )

{}


//-----------------------------------------------------------------------
//   first
//-----------------------------------------------------------------------


constexpr inline idx_t Slice::first () const noexcept
{
  return first_;
}


//-----------------------------------------------------------------------
//   last
//-----------------------------------------------------------------------


constexpr inline idx_t Slice::last () const noexcept
{
  return last_;
}


constexpr inline idx_t Slice::last ( idx_t ubound ) const noexcept
{
  // The comparison is made with unsigned numbers to suppress
  // warnings about signed overflow.

  return ((uidx_t) last_ > (uidx_t) ubound) ? ubound : last_;
}


//-----------------------------------------------------------------------
//   stride
//-----------------------------------------------------------------------


constexpr inline idx_t Slice::stride () const noexcept
{
  return stride_;
}


//=======================================================================
//   class SliceFromTo
//=======================================================================


constexpr inline SliceFromTo::SliceFromTo

( idx_t  first,
  idx_t  last ) noexcept :

  Slice ( first, last )

{}


//=======================================================================
//   class SliceTo
//=======================================================================


constexpr inline SliceTo::SliceTo ( idx_t last ) noexcept :

  SliceFromTo ( 0, last )

{}


//=======================================================================
//   class SliceFrom
//=======================================================================


constexpr inline SliceFrom::SliceFrom ( idx_t first ) noexcept :

  Slice ( first, Slice::MAX_INDEX )

{}


//=======================================================================
//   class SliceAll
//=======================================================================


constexpr inline SliceAll::SliceAll () noexcept :

  SliceFrom ( 0 )

{}


//=======================================================================
//   related functions
//=======================================================================


constexpr
  inline SliceFromTo    slice

  ( idx_t first, idx_t last ) noexcept

{
  return SliceFromTo ( first, last );
}


constexpr
  inline Slice          slice

  ( idx_t first, idx_t last, idx_t stride ) noexcept

{
  return Slice ( first, last, stride );
}


constexpr
  inline SliceTo        slice

  ( Begin, idx_t last ) noexcept

{
  return SliceTo ( last );
}


constexpr
  inline Slice          slice

  ( Begin, idx_t last, idx_t stride ) noexcept

{
  return Slice ( 0, last, stride );
}


constexpr
  inline SliceFrom      slice

  ( idx_t first, End ) noexcept

{
  return SliceFrom ( first );
}


constexpr
  inline Slice          slice

  ( idx_t first, End, idx_t stride ) noexcept

{
  return Slice ( first, Slice::MAX_INDEX, stride );
}


inline const SliceAll&  slice

  ( Begin, End ) noexcept

{
  return ALL;
}


constexpr
  inline Slice          slice

  ( Begin, End, idx_t stride ) noexcept

{
  return Slice ( 0, Slice::MAX_INDEX, stride );
}


inline const SliceAll&  slice

  ( const SliceAll& all ) noexcept

{
  return all;
}


inline Slice            slice

  ( const SliceAll&, idx_t stride ) noexcept

{
  return Slice ( 0, Slice::MAX_INDEX, stride );
}


JEM_END_PACKAGE_BASE

#endif
