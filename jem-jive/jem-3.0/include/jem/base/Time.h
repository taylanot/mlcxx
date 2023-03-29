
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

#ifndef JEM_BASE_TIME_H
#define JEM_BASE_TIME_H

#include <jem/base/TypeTraits.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class Time
//-----------------------------------------------------------------------


class Time
{
 public:

  constexpr inline      Time              ()       noexcept;

  explicit
    constexpr inline    Time

    ( int                 sec )                    noexcept;

  explicit
    constexpr inline    Time

    ( long                sec )                    noexcept;

  explicit
    constexpr inline    Time

    ( lint                sec )                    noexcept;

  explicit              Time

    ( lint                sec,
      lint                nsec );

  explicit              Time

    ( double              sec );

  constexpr inline      Time

    ( const Time&         rhs )                    noexcept;

  void                  operator +=

    ( const Time&         rhs );

  void                  operator -=

    ( const Time&         rhs );

  inline void           swap

    ( Time&               rhs )                    noexcept;

  constexpr
    inline lint         sec               () const noexcept;
  constexpr
    inline lint         nsec              () const noexcept;

  inline double         toDouble          () const noexcept;

  String                toString          () const;

  String                toString

    ( const String&       format )           const;

  static constexpr
    inline Time         zero              ()       noexcept;
  static Time           now               ();
  static Time           resolution        ();


 private:

  void                  normalize_        ();

  static void           initResolution_   ();


 private:

  static const lint     NANOS_PER_SEC_;

  static lint           tick_;
  static lint           nanoTick_;

  lint                  sec_;
  lint                  nsec_;

};


//-----------------------------------------------------------------------
//   Related functions
//-----------------------------------------------------------------------


Time                    operator -

  ( const Time&           rhs );

Time                    operator +

  ( const Time&           lhs,
    const Time&           rhs );

Time                    operator -

  ( const Time&           lhs,
    const Time&           rhs );

inline bool             operator ==

  ( const Time&           lhs,
    const Time&           rhs ) noexcept;

inline bool             operator !=

  ( const Time&           lhs,
    const Time&           rhs ) noexcept;

bool                    operator <

  ( const Time&           lhs,
    const Time&           rhs ) noexcept;

inline bool             operator <=

  ( const Time&           lhs,
    const Time&           rhs ) noexcept;

inline bool             operator >

  ( const Time&           lhs,
    const Time&           rhs ) noexcept;

inline bool             operator >=

  ( const Time&           lhs,
    const Time&           rhs ) noexcept;

template <class Input>

  inline void           decode

  ( Input&                in,
    Time&                 t );

template <class Output>

  inline void           encode

  ( Output&               out,
    const Time&           t );

template <class Output>

  inline void           print

  ( Output&               out,
    const Time&           t );

inline void             swap

  ( Time&                 lhs,
    Time&                 rhs ) noexcept;


//-----------------------------------------------------------------------
//   literal operators
//-----------------------------------------------------------------------


inline namespace literals
{

inline Time             operator"" _day

  ( long double           arg );

constexpr inline Time   operator"" _day

  ( unsigned long long    arg );

inline Time             operator"" _hour

  ( long double           arg );

constexpr inline Time   operator"" _hour

  ( unsigned long long    arg );

inline Time             operator"" _min

  ( long double           arg );

constexpr inline Time   operator"" _min

  ( unsigned long long    arg );

inline Time             operator"" _sec

  ( long double           arg );

constexpr inline Time   operator"" _sec

  ( unsigned long long    arg );

inline Time             operator"" _msec

  ( long double           arg );

inline Time             operator"" _msec

  ( unsigned long long    arg );

inline Time             operator"" _usec

  ( long double           arg );

inline Time             operator"" _usec

  ( unsigned long long    arg );

inline Time             operator"" _nsec

  ( long double           arg );

inline Time             operator"" _nsec

  ( unsigned long long    arg );

}


//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Time
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


constexpr inline Time::Time () noexcept :

  sec_  ( 0_lint ),
  nsec_ ( 0_lint )

{}


constexpr inline Time::Time ( int sec ) noexcept :

  sec_  ( (lint) sec ),
  nsec_ ( 0_lint )

{}


constexpr inline Time::Time ( long sec ) noexcept :

  sec_  ( (lint) sec ),
  nsec_ ( 0_lint )

{}


constexpr inline Time::Time ( lint sec ) noexcept :

  sec_  ( sec ),
  nsec_ ( 0_lint )

{}


constexpr inline Time::Time ( const Time& rhs ) noexcept :

  sec_  ( rhs.sec_  ),
  nsec_ ( rhs.nsec_ )

{}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void Time::swap ( Time& rhs ) noexcept
{
  jem::swap ( sec_,  rhs.sec_  );
  jem::swap ( nsec_, rhs.nsec_ );
}


//-----------------------------------------------------------------------
//   sec & nsec
//-----------------------------------------------------------------------


constexpr inline lint Time::sec () const noexcept
{
  return sec_;
}


constexpr inline lint Time::nsec () const noexcept
{
  return nsec_;
}


//-----------------------------------------------------------------------
//   toDouble
//-----------------------------------------------------------------------


inline double Time::toDouble () const noexcept
{
  return ((double) sec_ + 1.0e-9 * (double) nsec_);
}


//-----------------------------------------------------------------------
//   zero
//-----------------------------------------------------------------------


constexpr inline Time Time::zero () noexcept
{
  return Time ( 0_lint );
}


//=======================================================================
//   Related functions
//=======================================================================

//-----------------------------------------------------------------------
//   comparison operators
//-----------------------------------------------------------------------


inline bool     operator ==

  ( const Time&   lhs,
    const Time&   rhs ) noexcept

{
  return (lhs.sec() == rhs.sec() && lhs.nsec() == rhs.nsec());
}


inline bool     operator !=

  ( const Time&   lhs,
    const Time&   rhs ) noexcept

{
  return !( lhs == rhs );
}


inline bool     operator <=

  ( const Time&   lhs,
    const Time&   rhs ) noexcept

{
  return !( rhs < lhs );
}


inline bool     operator >

  ( const Time&   lhs,
    const Time&   rhs ) noexcept

{
  return ( rhs < lhs );
}


inline bool     operator >=

  ( const Time&   lhs,
    const Time&   rhs ) noexcept

{
  return !( lhs < rhs );
}


//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input>

  inline void   decode

  ( Input&        in,
    Time&         t )

{
  lint  sec, nsec;

  decode ( in, sec, nsec );

  t = Time ( sec, nsec );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output>

  inline void   encode

  ( Output&       out,
    const Time&   t )

{
  encode ( out, t.sec(), t.nsec() );
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output>

  inline void   print

  ( Output&       out,
    const Time&   t )

{
  print ( out, t.toString() );
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void swap ( Time& lhs, Time& rhs ) noexcept
{
  lhs.swap ( rhs );
}


//-----------------------------------------------------------------------
//   literal operators
//-----------------------------------------------------------------------


inline namespace literals
{

inline Time             operator"" _day

  ( long double           arg )

{
  return Time ( (double) (arg * 86400.0) );
}


constexpr inline Time   operator"" _day

  ( unsigned long long    arg )

{
  return Time ( (lint) (arg * 86400ULL) );
}


inline Time             operator"" _hour

  ( long double           arg )

{
  return Time ( (double) (arg * 3600.0) );
}


constexpr inline Time   operator"" _hour

  ( unsigned long long    arg )

{
  return Time ( (lint) (arg * 3600ULL) );
}


inline Time             operator"" _min

  ( long double           arg )

{
  return Time ( (double) (arg * 60.0) );
}


constexpr inline Time   operator"" _min

  ( unsigned long long    arg )

{
  return Time ( (lint) (arg * 60ULL) );
}


inline Time             operator"" _sec

  ( long double           arg )

{
  return Time ( (double) arg );
}


constexpr inline Time   operator"" _sec

  ( unsigned long long    arg )

{
  return Time ( (lint) arg );
}


inline Time             operator"" _msec

  ( long double           arg )

{
  return Time ( (double) (arg * 1.0e-3) );
}


inline Time             operator"" _msec

  ( unsigned long long    arg )

{
  return Time ( 0, (lint) (1000000ULL * arg) );
}


inline Time             operator"" _usec

  ( long double           arg )

{
  return Time ( (double) (arg * 1.0e-6) );
}


inline Time             operator"" _usec

  ( unsigned long long    arg )

{
  return Time ( 0, (lint) (1000ULL * arg) );
}


inline Time             operator"" _nsec

  ( long double           arg )

{
  return Time ( (double) (arg * 1.0e-9) );
}


inline Time             operator"" _nsec

  ( unsigned long long    arg )

{
  return Time ( 0, (lint) arg );
}

}


//=======================================================================
//   class TypeTraits<Time>
//=======================================================================


template <>

  class TypeTraits<Time> : public BasicTypeTraits

{
 public:

  static const bool     IS_SERIALIZABLE = true;


  static String         whatis    ()
  {
    return String ( "a time" );
  }

  static String         typeName  ()
  {
    return String ( "jem::Time" );
  }

  static lint           hashValue

    ( const Time&         t )

  {
    return ( t.sec() ^ t.nsec() );
  }

};


JEM_END_PACKAGE_BASE

#endif
