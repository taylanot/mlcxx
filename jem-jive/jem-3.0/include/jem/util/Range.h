
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

#ifndef JEM_UTIL_RANGE_H
#define JEM_UTIL_RANGE_H

#include <initializer_list>
#include <jem/base/TypeTraits.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class Range
//-----------------------------------------------------------------------


template <class T>

  class Range

{
 public:

  inline                  Range     ();

  inline                  Range

    ( T                     first,
      T                     last );

  template <class U>

    inline                Range

    ( std::
      initializer_list<U>   list );

  inline                  Range

    ( const Range&          rhs );

  inline bool             isEmpty   () const;
  inline bool             notEmpty  () const;

  inline Range&           operator  +=

    ( T                     rhs );

  inline Range&           operator  -=

    ( T                     rhs );

  inline Range&           operator  &=

    ( const Range&          rhs );

  inline Range&           operator  |=

    ( const Range&          rhs );


 public:

  T                       first;
  T                       last;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class Input, class T>

  inline void             decode

  ( Input&                  in,
    Range<T>&               rn );

template <class Output, class T>

  inline void             encode

  ( Output&                 out,
    const Range<T>&         rn );

template <class Output, class T>

  inline void             print

  ( Output&                 out,
    const Range<T>&         rn );

template <class T>

  inline Range<T>         operator +

  ( const Range<T>&         lhs,
    T                       rhs );

template <class T>

  inline Range<T>         operator -

  ( const Range<T>&         lhs,
    T                       rhs );

template <class T>

  inline Range<T>         operator &

  ( const Range<T>&         lhs,
    const Range<T>&         rhs );

template <class T>

  inline Range<T>         operator |

  ( const Range<T>&         lhs,
    const Range<T>&         rhs );

template <class T>

  inline bool             overlap

  ( const Range<T>&         lhs,
    const Range<T>&         rhs );

template <class T>

  inline Range<T>         range

  ( T                       first,
    T                       last );






//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Range
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class T>

  inline Range<T>::Range () :

    first ( T() ),
    last  ( T() )

{}


template <class T>

  inline Range<T>::Range ( T a, T b ) :

    first ( a ),
    last  ( b )

{}


template <class T>
  template <class U>

  inline Range<T>::Range ( std::initializer_list<U> list )

{
  JEM_ASSERT2 ( list.size() == 2,
                "invalid initializer list size" );

  auto it = list.begin ();

  first = *it; it++;
  last  = *it;
}


template <class T>

  inline Range<T>::Range ( const Range& rhs ) :

    first ( rhs.first ),
    last  ( rhs.last  )

{}


//-----------------------------------------------------------------------
//   isEmpty
//-----------------------------------------------------------------------


template <class T>

  inline bool Range<T>::isEmpty () const

{
  return ! ( first < last );
}


//-----------------------------------------------------------------------
//   notEmpty
//-----------------------------------------------------------------------


template <class T>

  inline bool Range<T>::notEmpty () const

{
  return ( first < last );
}


//-----------------------------------------------------------------------
//   computed assignment operators
//-----------------------------------------------------------------------


template <class T>

  inline Range<T>& Range<T>::operator += ( T rhs )

{
  first += rhs;
  last  += rhs;

  return *this;
}


template <class T>

  inline Range<T>& Range<T>::operator -= ( T rhs )

{
  first -= rhs;
  last  -= rhs;

  return *this;
}


template <class T>

  inline Range<T>& Range<T>::operator &= ( const Range& rhs )

{
  first = jem::max ( first, rhs.first );
  last  = jem::min ( last,  rhs.last  );

  return *this;
}


template <class T>

  inline Range<T>& Range<T>::operator |= ( const Range& rhs )

{
  if ( rhs.notEmpty() )
  {
    if ( notEmpty() )
    {
      first = jem::min ( first, rhs.first );
      last  = jem::max ( last,  rhs.last  );
    }
    else
    {
      first = rhs.first;
      last  = rhs.last;
    }

  }

  return *this;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input, class T>

  inline void       decode

  ( Input&            in,
    Range<T>&         rn )

{
  decode ( in, rn.first, rn.last );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output, class T>

  inline void       encode

  ( Output&           out,
    const Range<T>&   rn )

{
  encode ( out, rn.first, rn.last );
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output, class T>

  inline void       print

  ( Output&           out,
    const Range<T>&   rn )

{
  print ( out, '[', rn.first, ',', rn.last, ')' );
}


//-----------------------------------------------------------------------
//   binary operators
//-----------------------------------------------------------------------


template <class T>

  inline Range<T>   operator +

  ( const Range<T>&   lhs,
    T                 rhs )

{
  return Range<T> ( lhs.first + rhs, lhs.last + rhs );
}


template <class T>

  inline Range<T>   operator -

  ( const Range<T>&   lhs,
    T                 rhs )

{
  return Range<T> ( lhs.first - rhs, lhs.last - rhs );
}


template <class T>

  inline Range<T>   operator &

  ( const Range<T>&   lhs,
    const Range<T>&   rhs )

{
  return Range<T> ( jem::max( lhs.first, rhs.first ),
                    jem::min( lhs.last,  rhs.last  ) );
}


template <class T>

  inline Range<T>   operator |

  ( const Range<T>&   lhs,
    const Range<T>&   rhs )

{
  if      ( lhs.isEmpty() )
  {
    return rhs;
  }
  else if ( rhs.isEmpty() )
  {
    return lhs;
  }
  else
  {
    return Range<T> ( jem::min ( lhs.first, rhs.first ),
                      jem::max ( lhs.last,  rhs.last  ) );
  }
}


//-----------------------------------------------------------------------
//   overlap
//-----------------------------------------------------------------------


template <class T>

  inline bool       overlap

  ( const Range<T>&   lhs,
    const Range<T>&   rhs )

{
  if ( lhs.first < rhs.first )
  {
    return ( rhs.first < lhs.last && rhs.first < rhs.last );
  }
  else
  {
    return ( lhs.first < rhs.last && lhs.first < lhs.last );
  }
}


//-----------------------------------------------------------------------
//   range
//-----------------------------------------------------------------------


template <class T>

  inline Range<T>   range

  ( T                 first,
    T                 last )

{
  return Range<T> ( first, last );
}


JEM_END_PACKAGE( util )

JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class TypeTraits< util::Range<T> >
//=======================================================================


template <class T>

  class TypeTraits< util::Range<T> > : public BasicTypeTraits

{
 public:

  static const bool   IS_SERIALIZABLE = IsSerializable<T>::VALUE;


  static String       whatis   ()
  {
    return TypeTraits<T>::typeName() + String ( " range" );
  }

  static String       typeName ()
  {
    return ( String ( "jem::util::Range<" ) +
             TypeTraits<T>::typeName ()     +
             String ( '>' ) );
  }

};


JEM_END_PACKAGE_BASE

#endif
