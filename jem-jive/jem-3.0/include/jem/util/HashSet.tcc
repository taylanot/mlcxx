
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

#ifndef JEM_UTIL_HASHSET_TCC
#define JEM_UTIL_HASHSET_TCC

#include <utility>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class HashSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


template <class T, class A>

  inline HashSet<T,A>::HashSet ()

{}


template <class T, class A>

  inline HashSet<T,A>::HashSet ( idx_t cap ) :

    table_ ( cap )
{}


template <class T, class A>

  inline HashSet<T,A>::HashSet ( const HashSet& rhs ) :

    table_ ( rhs.table_ )

{}


template <class T, class A>

  inline HashSet<T,A>::HashSet ( HashSet&& rhs ) :

    table_ ( std::move( rhs.table_ ) )

{}


template <class T, class A>
  template <class InputIterator>

  inline HashSet<T,A>::HashSet

  ( InputIterator  first,
    InputIterator  last ) :

    table_ ( first, last )

{}


template <class T, class A>
  template <class U>

  inline HashSet<T,A>::HashSet

  ( std::initializer_list<U>  list ) :

    table_ ( list )

{}


//-----------------------------------------------------------------------
//   begin & end
//-----------------------------------------------------------------------


template <class T, class A>

  inline typename HashSet<T,A>::Iterator
  HashSet<T,A>::begin () noexcept

{
  return table_.begin ();
}


template <class T, class A>

  inline typename HashSet<T,A>::ConstIterator
  HashSet<T,A>::begin () const noexcept

{
  return table_.begin ();
}


template <class T, class A>

  inline typename HashSet<T,A>::Iterator
  HashSet<T,A>::end () noexcept

{
  return table_.end ();
}


template <class T, class A>

  inline typename HashSet<T,A>::ConstIterator
  HashSet<T,A>::end () const noexcept

{
  return table_.end ();
}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


template <class T, class A>

  inline HashSet<T,A>& HashSet<T,A>::operator =

  ( const HashSet& rhs )

{
  table_ = rhs.table_;

  return *this;
}


template <class T, class A>

  inline HashSet<T,A>& HashSet<T,A>::operator =

  ( HashSet&& rhs ) noexcept

{
  table_ = std::move ( rhs.table_ );

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T, class A>

  inline void HashSet<T,A>::swap ( HashSet& rhs ) noexcept

{
  table_.swap ( rhs.table_ );
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


template <class T, class A>

  inline bool HashSet<T,A>::contains ( const ValueType& item ) const

{
  return table_.contains ( item );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


template <class T, class A>

  inline typename HashSet<T,A>::Iterator
  HashSet<T,A>::find ( const ValueType& item )

{
  return table_.find ( item );
}


template <class T, class A>

  inline typename HashSet<T,A>::ConstIterator
  HashSet<T,A>::find ( const ValueType& item ) const

{
  return table_.find ( item );
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


template <class T, class A>

  inline typename HashSet<T,A>::Iterator
  HashSet<T,A>::insert ( const ValueType& item )

{
  return table_.insert ( item );
}


template <class T, class A>

  inline typename HashSet<T,A>::Iterator
  HashSet<T,A>::insert ( ValueType&& item )

{
  return table_.insert ( std::move( item ) );
}


template <class T, class A>

  template <class InputIterator>
  inline void HashSet<T,A>::insert

  ( InputIterator  first,
    InputIterator  last )

{
  table_.insert ( first, last );
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


template <class T, class A>

  inline void HashSet<T,A>::erase ( const ValueType& item )

{
  table_.erase ( item );
}


template <class T, class A>

  inline void HashSet<T,A>::erase ( Iterator pos )

{
  table_.erase ( pos );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


template <class T, class A>

  inline void HashSet<T,A>::clear ()

{
  table_.clear ();
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


template <class T, class A>

  inline void HashSet<T,A>::reserve ( idx_t n )

{
  table_.reserve ( n );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


template <class T, class A>

  inline void HashSet<T,A>::trimToSize ()

{
  table_.trimToSize ();
}


//-----------------------------------------------------------------------
//   capacity
//-----------------------------------------------------------------------


template <class T, class A>

  inline idx_t HashSet<T,A>::capacity () const noexcept

{
  return table_.capacity ();
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


template <class T, class A>

  inline idx_t HashSet<T,A>::size () const noexcept

{
  return table_.size ();
}


//-----------------------------------------------------------------------
//   printStats
//-----------------------------------------------------------------------


template <class T, class A>
  template <class Output>

  inline void HashSet<T,A>::printStats ( Output& out ) const

{
  table_.printStats ( out );
}


//=======================================================================
//   Related functions
//=======================================================================

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input,
          class T, class A>

  void                  decode

  ( Input&                in,
    HashSet<T,A>&         set )

{
  uidx_t  sz;

  decode ( in, sz );

  idx_t  n = (idx_t) sz;

  set.clear   ();
  set.reserve ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    T  item;

    decode ( in, item );

    set.insert ( item );
  }
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output,
          class T, class A>

  void                  encode

  ( Output&               out,
    const HashSet<T,A>&   set )

{
  typename HashSet<T,A>::ConstIterator first = set.begin ();
  typename HashSet<T,A>::ConstIterator last  = set.end   ();

  encode ( out, set.size() );

  for ( ; first != last; ++first )
  {
    encode ( out, *first );
  }
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output,
          class T, class A>

  void                  print

  ( Output&               out,
    const HashSet<T,A>&   set )

{
  if ( set.size() > 0 )
  {
    print         ( out, "[ " );
    printSequence ( out, set.begin(), set.end(), ", " );
    print         ( out, " ]" );
  }
  else
  {
    print         ( out, "[]" );
  }
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T, class A>

  inline void     swap

  ( HashSet<T,A>&   lhs,
    HashSet<T,A>&   rhs ) noexcept

{
  lhs.swap ( rhs );
}


JEM_END_PACKAGE( util )

JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class TypeTraits< util::HashSet<T,A> >
//=======================================================================


template <class T, class A>
  class TypeTraits< util::HashSet<T,A> > : public BasicTypeTraits
{
 public:

  static const bool   IS_SERIALIZABLE = IsSerializable<T>::VALUE;

  static String       whatis   ()
  {
    return TypeTraits<T>::whatis() + String ( " hash set" );
  }

  static String       typeName ()
  {
    return ( String ( "jem::util::HashSet<" ) +
             TypeTraits<T>::typeName ()       +
             String ( ',' )                   +
             TypeTraits<A>::typeName ()       +
             String ( '>' ) );
  }

};


JEM_END_PACKAGE_BASE

#endif
