
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

#ifndef JEM_UTIL_HASHMAP_TCC
#define JEM_UTIL_HASHMAP_TCC

#include <utility>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class HashMap
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class K, class V, class A>

  inline HashMap<K,V,A>::HashMap ()

{}


template <class K, class V, class A>

  inline HashMap<K,V,A>::HashMap ( idx_t cap ) :

    table_ ( cap )

{}


template <class K, class V, class A>

  inline HashMap<K,V,A>::HashMap ( const HashMap& rhs ) :

    table_ ( rhs.table_ )

{}


template <class K, class V, class A>

  inline HashMap<K,V,A>::HashMap ( HashMap&& rhs ) :

    table_ ( std::move( rhs.table_ ) )

{}


template <class K, class V, class A>
  template <class InputIterator>

  inline HashMap<K,V,A>::HashMap

  ( InputIterator  first,
    InputIterator  last ) :

    table_ ( first, last )

{}


template <class K, class V, class A>
  template <class U>

  inline HashMap<K,V,A>::HashMap

  ( std::initializer_list<U>  list ) :

    table_ ( list )

{}


//-----------------------------------------------------------------------
//   begin & end
//-----------------------------------------------------------------------


template <class K, class V, class A>

  inline typename HashMap<K,V,A>::Iterator
  HashMap<K,V,A>::begin () noexcept

{
  return table_.begin ();
}


template <class K, class V, class A>

  inline typename HashMap<K,V,A>::ConstIterator
  HashMap<K,V,A>::begin () const noexcept

{
  return table_.begin ();
}


template <class K, class V, class A>

  inline typename HashMap<K,V,A>::Iterator
  HashMap<K,V,A>::end () noexcept

{
  return table_.end ();
}


template <class K, class V, class A>

  inline typename HashMap<K,V,A>::ConstIterator
  HashMap<K,V,A>::end () const noexcept

{
  return table_.end ();
}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


template <class K, class V, class A>

  inline HashMap<K,V,A>& HashMap<K,V,A>::operator =

  ( const HashMap&  rhs )

{
  table_ = rhs.table_;

  return *this;
}


template <class K, class V, class A>

  inline HashMap<K,V,A>& HashMap<K,V,A>::operator =

  ( HashMap&&  rhs ) noexcept

{
  table_ = std::move ( rhs.table_ );

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class K, class V, class A>

  inline void HashMap<K,V,A>::swap ( HashMap& rhs ) noexcept

{
  table_.swap ( rhs.table_ );
}


//-----------------------------------------------------------------------
//   operator []
//-----------------------------------------------------------------------


template <class K, class V, class A>

  typename HashMap<K,V,A>::MappedType&
  HashMap<K,V,A>::operator []

  ( const KeyType& key )

{
  Iterator  i = table_.find ( key );

  if ( i != table_.end() )
  {
    return i->second;
  }
  else
  {
    return table_.insert( ValueType( key, MappedType() ) )->second;
  }
}


//-----------------------------------------------------------------------
//   contains
//-----------------------------------------------------------------------


template <class K, class V, class A>

  inline bool HashMap<K,V,A>::contains ( const KeyType& key ) const

{
  return ( find( key ) != end() );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


template <class K, class V, class A>

  inline typename HashMap<K,V,A>::Iterator
  HashMap<K,V,A>::find ( const KeyType& key )

{
  return table_.find ( key );
}


template <class K, class V, class A>

  inline typename HashMap<K,V,A>::ConstIterator
  HashMap<K,V,A>::find ( const KeyType& key ) const

{
  return table_.find ( key );
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


template <class K, class V, class A>

  inline typename HashMap<K,V,A>::Iterator
  HashMap<K,V,A>::insert ( const ValueType& value )

{
  return table_.insert ( value );
}


template <class K, class V, class A>

  inline typename HashMap<K,V,A>::Iterator
  HashMap<K,V,A>::insert ( ValueType&& value )

{
  return table_.insert ( std::move( value ) );
}


template <class K, class V, class A>

  inline typename HashMap<K,V,A>::Iterator
  HashMap<K,V,A>::insert

  ( const KeyType&     key,
    const MappedType&  value )

{
  return table_.insert ( ValueType( key, value ) );
}


template <class K, class V, class A>

  inline typename HashMap<K,V,A>::Iterator
  HashMap<K,V,A>::insert

  ( KeyType&&     key,
    MappedType&&  value )

{
  return table_.insert ( ValueType( std::move( key ),
                                    std::move( value ) ) );
}


template <class K, class V, class A>
  template <class InputIterator>

  inline void HashMap<K,V,A>::insert

  ( InputIterator  first,
    InputIterator  last )

{
  table_.insert ( first, last );
}


template <class K, class V, class A>
  template <class U>

  inline void HashMap<K,V,A>::insert

  ( std::initializer_list<U>  list )

{
  table_.insert ( list );
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


template <class K, class V, class A>

  inline void HashMap<K,V,A>::erase ( const KeyType& key )

{
  table_.erase ( key );
}


template <class K, class V, class A>

  inline void HashMap<K,V,A>::erase ( Iterator pos )

{
  table_.erase ( pos );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


template <class K, class V, class A>

  inline void HashMap<K,V,A>::clear ()

{
  table_.clear ();
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


template <class K, class V, class A>

  inline void HashMap<K,V,A>::reserve ( idx_t n )

{
  table_.reserve ( n );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


template <class K, class V, class A>

  inline void HashMap<K,V,A>::trimToSize ()

{
  table_.trimToSize ();
}


//-----------------------------------------------------------------------
//   capacity
//-----------------------------------------------------------------------


template <class K, class V, class A>

  inline idx_t HashMap<K,V,A>::capacity () const noexcept

{
  return table_.capacity ();
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


template <class K, class V, class A>

  inline idx_t HashMap<K,V,A>::size () const noexcept

{
  return table_.size ();
}


//-----------------------------------------------------------------------
//   printStats
//-----------------------------------------------------------------------


template <class K, class V, class A>
  template <class Output>

  inline void HashMap<K,V,A>::printStats ( Output& out ) const

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
          class K, class V, class A>

  void                    decode

  ( Input&                  in,
    HashMap<K,V,A>&         map )

{
  uidx_t  sz;

  decode ( in, sz );

  idx_t  n = (idx_t) sz;

  map.clear   ();
  map.reserve ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    K  key;
    V  value;

    decode ( in, key, value );

    map.insert ( HashMap<K,V,A>::ValueType( key, value ) );
  }
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output,
          class K, class V, class A>

  void                    encode

  ( Output&                 out,
    const HashMap<K,V,A>&   map )

{
  typename HashMap<K,V,A>::ConstIterator first = map.begin ();
  typename HashMap<K,V,A>::ConstIterator last  = map.end   ();

  encode ( out, map.size() );

  for ( ; first != last; ++first )
  {
    encode ( out, first->first, first->second );
  }
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output,
          class K, class V, class A>

  void                    print

  ( Output&                 out,
    const HashMap<K,V,A>&   map )

{
  if ( map.size() > 0 )
  {
    print         ( out, "[ " );
    printSequence ( out, map.begin(), map.end(), ", " );
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


template <class K, class V, class A>

  inline void       swap

  ( HashMap<K,V,A>&   lhs,
    HashMap<K,V,A>&   rhs ) noexcept

{
  lhs.swap ( rhs );
}


JEM_END_PACKAGE( util )

JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class TypeTraits< util::HashMap<K,V,A> >
//=======================================================================


template <class K, class V, class A>

  class TypeTraits< util::HashMap<K,V,A> > : public BasicTypeTraits

{
 public:

  static const bool   IS_SERIALIZABLE =

    IsSerializable<K>::VALUE && IsSerializable<V>::VALUE;

  static String       whatis   ()
  {
    return ( TypeTraits<K>::whatis () +
             String ( " to " )        +
             TypeTraits<V>::whatis () +
             String ( " hash map" ) );
  }

  static String       typeName ()
  {
    return ( String ( "jem::util::HashMap<" ) +
             TypeTraits<K>::typeName ()       +
             String ( ',' )                   +
             TypeTraits<V>::typeName ()       +
             String ( ',' )                   +
             TypeTraits<A>::typeName()        +
             String ( '>' ) );
  }

};


JEM_END_PACKAGE_BASE

#endif
