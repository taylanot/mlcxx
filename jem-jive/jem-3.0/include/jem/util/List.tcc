
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

#ifndef JEM_UTIL_LIST_TCC
#define JEM_UTIL_LIST_TCC

#include <utility>
#include <jem/base/assert.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class ListEntry
//=======================================================================


template <class T>

  inline ListEntry<T>::ListEntry ( const T& val ) :

    value ( val )

{}


template <class T>

  inline ListEntry<T>::ListEntry ( T&& val ) :

    value ( std::move( val ) )

{}


//=======================================================================
//   class List
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


template <class T, class A>

  inline List<T,A>::List () :

    allocator_ ( sizeof(Entry) )

{
  size_ = 0;
}


template <class T, class A>

  List<T,A>::List ( const List& rhs ) :

    allocator_ ( sizeof(Entry) )

{
  size_ = 0;

  try
  {
    insert ( end(), rhs.begin(), rhs.end() );
  }
  catch ( ... )
  {
    clear ();
    throw;
  }
}


template <class T, class A>

  List<T,A>::List ( List&& rhs ) :

    allocator_ ( std::move( rhs.allocator_ ) )

{
  size_ = rhs.size_;

  if ( rhs.size_ )
  {
    ListNode*  rhead = rhs.end_.next ();

    rhs.end_.unlink  ();
    rhead->pushFront ( &end_ );

    rhs.size_ = 0;
  }
}


template <class T, class A>
  template <class InputIterator>

  List<T,A>::List

  ( InputIterator  first,
    InputIterator  last ) :

    allocator_ ( sizeof(Entry) )

{
  size_ = 0;

  try
  {
    insert ( end(), first, last );
  }
  catch ( ... )
  {
    clear ();
    throw;
  }
}


template <class T, class A>
  template <class U>

  inline List<T,A>::List ( std::initializer_list<U> list ) :

    List ( list.begin(), list.end() )

{}


template <class T, class A>

  inline List<T,A>::~List ()

{
  clear ();
}


//-----------------------------------------------------------------------
//   begin & end
//-----------------------------------------------------------------------


template <class T, class A>

  inline typename List<T,A>::Iterator
  List<T,A>::begin () noexcept

{
  return Iterator ( end_.next() );
}


template <class T, class A>

  inline typename List<T,A>::ConstIterator
  List<T,A>::begin () const noexcept

{
  return ConstIterator ( end_.next() );
}


template <class T, class A>

  inline typename List<T,A>::Iterator
  List<T,A>::end () noexcept

{
  return Iterator ( &end_ );
}


template <class T, class A>

  inline typename List<T,A>::ConstIterator
  List<T,A>::end () const noexcept

{
  return ConstIterator ( const_cast<ListNode*>( &end_ ) );
}


//-----------------------------------------------------------------------
//   front & back
//-----------------------------------------------------------------------


template <class T, class A>

  inline T& List<T,A>::front ()

{
  JEM_ASSERT2 ( size() > 0, "empty List" );

  return static_cast<Entry*>( end_.next() )->value;
}


template <class T, class A>

  inline const T& List<T,A>::front () const

{
  JEM_ASSERT2 ( size() > 0, "empty List" );

  return static_cast<Entry*>( end_.next() )->value;
}


template <class T, class A>

  inline T& List<T,A>::back ()

{
  JEM_ASSERT2 ( size() > 0, "empty List" );

  return static_cast<Entry*>( end_.prev() )->value;
}


template <class T, class A>

  inline const T& List<T,A>::back () const

{
  JEM_ASSERT2 ( size() > 0, "empty List" );

  return static_cast<Entry*>( end_.prev() )->value;
}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


template <class T, class A>

  inline List<T,A>& List<T,A>::operator = ( const List& rhs )

{
  if ( this != &rhs )
  {
    clear  ();
    insert ( end(), rhs.begin(), rhs.end() );
  }

  return *this;
}


template <class T, class A>

  inline List<T,A>& List<T,A>::operator = ( List&& rhs ) noexcept

{
  swap ( rhs );

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T, class A>

  void List<T,A>::swap ( List& rhs ) noexcept

{
  ListNode*  rhead = nullptr;

  if ( rhs.size_ )
  {
    rhead = rhs.end_.next ();

    rhs.end_.unlink ();
  }

  if ( size_ )
  {
    ListNode*  lhead = end_.next ();

    end_.unlink ();

    lhead->pushFront ( &rhs.end_ );
  }

  if ( rhead )
  {
    rhead->pushFront ( &end_ );
  }

  allocator_.swap ( rhs.allocator_ );
  jem::swap       ( size_, rhs.size_ );
}


//-----------------------------------------------------------------------
//   pushFront
//-----------------------------------------------------------------------


template <class T, class A>

  inline void List<T,A>::pushFront ( const T& item )

{
  end_.pushBack ( newEntry_( item ) );
}


template <class T, class A>

  inline void List<T,A>::pushFront ( T&& item )

{
  end_.pushBack ( newEntry_( std::move( item ) ) );
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


template <class T, class A>

  inline void List<T,A>::pushBack ( const T& item )

{
  end_.pushFront ( newEntry_( item ) );
}


template <class T, class A>

  inline void List<T,A>::pushBack ( T&& item )

{
  end_.pushFront ( newEntry_( std::move( item ) ) );
}


//-----------------------------------------------------------------------
//   popFront
//-----------------------------------------------------------------------


template <class T, class A>

  inline void List<T,A>::popFront ()

{
  JEM_ASSERT2 ( size() > 0, "empty List" );

  free_ ( static_cast<Entry*>( end_.next() ) );
}


//-----------------------------------------------------------------------
//   popBack
//-----------------------------------------------------------------------


template <class T, class A>

  inline void List<T,A>::popBack ()

{
  JEM_ASSERT2 ( size() > 0, "empty List" );

  free_ ( static_cast<Entry*>( end_.prev() ) );
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


template <class T, class A>

  inline void List<T,A>::insert

  ( Iterator  pos,
    const T&  item )

{
  pos.getNode()->pushFront ( newEntry_( item ) );
}


template <class T, class A>

  inline void List<T,A>::insert

  ( Iterator  pos,
    T&&       item )

{
  pos.getNode()->pushFront ( newEntry_( std::move( item ) ) );
}


template <class T, class A>
  template <class InputIterator>

  void List<T,A>::insert

  ( Iterator       pos,
    InputIterator  first,
    InputIterator  last )

{
  ListNode* n = pos.getNode ();

  while ( first != last )
  {
    n->pushFront ( newEntry_( *first ) );
    ++first;
  }
}


template <class T, class A>
  template <class U>

  inline void List<T,A>::insert

    ( Iterator                  pos,
      std::initializer_list<U>  list )

{
  insert ( pos, list.begin(), list.end() );
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


template <class T, class A>

  inline void List<T,A>::erase ( Iterator pos )

{
  JEM_ASSERT2 ( pos != end(), "invalid List iterator" );

  free_ ( static_cast<Entry*>( pos.getNode() ) );
}


template <class T, class A>

  void List<T,A>::erase ( Iterator first, Iterator last )

{
  while ( first != last )
  {
    erase ( first++ );
  }
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


template <class T, class A>

  void List<T,A>::clear ()

{
  Entry* entry;

  size_ = 0;

  while ( end_.next() != &end_ )
  {
    entry = static_cast<Entry*> ( end_.next() );
    entry->~Entry ();
    allocator_.dealloc ( entry );
  }
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


template <class T, class A>

  inline idx_t List<T,A>::size () const noexcept

{
  return size_;
}


//-----------------------------------------------------------------------
//   newEntry_
//-----------------------------------------------------------------------


template <class T, class A>

  typename List<T,A>::Entry* List<T,A>::newEntry_

  ( const T& value )

{
  Entry*  entry = (Entry*) allocator_.alloc ();

  try
  {
    new (entry) Entry ( value );
  }
  catch( ... )
  {
    allocator_.dealloc ( entry );
    throw;
  }

  size_++;

  return entry;
}


template <class T, class A>

  typename List<T,A>::Entry* List<T,A>::newEntry_ ( T&& value )

{
  Entry*  entry = (Entry*) allocator_.alloc ();

  try
  {
    new (entry) Entry ( std::move( value ) );
  }
  catch( ... )
  {
    allocator_.dealloc ( entry );
    throw;
  }

  size_++;

  return entry;
}


//-----------------------------------------------------------------------
//   free_
//-----------------------------------------------------------------------


template <class T, class A>

  inline void List<T,A>::free_ ( Entry* entry )

{
  size_--;
  entry->~Entry();
  allocator_.dealloc( entry );
}


//=======================================================================
//   Related functions
//=======================================================================

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input,
          class T, class A>

  void              decode

  ( Input&            in,
    List<T,A>&        lst )

{
  uidx_t  sz;

  decode ( in, sz );

  idx_t  n = (idx_t) sz;

  lst.clear ();

  for ( idx_t i = 0; i < n; i++ )
  {
    T  item;

    decode ( in, item );

    lst.pushBack ( item );
  }
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output,
          class T, class A>

  void              encode

  ( Output&           out,
    const List<T,A>&  lst )

{
  typename List<T,A>::ConstIterator  first = lst.begin ();
  typename List<T,A>::ConstIterator  last  = lst.end   ();

  encode ( out, lst.size() );

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

  void              print

  ( Output&           out,
    const List<T,A>&  lst )

{
  if ( lst.size() > 0 )
  {
    print         ( out, "[ " );
    printSequence ( out, lst.begin(), lst.end(), ", " );
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

  inline void   swap

  ( List<T,A>&    lhs,
    List<T,A>&    rhs ) noexcept

{
  lhs.swap ( rhs );
}


JEM_END_PACKAGE( util )

JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class TypeTraits< util::List<T,A> >
//=======================================================================


template <class T, class A>

  class TypeTraits< util::List<T,A> > : public BasicTypeTraits

{
 public:

  static const bool   IS_SERIALIZABLE = IsSerializable<T>::VALUE;

  static String       whatis   ()
  {
    return TypeTraits<T>::whatis() + String ( " list" );
  }

  static String       typeName ()
  {
    return ( String ( "jem::util::List<" ) +
             TypeTraits<T>::typeName ()    +
             String ( ',' )                +
             TypeTraits<A>::typeName()     +
             String ( '>' ) );
  }

};


JEM_END_PACKAGE_BASE

#endif
