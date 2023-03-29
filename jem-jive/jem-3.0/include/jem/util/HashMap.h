
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

#ifndef JEM_UTIL_HASHMAP_H
#define JEM_UTIL_HASHMAP_H

#include <initializer_list>
#include <jem/base/TypeTraits.h>
#include <jem/util/HashTable.h>
#include <jem/util/HashMapTraits.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class HashMap
//-----------------------------------------------------------------------


template <class Key, class Value, class Alloc = StdAllocator>

  class HashMap

{
 public:

  typedef HashMapTraits
    <Key,Value,Alloc>     Traits;
  typedef Key             KeyType;
  typedef Value           MappedType;
  typedef typename
    Traits::ValueType     ValueType;
  typedef
    HashTable<Traits>     Table;
  typedef typename
    Table::Iterator       Iterator;
  typedef typename
    Table::ConstIterator  ConstIterator;



  inline                  HashMap     ();

  explicit inline         HashMap

    ( idx_t                 cap );

  inline                  HashMap

    ( const HashMap&        rhs );

  inline                  HashMap

    ( HashMap&&             rhs );

  template <class InputIterator>

    inline                HashMap

    ( InputIterator         first,
      InputIterator         last );

  template <class U>

    inline                HashMap

    ( std::
      initializer_list<U>   list );

  inline Iterator         begin       ()       noexcept;
  inline ConstIterator    begin       () const noexcept;
  inline Iterator         end         ()       noexcept;
  inline ConstIterator    end         () const noexcept;

  inline HashMap&         operator =

    ( const HashMap&        rhs );

  inline HashMap&         operator =

    ( HashMap&&             rhs )              noexcept;

  inline void             swap

    ( HashMap&              rhs )              noexcept;

  MappedType&             operator    []

    ( const KeyType&        key );

  inline bool             contains

    ( const KeyType&        key )        const;

  inline Iterator         find

    ( const KeyType&        key );

  inline ConstIterator    find

    ( const KeyType&        key )        const;

  inline Iterator         insert

    ( const ValueType&      value );

  inline Iterator         insert

    ( ValueType&&           value );

  inline Iterator         insert

    ( const KeyType&        key,
      const MappedType&     value );

  inline Iterator         insert

    ( KeyType&&             key,
      MappedType&&          value );

  template <class InputIterator>

    inline void           insert

    ( InputIterator         first,
      InputIterator         last );

  template <class U>

    inline void           insert

    ( std::
      initializer_list<U>   list );

  inline void             erase

    ( const KeyType&        key );

  inline void             erase

    ( Iterator              pos );

  inline void             clear       ();

  inline void             reserve

    ( idx_t                 n );

  inline void             trimToSize  ();
  inline idx_t            capacity    () const noexcept;
  inline idx_t            size        () const noexcept;

  template <class Output>

    void                  printStats

    ( Output&               out )        const;


 private:

  Table                   table_;

};


//-----------------------------------------------------------------------
//   Related functions
//-----------------------------------------------------------------------


template <class Input,
          class K, class V, class A>

  void                    decode

  ( Input&                  in,
    HashMap<K,V,A>&         map );

template <class Output,
          class K, class V, class A>

  void                    encode


  ( Output&                 out,
    const HashMap<K,V,A>&   map );

template <class Output,
          class K, class V, class A>

  void                    print

  ( Output&                 out,
    const HashMap<K,V,A>&   map );

template <class K, class V, class A>

  inline void             swap

  ( HashMap<K,V,A>&         lhs,
    HashMap<K,V,A>&         rhs ) noexcept;




//#######################################################################
//   Implementation
//#######################################################################

JEM_END_PACKAGE( util )

#include <jem/util/HashMap.tcc>

#endif
