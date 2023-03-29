
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

#ifndef JEM_UTIL_HASHTABLE_H
#define JEM_UTIL_HASHTABLE_H

#include <initializer_list>
#include <jem/base/utilities.h>
#include <jem/util/HashTableIterator.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class HashTable
//-----------------------------------------------------------------------


template <class T>

  class HashTable

{
 public:

  typedef T               Traits;
  typedef typename
    T::ValueType          ValueType;
  typedef typename
    T::KeyType            KeyType;
  typedef typename
    T::Reference          Reference;
  typedef typename
    T::ConstReference     ConstReference;
  typedef typename
    T::Pointer            Pointer;
  typedef typename
    T::ConstPointer       ConstPointer;
  typedef typename
    T::Allocator          Allocator;

  typedef HashTableNode
    <ValueType>           Node;

  typedef HashTableIterator
    < ValueType,
      Reference,
      Pointer >           Iterator;

  typedef HashTableIterator
    < ValueType,
      ConstReference,
      ConstPointer >      ConstIterator;


                          HashTable   ();

  explicit                HashTable

    ( idx_t                 cap );

                          HashTable

    ( const HashTable&      rhs );

                          HashTable

    ( HashTable&&           rhs );

  template <class InputIterator>

                          HashTable

    ( InputIterator         first,
      InputIterator         last );

  template <class U>

    inline                HashTable

    ( std::
      initializer_list<U>   list );

  inline                 ~HashTable   ();

  inline Iterator         begin       ()       noexcept;
  inline ConstIterator    begin       () const noexcept;
  inline Iterator         end         ()       noexcept;
  inline ConstIterator    end         () const noexcept;

  HashTable&              operator =

    ( const HashTable&      rhs );

  inline HashTable&       operator =

    ( HashTable&&           rhs )              noexcept;

  void                    swap

    ( HashTable&            rhs )              noexcept;

  inline bool             contains

    ( const KeyType&        key )        const;

  Iterator                find

    ( const KeyType&        key );

  ConstIterator           find

    ( const KeyType&        key )        const;

  Iterator                insert

    ( const ValueType&      item );

  Iterator                insert

    ( ValueType&&           item );

  template <class InputIterator>

    void                  insert

    ( InputIterator         first,
      InputIterator         last );

  template <class U>

    inline void           insert

    ( std::
      initializer_list<U>   list );

  void                    erase

    ( const KeyType&        key );

  void                    erase

    ( Iterator              pos );

  void                    clear ();

  void                    reserve

    ( idx_t                 cap );

  void                    trimToSize  ();
  inline idx_t            capacity    () const noexcept;
  inline idx_t            size        () const noexcept;

  template <class Output>

    void                  printStats

    ( Output&               out )        const;


 private:

  void                    init_

    ( idx_t                 cap );

  inline void             clear_      ();

  void                    reserve_

    ( idx_t                 cap );

  inline void             erase_

    ( void**                chain,
      lint                  hash,
      const KeyType&        key );

  Node*                   newNode_

    ( const ValueType&      val,
      lint                  hash );

  Node*                   newNode_

    ( ValueType&&           val,
      lint                  hash );

  inline void             free_

    ( Node*                 node );

  static inline idx_t     tableIndex_

    ( lint                  hash,
      int                   nbits )            noexcept;


 private:

  Allocator               allocator_;
  void**                  table_;
  int                     tableBits_;
  idx_t                   size_;
  idx_t                   capacity_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class Output, class T>

  void                    print

  ( Output&                 out,
    const HashTable<T>&     tbl );

template <class T>

  inline void             swap

  ( HashTable<T>&           lhs,
    HashTable<T>&           rhs ) noexcept;






//#######################################################################
//   Implementation
//#######################################################################


JEM_END_PACKAGE( util )

#include <jem/util/HashTable.tcc>

#endif
