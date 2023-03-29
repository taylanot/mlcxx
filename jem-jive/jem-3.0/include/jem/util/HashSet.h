
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

#ifndef JEM_UTIL_HASHSET_H
#define JEM_UTIL_HASHSET_H

#include <initializer_list>
#include <jem/base/TypeTraits.h>
#include <jem/util/HashTable.h>
#include <jem/util/HashSetTraits.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class HashSet
//-----------------------------------------------------------------------


template <class T, class Alloc = StdAllocator>

  class HashSet

{
 public:

  typedef HashSetTraits
    <T,Alloc>             Traits;
  typedef
    HashTable<Traits>     Table;
  typedef typename
    Table::ValueType      ValueType;
  typedef typename
    Table::Iterator       Iterator;
  typedef typename
    Table::ConstIterator  ConstIterator;



  inline                  HashSet     ();

  explicit inline         HashSet

    ( idx_t                 cap );

  inline                  HashSet

    ( const HashSet&        rhs );

  inline                  HashSet

    ( HashSet&&             rhs );

  template <class InputIterator>

    inline                HashSet

    ( InputIterator         first,
      InputIterator         last );

  template <class U>

    inline                HashSet

    ( std::
      initializer_list<U>   list );

  inline Iterator         begin       ()       noexcept;
  inline ConstIterator    begin       () const noexcept;
  inline Iterator         end         ()       noexcept;
  inline ConstIterator    end         () const noexcept;

  inline HashSet&         operator =

    ( const HashSet&        rhs );

  inline HashSet&         operator =

    ( HashSet&&             rhs )              noexcept;

  inline void             swap

    ( HashSet&              rhs )              noexcept;

  inline bool             contains

    ( const ValueType&      item )       const;

  inline Iterator         find

    ( const ValueType&      item );

  inline ConstIterator    find

    ( const ValueType&      item )       const;

  inline Iterator         insert

    ( const ValueType&      item );

  inline Iterator         insert

    ( ValueType&&           item );

  template <class InputIterator>

    inline void           insert

    ( InputIterator         first,
      InputIterator         last );

  template <class U>

    inline void           insert

    ( std::
      initializer_list<U>   list );

  inline void             erase

    ( const ValueType&      item );

  inline void             erase

    ( Iterator              pos );

  inline void             clear       ();

  inline void             reserve

    ( idx_t                 n );

  inline void             trimToSize  ();
  inline idx_t            capacity    () const noexcept;
  inline idx_t            size        () const noexcept;

  template <class Output>

    inline void           printStats

    ( Output&               out )        const;


 private:

  Table                   table_;

};


//-----------------------------------------------------------------------
//   Related functions
//-----------------------------------------------------------------------


template <class Input,
          class T, class A>

  void                    decode

  ( Input&                  in,
    HashSet<T,A>&           set );

template <class Output,
          class T, class A>

  void                    encode

  ( Output&                 out,
    const HashSet<T,A>&     set );

template <class Output,
          class T, class A>

  void                    print

  ( Output&                 out,
    const HashSet<T,A>&     set );

template <class T, class A>

  inline void             swap

  ( HashSet<T,A>&           lhs,
    HashSet<T,A>&           rhs ) noexcept;





//#######################################################################
//   Implementation
//#######################################################################

JEM_END_PACKAGE( util )

#include <jem/util/HashSet.tcc>

#endif
