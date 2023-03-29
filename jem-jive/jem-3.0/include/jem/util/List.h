
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

#ifndef JEM_UTIL_LIST_H
#define JEM_UTIL_LIST_H

#include <initializer_list>
#include <jem/base/TypeTraits.h>
#include <jem/util/ListIterator.h>
#include <jem/util/StdAllocator.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class ListEntry
//-----------------------------------------------------------------------


template <class T>
  class ListEntry : public ListNode
{
 public:

  typedef T               ValueType;


  explicit inline         ListEntry ( const T& val );

  explicit inline         ListEntry ( T&&      val );


 public:

  T                       value;

};


//-----------------------------------------------------------------------
//   class List
//-----------------------------------------------------------------------


template <class T, class Allocator = StdAllocator>

  class List

{
 public:

  typedef ListEntry<T>    Entry;
  typedef T               ValueType;

  typedef ListIterator
    < Entry >             Iterator;

  typedef ListIterator
    < Entry,
      const T&,
      const T* >          ConstIterator;


  inline                  List      ();

                          List

    ( const List&           rhs );

                          List

    ( List&&                rhs );

  template <class InputIterator>

                          List

    ( InputIterator         first,
      InputIterator         last );

  template <class U>

    inline                List

    ( std::
      initializer_list<U>   list );

  inline                 ~List      ();

  inline Iterator         begin     ()       noexcept;
  inline ConstIterator    begin     () const noexcept;
  inline Iterator         end       ()       noexcept;
  inline ConstIterator    end       () const noexcept;

  inline T&               front     ();
  inline const T&         front     () const;
  inline T&               back      ();
  inline const T&         back      () const;

  inline List&            operator =

    ( const List&           rhs );

  inline List&            operator =

    ( List&&                rhs )            noexcept;

  void                    swap

    ( List&                 rhs )            noexcept;

  inline void             pushFront

    ( const T&              item );

  inline void             pushFront

    ( T&&                   item );

  inline void             pushBack

    ( const T&              item );

  inline void             pushBack

    ( T&&                   item );

  inline void             popFront  ();
  inline void             popBack   ();

  inline void             insert

    ( Iterator              pos,
      const T&              item );

  inline void             insert

    ( Iterator              pos,
      T&&                   item );

  template <class InputIterator>

    void                  insert

    ( Iterator              pos,
      InputIterator         first,
      InputIterator         last );

  template <class U>

    inline void           insert

    ( Iterator              pos,
      std::
      initializer_list<U>   list );

  inline void             erase

    ( Iterator              pos );

  void                    erase

    ( Iterator              first,
      Iterator              last );

  void                    clear     ();
  inline idx_t            size      () const noexcept;


 private:

  Entry*                  newEntry_

    ( const T&              value );

  Entry*                  newEntry_

    ( T&&                   value );

  inline void             free_

    ( Entry*                entry );


 private:

  Allocator               allocator_;
  idx_t                   size_;
  ListNode                end_;

};


//-----------------------------------------------------------------------
//   Related functions
//-----------------------------------------------------------------------


template <class Input,
          class T, class A>

  void                    decode

  ( Input&                  in,
    List<T,A>&              lst );

template <class Output,
          class T, class A>

  void                    encode

  ( Output&                 out,
    const List<T,A>&        lst );

template <class Output,
          class T, class A>

  void                    print

  ( Output&                 out,
    const List<T,A>&        lst );

template <class T, class A>

  inline void             swap

  ( List<T,A>&              lhs,
    List<T,A>&              rhs ) noexcept;





//#######################################################################
//   Implementation
//#######################################################################

JEM_END_PACKAGE( util )

#include <jem/util/List.tcc>

#endif
