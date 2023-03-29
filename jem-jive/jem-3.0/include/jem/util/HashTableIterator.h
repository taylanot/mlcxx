
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

#ifndef JEM_UTIL_HASHTABLEITERATOR_H
#define JEM_UTIL_HASHTABLEITERATOR_H

#include <utility>
#include <iterator>
#include <jem/defines.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class HashTableNode
//-----------------------------------------------------------------------


template <class T>

  class HashTableNode

{
 public:

  inline            HashTableNode

    ( const T&        t,
      lint            h );

  inline            HashTableNode

    ( T&&             t,
      lint            h );


 public:

  T                 value;
  lint              hash;
  HashTableNode*    next;

};


//-----------------------------------------------------------------------
//   class HashTableIterator
//-----------------------------------------------------------------------


template <class T,
          class R = T&,
          class P = T*>

  class HashTableIterator

{
 public:

  typedef HashTableIterator          Self;
  typedef HashTableNode<T>           Node;

  typedef std::forward_iterator_tag  iterator_category;
  typedef idx_t                      difference_type;
  typedef T                          value_type;
  typedef R                          reference;
  typedef P                          pointer;


  inline            HashTableIterator () noexcept;

  explicit          HashTableIterator

    ( void**          chain )            noexcept;

  inline            HashTableIterator

    ( void**          chain,
      void*           node )             noexcept;

  inline            HashTableIterator

    ( const Self&     rhs )              noexcept;

  inline Self&      operator ++ ();
  inline Self       operator ++ (int);

  inline reference  operator *  () const;
  inline pointer    operator -> () const;

  inline bool       operator ==

    ( const Self&     rhs )        const noexcept;

  inline bool       operator !=

    ( const Self&     rhs )        const noexcept;

  inline void**     getChain ()    const noexcept;
  inline void*      getNode  ()    const noexcept;


 private:

  void              toNext_ ();


 private:

  void**            chain_;
  void*             node_;

};






//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class HashTableNode
//=======================================================================


template <class T>

  inline HashTableNode<T>::HashTableNode

  ( const T&  t,
    lint      h ) :

  value ( t ),
  hash  ( h ),
  next  ( nullptr )

{}


template <class T>

  inline HashTableNode<T>::HashTableNode

  ( T&&   t,
    lint  h ) :

  value ( std::move( t ) ),
  hash  ( h ),
  next  ( nullptr )

{}


//=======================================================================
//   class HashTableIterator
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class T, class R, class P>

  inline HashTableIterator<T,R,P>::HashTableIterator () noexcept :

  chain_ ( nullptr ),
  node_  ( nullptr )

{}


template <class T, class R, class P>

  HashTableIterator<T,R,P>::HashTableIterator

  ( void**  chain ) noexcept :

  chain_ ( chain )

{
  while ( * chain_ == nullptr )
  {
    chain_++;
  }

  node_ = * chain_;
}


template <class T, class R, class P>

  inline HashTableIterator<T,R,P>::HashTableIterator

  ( void**  chain,
    void*   node ) noexcept :

    chain_ ( chain ),
    node_  (  node )

{}


// Note: without a copy constructor, the Intel compiler may
//       generate invalid code.

template <class T, class R, class P>

  inline HashTableIterator<T,R,P>::HashTableIterator

  ( const Self& rhs ) noexcept :

    chain_ ( rhs.chain_ ),
    node_  ( rhs.node_  )

{}


//-----------------------------------------------------------------------
//   incr operators
//-----------------------------------------------------------------------


template <class T, class R, class P>

  inline HashTableIterator<T,R,P>&
  HashTableIterator<T,R,P>::operator ++ ()

{
  node_ = static_cast<Node*>(node_)->next;

  if ( node_ == nullptr )
  {
    chain_++;
    node_ = * chain_;

    if ( node_ == nullptr )
    {
      toNext_ ();
    }
  }

  return *this;
}



template <class T, class R, class P>

  inline HashTableIterator<T,R,P>
  HashTableIterator<T,R,P>::operator ++ (int)

{
  Self tmp ( *this );

  operator ++ ();
  return tmp;
}


//-----------------------------------------------------------------------
//   access operators
//-----------------------------------------------------------------------


template <class T, class R, class P>

  inline typename HashTableIterator<T,R,P>::reference
  HashTableIterator<T,R,P>::operator * () const

{
  return static_cast<Node*>(node_)->value;
}


template <class T, class R, class P>

  inline typename HashTableIterator<T,R,P>::pointer
  HashTableIterator<T,R,P>::operator -> () const

{
  return & ( static_cast<Node*>(node_)->value );
}


//-----------------------------------------------------------------------
//   (in)equality operators
//-----------------------------------------------------------------------


template <class T, class R, class P>

  inline bool HashTableIterator<T,R,P>::operator ==

  ( const Self& rhs ) const noexcept

{
  return (node_ == rhs.node_);
}


template <class T, class R, class P>

  inline bool HashTableIterator<T,R,P>::operator !=

  ( const Self& rhs ) const noexcept

{
  return (node_ != rhs.node_);
}


//-----------------------------------------------------------------------
//   getChain
//-----------------------------------------------------------------------


template <class T, class R, class P>

  inline void** HashTableIterator<T,R,P>::getChain () const noexcept

{
  return chain_;
}


//-----------------------------------------------------------------------
//   getNode
//-----------------------------------------------------------------------


template <class T, class R, class P>

  inline void* HashTableIterator<T,R,P>::getNode () const noexcept

{
  return node_;
}


//-----------------------------------------------------------------------
//   toNext_
//-----------------------------------------------------------------------


template <class T, class R, class P>

  void HashTableIterator<T,R,P>::toNext_ ()

{
  while ( node_ == nullptr )
  {
    chain_++;
    node_ = * chain_;
  }
}


JEM_END_PACKAGE( util )

#endif
