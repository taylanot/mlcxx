
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

#ifndef JEM_XML_DOMITERATOR_H
#define JEM_XML_DOMITERATOR_H

#include <jem/base/Collectable.h>
#include <jem/util/Flex.h>


JEM_BEGIN_PACKAGE( xml )


class DOMElement;


//-----------------------------------------------------------------------
//   class DOMIterator
//-----------------------------------------------------------------------


class DOMIterator
{
 public:

  typedef DOMIterator       Self;


                            DOMIterator ()       noexcept;

  explicit                  DOMIterator

    ( Ref<DOMElement>         root );

  bool                      operator ==

    ( const Self&             rhs )        const noexcept;

  inline bool               operator !=

    ( const Self&             rhs )        const noexcept;

  void                      start

    ( Ref<DOMElement>         root );

  inline DOMElement*        get         () const noexcept;
  inline int                dir         () const noexcept;
  inline int                depth       () const noexcept;
  DOMElement*               next        ();
  DOMElement*               goUp        ();
  void                      stop        ();

  void                      setMaxDepth

    ( int                     maxDepth );


 private:

  int                       maxDepth_;

  util::Flex<DOMElement*>   stack_;
  Ref<DOMElement>           root_;
  DOMElement*               parent_;
  DOMElement*               elem_;
  int                       depth_;
  int                       dir_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   operator !=
//-----------------------------------------------------------------------


inline bool
  DOMIterator::operator != ( const Self& rhs ) const noexcept
{
  return ! operator == ( rhs );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


inline DOMElement* DOMIterator::get () const noexcept
{
  return elem_;
}


//-----------------------------------------------------------------------
//   dir
//-----------------------------------------------------------------------


inline int DOMIterator::dir () const noexcept
{
  return dir_;
}


//-----------------------------------------------------------------------
//   depth
//-----------------------------------------------------------------------


inline int DOMIterator::depth () const noexcept
{
  return depth_;
}


JEM_END_PACKAGE( xml )

#endif
