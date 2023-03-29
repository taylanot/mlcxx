
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

#ifndef JEM_XML_PARSERSTACK_H
#define JEM_XML_PARSERSTACK_H

#include <utility>
#include <jem/base/Object.h>
#include <jem/util/Flex.h>


JEM_BEGIN_PACKAGE( xml )


class Parser;


//-----------------------------------------------------------------------
//   class ParserStack
//-----------------------------------------------------------------------


class ParserStack
{
 public:

  typedef ParserStack     Self;
  typedef Ref<Parser>     ParserRef;


                          ParserStack ();

  inline void             push

    ( const ParserRef&      p );

  inline void             push

    ( ParserRef&&           p );

  ParserRef               pop         ();
  inline Parser*          top         () const noexcept;

  inline Parser*          operator []

    ( idx_t                 i )          const;

  inline idx_t            size        () const noexcept;


 private:

                          ParserStack

    ( const Self&           rhs );

  Self&                   operator =

    ( const Self&           rhs );


 private:

  util::Flex<ParserRef>   stack_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   push
//-----------------------------------------------------------------------


inline void ParserStack::push ( const ParserRef& p )
{
  stack_.pushBack ( p );
}


inline void ParserStack::push ( ParserRef&& p )
{
  stack_.pushBack ( std::move( p ) );
}


//-----------------------------------------------------------------------
//   top
//-----------------------------------------------------------------------


inline Parser* ParserStack::top () const noexcept
{
  if ( stack_.size() > 0 )
  {
    return stack_.back().get();
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   operator []
//-----------------------------------------------------------------------


inline Parser* ParserStack::operator [] ( idx_t i ) const
{
  return stack_[i].get();
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t ParserStack::size () const noexcept
{
  return stack_.size ();
}


JEM_END_PACKAGE( xml )

#endif
