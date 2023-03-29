
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

#ifndef JEM_XML_SOURCESTACK_H
#define JEM_XML_SOURCESTACK_H

#include <utility>
#include <jem/util/Flex.h>


JEM_BEGIN_PACKAGE( xml )


//-----------------------------------------------------------------------
//   class SourceStack
//-----------------------------------------------------------------------


class SourceStack
{
 public:

                          SourceStack ();

  inline void             push

    ( const String&         src );

  String                  pop         ();
  inline const String&    top         () const noexcept;

  inline const String&    operator []

    ( idx_t                 idx )        const;

  inline idx_t            size        () const noexcept;


 private:

                          SourceStack ( const SourceStack& );
  SourceStack&            operator =  ( const SourceStack& );


  String                  top_;
  util::Flex<String>      stack_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   push
//-----------------------------------------------------------------------


inline void SourceStack::push ( const String& s )
{
  stack_.pushBack ( s );
  top_ = s;
}


//-----------------------------------------------------------------------
//   top
//-----------------------------------------------------------------------


inline const String& SourceStack::top () const noexcept
{
  return top_;
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t SourceStack::size () const noexcept
{
  return stack_.size ();
}


//-----------------------------------------------------------------------
//   operator []
//-----------------------------------------------------------------------


inline const String& SourceStack::operator [] ( idx_t i ) const
{
  return stack_[i];
}


JEM_END_PACKAGE( xml )

#endif
