
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

#ifndef JEM_BASE_CSTRING_H
#define JEM_BASE_CSTRING_H

#include <jem/defines.h>


JEM_BEGIN_PACKAGE_BASE


class String;


//-----------------------------------------------------------------------
//   class CString
//-----------------------------------------------------------------------


class CString
{
 public:

                        CString   ()       noexcept;

  explicit              CString

    ( const String&       str );

  inline                CString

    ( const CString&      rhs );

  inline               ~CString   ();

  CString&              operator =

    ( const CString&      rhs );

  inline                operator

    const char*                   () const noexcept;

  inline const char*    addr      () const noexcept;


 private:

  void                  free_     ();

  void                  copy_

    ( const CString&      rhs );


 private:

  idx_t                 size_;
  bool                  owner_;
  char                  buffer_[32];

  union
  {
    char*               str_;
    const char*         cstr_;
  };

};


//-----------------------------------------------------------------------
//   Related functions
//-----------------------------------------------------------------------


inline CString          makeCString

  ( const String&         str );




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


inline CString::CString ( const CString& rhs )
{
  copy_ ( rhs );
}


inline CString::~CString ()
{
  free_ ();
}


//-----------------------------------------------------------------------
//   cast operator
//-----------------------------------------------------------------------


inline CString::operator const char* () const noexcept
{
  return cstr_;
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


inline const char* CString::addr () const noexcept
{
  return cstr_;
}


//-----------------------------------------------------------------------
//   makeCString
//-----------------------------------------------------------------------


inline CString makeCString ( const String& str )
{
  return CString ( str );
}


JEM_END_PACKAGE_BASE

#endif
