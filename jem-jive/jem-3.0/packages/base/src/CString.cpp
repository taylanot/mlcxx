
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


#include <cstring>
#include <jem/base/String.h>
#include <jem/base/CString.h>


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class CString
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


CString::CString () noexcept
{
  size_      = 0;
  owner_     = false;
  buffer_[0] = '\0';
  str_       = buffer_;
}


CString::CString ( const String& str )
{
  idx_t  n = str.size ();

  size_ = n + 1;

  if ( str.hasNull() )
  {
    cstr_  = str.addr ();
    owner_ = false;
  }
  else
  {
    owner_ = true;

    if ( (n + 1) <= (idx_t) sizeof(buffer_) )
    {
      str_ = buffer_;
    }
    else
    {
      str_ = new char[n + 1];
    }

    str_[n] = '\0';

    std::memcpy ( str_, str.addr(), (size_t) n );
  }
}


//-----------------------------------------------------------------------
//   assignment operator
//-----------------------------------------------------------------------


CString& CString::operator = ( const CString& rhs )
{
  if ( this != &rhs )
  {
    free_ ();
    copy_ ( rhs );
  }

  return *this;
}


//-----------------------------------------------------------------------
//   free_
//-----------------------------------------------------------------------


void CString::free_ ()
{
  if ( owner_ && (str_ != buffer_) )
  {
    delete [] str_;
  }
}


//-----------------------------------------------------------------------
//   copy_
//-----------------------------------------------------------------------


void CString::copy_ ( const CString& rhs )
{
  size_  = rhs.size_;
  owner_ = rhs.owner_;

  if ( rhs.str_ == rhs.buffer_ )
  {
    str_ = buffer_;

    std::memcpy ( str_, rhs.str_, (size_t) size_ );
  }
  else
  {
    str_ = rhs.str_;
  }

  const_cast<CString*> ( &rhs ) -> owner_ = false;
}


JEM_END_PACKAGE_BASE
