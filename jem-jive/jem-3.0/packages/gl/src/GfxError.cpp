
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


#include <jem/base/Exception.h>
#include <jem/gl/GfxError.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GfxError::Exception_
//=======================================================================


class GfxError::Exception_ : public Exception
{
 public:

  typedef Exception     Super;
  typedef Exception_    Self;


  explicit              Exception_

    ( const GfxError&     err );

                        Exception_

    ( const Self&         rhs );

  virtual String        name          () const override;
  virtual String        getStackTrace () const override;


 private:

  String                name_;
  String                trace_;

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


GfxError::Exception_::Exception_ ( const GfxError& err ) :

  Super  ( err.where_, err.what_ ),
  name_  ( err.name_  ),
  trace_ ( err.trace_ )

{}


GfxError::Exception_::Exception_ ( const Self& rhs ) :

  Super  ( rhs        ),
  name_  ( rhs.name_  ),
  trace_ ( rhs.trace_ )

{}


//-----------------------------------------------------------------------
//   name
//-----------------------------------------------------------------------


String GfxError::Exception_::name () const
{
  return name_;
}


//-----------------------------------------------------------------------
//   getStackTrace
//-----------------------------------------------------------------------


String GfxError::Exception_::getStackTrace () const
{
  return trace_;
}


//=======================================================================
//   class GfxError
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


GfxError::GfxError () :

  name_  ( "unknown" ),
  where_ ( "unknown" ),
  what_  ( "unknown" )

{}


GfxError::GfxError ( const Throwable& ex ) :

  name_  ( ex.name          () ),
  where_ ( ex.where         () ),
  what_  ( ex.what          () ),
  trace_ ( ex.getStackTrace () )

{}


GfxError::~GfxError ()
{}


//-----------------------------------------------------------------------
//   rethrow
//-----------------------------------------------------------------------


void GfxError::rethrow () const
{
  throw Exception_ ( *this );
}


JEM_END_PACKAGE( gl )
