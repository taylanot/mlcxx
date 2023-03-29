
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


#include <jem/base/ClassTemplate.h>
#include <jem/base/utilities.h>
#include <jem/io/TeeWriter.h>


JEM_DEFINE_CLASS( jem::io::TeeWriter );


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class TeeWriter
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


TeeWriter::TeeWriter

  ( const Ref<Writer>&  first,
    const Ref<Writer>&  second ) :

    first_  (  first ),
    second_ ( second )

{
  JEM_PRECHECK ( first && second );
}


TeeWriter::TeeWriter ( const Self& rhs ) :

  first_  ( rhs.first_ ->dup() ),
  second_ ( rhs.second_->dup() )

{}


TeeWriter::~TeeWriter ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Writer> TeeWriter::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void TeeWriter::close ()
{
  first_ ->close ();
  second_->close ();
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void TeeWriter::flush ()
{
  first_ ->flush ();
  second_->flush ();
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void TeeWriter::write ( int c )
{
  first_ ->write ( c );
  second_->write ( c );
}


void TeeWriter::write ( int c, idx_t n )
{
  first_ ->write ( c, n );
  second_->write ( c, n );
}


void TeeWriter::write ( const char* buf, idx_t n )
{
  first_ ->write ( buf, n );
  second_->write ( buf, n );
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void TeeWriter::writeNoThrow ( const char* buf, idx_t n ) noexcept
{
  first_ ->writeNoThrow ( buf, n );
  second_->writeNoThrow ( buf, n );
}


JEM_END_PACKAGE( io )
