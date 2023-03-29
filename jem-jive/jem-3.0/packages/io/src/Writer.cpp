
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


#include <jem/base/Float.h>
#include <jem/base/Integer.h>
#include <jem/base/Character.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/io/Writer.h>


JEM_DEFINE_CUSTOM_CLASS( jem::io::Writer );


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class Writer
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


Writer::~Writer ()
{}


//-----------------------------------------------------------------------
//   makeClass
//-----------------------------------------------------------------------


Class* Writer::makeClass ()
{
  ClassTemplate<Self>*  myClass = new ClassTemplate<Self> ();

  myClass->setWhatis ( "a text output stream" );

  return myClass;
}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Writer> Writer::dup ()
{
  throw IllegalOperationException (
    JEM_FUNC,
    "text output stream can not be duplicated"
  );

  return nullptr;
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void Writer::close ()
{}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void Writer::flush ()
{}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void Writer::write ( int c, idx_t n )
{
  for ( ; n > 0; n-- )
  {
    write ( c );
  }
}


void Writer::write ( const char* buf, idx_t n )
{
  for ( idx_t i = 0; i < n; i++ )
  {
    write ( buf[i] );
  }
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void Writer::writeNoThrow ( const char* buf, idx_t n ) noexcept
{
  write ( buf, n );
}


//-----------------------------------------------------------------------
//   printByte
//-----------------------------------------------------------------------


void Writer::printByte ( byte b )
{
  const char*  digits = Character::LOWERCASE_DIGITS;

  const byte   MASK   = 15;

  char         buf[4];

  buf[0] = '0';
  buf[1] = 'x';
  buf[2] = digits[(b >> 4) & MASK];
  buf[3] = digits[b & MASK];

  write ( buf, 4 );
}


//-----------------------------------------------------------------------
//   printBool
//-----------------------------------------------------------------------


void Writer::printBool ( bool b )
{
  if ( b )
  {
    write ( "true",  4 );
  }
  else
  {
    write ( "false", 5 );
  }
}


//-----------------------------------------------------------------------
//   printInt
//-----------------------------------------------------------------------


void Writer::printInt ( lint i )
{
  char   buf[BUFSIZE_];

  idx_t  n = Integer::print ( buf, BUFSIZE_, i );

  write ( buf, n );
}


//-----------------------------------------------------------------------
//   printFloat
//-----------------------------------------------------------------------


void Writer::printFloat ( double d )
{
  char   buf[BUFSIZE_];

  idx_t  n = Float::print ( buf, BUFSIZE_, d );

  write ( buf, n );
}


//=======================================================================
//   related classes
//=======================================================================


const Flush  flush = {};


JEM_END_PACKAGE( io )
