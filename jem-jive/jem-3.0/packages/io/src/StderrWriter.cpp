
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


#include <cstdio>
#include <jem/base/ClassTemplate.h>
#include <jem/io/StderrWriter.h>

#undef JEM_MUST_FLUSH

#ifdef JEM_OS_WINDOWS
#  define JEM_MUST_FLUSH
#endif


JEM_DEFINE_CLASS( jem::io::StderrWriter );


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class StderrWriter
//=======================================================================

/*
  Note that the StderrWriter class does not throw exceptions as it is
  typically used in emergency situations in which an exception will do
  more harm than good.
*/

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


StderrWriter::StderrWriter ()
{}


StderrWriter::StderrWriter ( const Self& rhs )
{}


StderrWriter::~StderrWriter ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Writer> StderrWriter::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void StderrWriter::close ()
{}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void StderrWriter::flush ()
{
  std::fflush ( stdout );
  std::fflush ( stderr );
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void StderrWriter::write ( int c )
{
  std::fflush (    stdout );
  std::putc   ( c, stderr );

#ifdef JEM_MUST_FLUSH

  std::fflush ( stderr );

#endif
}


void StderrWriter::write

  ( int    c,
    idx_t  n )

{
  std::fflush ( stdout );

  for ( idx_t i = 0; i < n; i++ )
  {
    std::putc ( c, stderr );
  }

#ifdef JEM_MUST_FLUSH

  std::fflush ( stderr );

#endif
}


void StderrWriter::write

  ( const char*  buf,
    idx_t        n )

{
  std::fflush ( stdout );

  for ( idx_t i = 0; i < n; i++ )
  {
    std::putc ( buf[i], stderr );
  }

#ifdef JEM_MUST_FLUSH

  std::fflush ( stderr );

#endif
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void StderrWriter::writeNoThrow

  ( const char*   buf,
    idx_t         n ) noexcept

{
  std::fflush ( stdout );

  for ( idx_t i = 0; i < n; i++ )
  {
    std::putc ( buf[i], stderr );
  }

#ifdef JEM_MUST_FLUSH

  std::fflush ( stderr );

#endif
}


JEM_END_PACKAGE( io )
