
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


#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/params.h>
#include <jem/io/IOException.h>
#include <jem/io/ZipOutputStream.h>
#include <jem/io/Deflator.h>


JEM_DEFINE_CLASS( jem::io::ZipOutputStream );


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class ZipOutputStream
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ZipOutputStream::ZipOutputStream

  ( Ref<OutputStream>  out,
    int                level,
    idx_t              bufsize ) :

    output_ ( out )

{
  JEM_PRECHECK ( out );

  if ( bufsize < 0 )
  {
    bufsize = DEFAULT_BUFSIZE;
  }

  deflator_ = newInstance<Deflator> ( out,   ZLIB_MODE,
                                      level, bufsize );
}


ZipOutputStream::~ZipOutputStream ()
{
  if ( deflator_ )
  {
    deflator_->finish ( NO_THROW );
    deflator_->flush  ( NO_THROW );
  }
}


//-----------------------------------------------------------------------
//   close
//-----------------------------------------------------------------------


void ZipOutputStream::close ()
{
  if ( deflator_ )
  {
    deflator_->finish ();
    deflator_->flush  ();
    output_  ->close  ();

    deflator_ = nullptr;
  }
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void ZipOutputStream::flush ()
{
  if ( deflator_ )
  {
    deflator_->sync  ();
    output_  ->flush ();
  }
}


//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------


void ZipOutputStream::write ( const void* buf, idx_t n )
{
  if ( ! deflator_ )
  {
    throw IOException ( JEM_FUNC, "output stream not open" );
  }

  deflator_->deflate ( buf, n );
}


//-----------------------------------------------------------------------
//   writeNoThrow
//-----------------------------------------------------------------------


void ZipOutputStream::writeNoThrow

  ( const void*  buf,
    idx_t        n ) noexcept

{
  if ( deflator_ )
  {
    deflator_->deflate ( buf, n, NO_THROW );
  }
}


JEM_END_PACKAGE( io )
