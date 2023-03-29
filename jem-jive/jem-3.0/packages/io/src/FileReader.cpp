
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
#include <jem/io/FileInputStream.h>
#include <jem/io/FileReader.h>


JEM_DEFINE_CLASS( jem::io::FileReader );


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class FileReader
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


FileReader::FileReader

  ( const String&  name,
    OpenFlags      flags ) :

  Super ( newInstance<FileInputStream>( name, flags ) )

{}


FileReader::FileReader ( Ref<FileStream> fs ) :

  Super ( newInstance<FileInputStream>( fs ) )

{}


FileReader::FileReader ( const Self& rhs ) :

  Super ( rhs )

{}


FileReader::~FileReader ()
{}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<Reader> FileReader::dup ()
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   getFileStream
//-----------------------------------------------------------------------


Ref<FileStream> FileReader::getFileStream () const
{
  FileInputStream*  fin = staticCast<FileInputStream*> ( input_ );

  return fin->getFileStream ();
}


JEM_END_PACKAGE( io )
