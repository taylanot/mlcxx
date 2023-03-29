
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


#include <jem/base/System.h>
#include <jem/io/FileName.h>
#include <jem/io/FileReader.h>
#include <jem/io/GzipFileWriter.h>


using namespace jem;
using namespace jem::io;


//-----------------------------------------------------------------------
//   gzip
//-----------------------------------------------------------------------


int gzip ( int argc, char** argv )
{
  const int    N = 128;

  Ref<Reader>  input;
  Ref<Writer>  output;
  char         buf[N];


  for ( int i = 1; i < argc; i++ )
  {
    String  fname = argv[i];

    if ( FileName::getSuffix( fname ) == ".gz" )
    {
      continue;
    }

    input  = newInstance<FileReader>     ( fname );
    output = newInstance<GzipFileWriter> ( fname + ".gz" );

    idx_t  k = input->read ( buf, N );

    while ( k > 0 )
    {
      output->write   ( buf, k );
      k = input->read ( buf, N );
    }
  }

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ( int argc, char** argv )
{
  return System::exec ( & gzip, argc, argv );
}

