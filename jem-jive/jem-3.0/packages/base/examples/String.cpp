
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

/************************************************************************
 *
 *   This example demonstrates how to use the String class
 *
 ***********************************************************************/

#include <jem/base/System.h>
#include <jem/base/String.h>
#include <jem/base/CString.h>


using namespace jem;
using namespace jem::io;


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


int run ()
{
  Writer&  out = System::out ();

  String   s1  = "  Hello I am a string  ";
  String   s2  = s1.stripWhite ();
  String   s3  = "somebody";
  String   s4  = "a";

  idx_t    i;


  out << endl;
  out << s1 << endl << endl;
  out << s2 << endl << endl;

  out << s1[slice(2,13)] + s3.toUpper() << endl << endl;

  i = s1.rfind ( s4 );

  if ( i >= 0 )
  {
    out << s1[slice(i,END)] << endl << endl;
  }
  else
  {
    out << "substring " + s4 + " cannot be found"
        << endl << endl;
  }

  i = s1.find ( s4 );

  if ( i >= 0 )
  {
    out << s1[slice(i,END)] << endl << endl;
  }
  else
  {
    out << "substring \"" + s4 + "\" cannot be found"
        << endl << endl;
  }

  out << String::format ( "integer : %d\n"
                          "float   : %g\n"
                          "pointer : %p\n",
                          i,
                          1.2,
                          &s1 )
      << endl;

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ()
{
  return System::exec ( & run );
}
