
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


#include <jem/base/SerialSection.h>
#include <jem/net/NetworkException.h>
#include "WinSockLib.h"


#if defined(_MSC_VER)
#  pragma comment(lib,"ws2_32.lib")
#endif


JEM_BEGIN_PACKAGE( net )


//=======================================================================
//   class WinSockLib
//=======================================================================

//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void WinSockLib::init ()
{
  static bool    initialized = false;

  SerialSection  section;

  if ( ! initialized )
  {
    // Request version 2.0 of the Winsock2 DLL.

    unsigned short  vers = (unsigned short) (0 * 256 + 2);

    WSADATA         data;
    int             ierr;

    initialized = true;

    ierr = WSAStartup ( vers, &data );

    if ( ierr )
    {
      String  what =

        "failed to initialize the Windows network subsystem";

      if ( ierr == WSAVERNOTSUPPORTED )
      {
        what = what +

          " (version 2.0 of the Ws2 DLL not found)";
      }

      throw NetworkException ( JEM_FUNC, what );
    }
  }
}


JEM_END_PACKAGE( net )

