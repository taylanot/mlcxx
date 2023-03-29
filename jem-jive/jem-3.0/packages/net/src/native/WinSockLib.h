
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

#ifndef JEM_NET_NATIVE_WINSOCKLIB_H
#define JEM_NET_NATIVE_WINSOCKLIB_H

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

// Note: do not use `extern "C" { ... }' as the header files appear
// to contain C++ declarations.

#include <winsock2.h>
#include <ws2tcpip.h>

// $##@!% Windows ...

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif


JEM_BEGIN_PACKAGE( net )


//-----------------------------------------------------------------------
//   class WinSockLib
//-----------------------------------------------------------------------


class WinSockLib
{
 public:

  static void             init   ();

};


JEM_END_PACKAGE( net )

#endif

