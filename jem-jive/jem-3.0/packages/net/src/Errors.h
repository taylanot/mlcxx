
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

#ifndef JEM_NET_ERRORS_H
#define JEM_NET_ERRORS_H

#include <jem/base/String.h>


JEM_BEGIN_PACKAGE( net )


//-----------------------------------------------------------------------
//   class Errors
//-----------------------------------------------------------------------


class Errors
{
 public:

  static void             invalidDomain

    ( const String&         where,
      int                   type );

  static void             unknownDomain

    ( const String&         where );

  static void             notInetDomain

    ( const String&         where );

  static void             closeFailed

    ( const String&         where );

  static void             closedSocket

    ( const String&         where );

  static void             invalidSocket

    ( const String&         where );

  static void             recvFailed

    ( const String&         where );

  static void             sendFailed

    ( const String&         where );

  static void             pollFailed

    ( const String&         where );

  static void             bindFailed

    ( const String&         where,
      const String&         addr );

  static void             acceptFailed

    ( const String&         where );

  static void             connectFailed

    ( const String&         where,
      const String&         addr );

  static void             connectTimedOut

    ( const String&         where,
      const String&         addr );

  static void             threadCancelled

    ( const String&         where );

  static void             readTimedOut

    ( const String&         where );

  static void             writeTimedOut

    ( const String&         where );

  static void             readTruncated

    ( const String&         where,
      idx_t                 count );

  static void             writeTruncated

    ( const String&         where,
      idx_t                 count );

  static void             lookupFailed

    ( const String&         where,
      const String&         addr,
      const String&         what = "" );

  static void             portLookupFailed

    ( const String&         where );

  static void             hostLookupFailed

    ( const String&         where );

  static void             newSocketFailed

    ( const String&         where );

  static void             setNoBlockFailed

    ( const String&         where );

  static void             setNoDelayFailed

    ( const String&         where );

  static void             getNoDelayFailed

    ( const String&         where );

  static void             getLocalAddrFailed

    ( const String&         where );


 private:

  static String           makeMessage_

    ( const String&         what,
      const String&         details );

};


JEM_END_PACKAGE( net )

#endif
