
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */

#ifndef JIVE_MP_GLOBDAT_H
#define JIVE_MP_GLOBDAT_H

#include <jive/util/Globdat.h>
#include <jive/mp/import.h>


JIVE_BEGIN_PACKAGE( mp )


//-----------------------------------------------------------------------
//   class Globdat
//-----------------------------------------------------------------------


class Globdat : public util::Globdat
{
 public:

  typedef Globdat           Self;
  typedef util::Globdat     Super;

  static const char*        MP_CONTEXT;
  static const char*        RECV_BORDERS;
  static const char*        SEND_BORDERS;
  static const char*        ITEM_MASK;
  static const char*        DOFS_TRACKER;
  static const char*        DATA_EXCHANGER;
  static const char*        TABLE_EXCHANGER;
  static const char*        VECTOR_EXCHANGER;


  static int                myRank

    ( const Properties&       globdat );

  static int                procCount

    ( const Properties&       globdat );

  static Ref<Context>       getMPContext

    ( const Properties&       globdat );

};


JIVE_END_PACKAGE( mp )

#endif
