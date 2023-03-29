
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


#include <jem/mp/UniContext.h>
#include <jive/mp/Globdat.h>


using jem::mp::UniContext;


JIVE_BEGIN_PACKAGE( mp )


//=======================================================================
//   class Globdat
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Globdat::MP_CONTEXT       = "temp.mpContext";
const char*  Globdat::RECV_BORDERS     = "recvBorders";
const char*  Globdat::SEND_BORDERS     = "sendBorders";
const char*  Globdat::ITEM_MASK        = "itemMask";
const char*  Globdat::DOFS_TRACKER     = "temp.dofsTracker";
const char*  Globdat::DATA_EXCHANGER   = "temp.dataExchanger";
const char*  Globdat::TABLE_EXCHANGER  = "temp.tableExchanger";
const char*  Globdat::VECTOR_EXCHANGER = "temp.vectorExchanger";


//-----------------------------------------------------------------------
//   myRank
//-----------------------------------------------------------------------


int Globdat::myRank ( const Properties& globdat )
{
  Ref<Context>  mpx;

  if ( globdat.find( mpx, MP_CONTEXT ) )
  {
    return mpx->myRank ();
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------
//   procCount
//-----------------------------------------------------------------------


int Globdat::procCount ( const Properties& globdat )
{
  Ref<Context>  mpx;

  if ( globdat.find( mpx, MP_CONTEXT ) )
  {
    return mpx->size ();
  }
  else
  {
    return 1;
  }
}


//-----------------------------------------------------------------------
//   getMPContext
//-----------------------------------------------------------------------


Ref<Context> Globdat::getMPContext ( const Properties& globdat )
{
  Ref<Context>  mpx;

  if ( ! globdat.find( mpx, MP_CONTEXT ) )
  {
    mpx = jem::newInstance<UniContext> ();
    globdat.set ( MP_CONTEXT, mpx );
  }

  return mpx;
}


JIVE_END_PACKAGE( mp )
