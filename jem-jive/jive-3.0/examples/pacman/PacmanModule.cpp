
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


#include <jem/base/System.h>
#include <jem/base/PrecheckException.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/Properties.h>
#include <jive/fem/XNodeSet.h>
#include <jive/fem/XElementSet.h>
#include <jive/fem/ElementGroup.h>

#include "PacmanModule.h"


using namespace jem;

using jive::IdxVector;
using jive::fem::toXNodeSet;
using jive::fem::toXElementSet;
using jive::fem::XNodeSet;
using jive::fem::XElementSet;
using jive::fem::ElementGroup;


//=======================================================================
//   class PacmanModule::RunData_
//=======================================================================


class PacmanModule::RunData_ : public Collectable
{
 public:

                            RunData_

    ( const String&           context,
      const ElementGroup&     egroup );

  bool                      eatNextElement  ();


 private:

  XElementSet               elems_;
  XNodeSet                  nodes_;
  IdxVector                 elemIDs_;
  idx_t                     istep_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


PacmanModule::RunData_::RunData_

  ( const String&        context,
    const ElementGroup&  egroup ) :

    elems_   ( toXElementSet ( egroup.getElements() ) ),
    nodes_   ( toXNodeSet    ( elems_.getNodes()    ) ),

    elemIDs_ ( egroup.getIDs() )

{
  if ( ! elems_ )
  {
    throw IllegalInputException (
      context,
      "element set can not be modified"
    );
  }

  if ( ! nodes_ )
  {
    throw IllegalInputException (
      context,
      "node set can not be modified"
    );
  }

  istep_ = 0;
}


//-----------------------------------------------------------------------
//   eatNextElement
//-----------------------------------------------------------------------


bool PacmanModule::RunData_::eatNextElement ()
{
  if ( istep_ >= elemIDs_.size() )
  {
    return false;
  }

  XNodeSet   nodes  = toXNodeSet ( elems_.getNodes() );

  IdxVector  count  ( nodes.size() );

  IdxVector  ielems ( 1 );
  IdxVector  inodes;

  idx_t      i, n;


  ielems[0] = elems_.findElement ( elemIDs_[istep_] );

  if ( ielems[0] < 0 )
  {
    return false;
  }

  print ( System::out(), "Erasing element ",
          elemIDs_[istep_], " ...\n" );

  elems_.eraseElements ( ielems );

  inodes.ref ( elems_.toVector() );

  count = 0;

  for ( i = 0; i < inodes.size(); i++ )
  {
    count[inodes[i]]++;
  }

  n = 0;

  for ( i = 0; i < count.size(); i++ )
  {
    if ( count[i] == 0 )
    {
      n++;
    }
  }

  if ( n )
  {
    inodes.resize ( n );

    n = 0;

    for ( i = 0; i < count.size(); i++ )
    {
      if ( count[i] == 0 )
      {
        inodes[n++] = i;
      }
    }

    print ( System::out(), "Erasing ", n, " nodes ...\n" );

    nodes.eraseNodes ( inodes );
  }

  istep_++;

  return true;
}


//=======================================================================
//   class PacmanModule
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


PacmanModule::PacmanModule ( const String& name ) :

  Super ( name )

{}


PacmanModule::~PacmanModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status PacmanModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  const String  context = getContext ();

  Properties    myConf  = conf .makeProps ( myName_ );
  Properties    myProps = props.getProps  ( myName_ );

  ElementGroup  egroup  =

    ElementGroup::get ( myConf, myProps, globdat, context );


  runData_ = newInstance<RunData_> ( context, egroup );

  return OK;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status PacmanModule::run ( const Properties& globdat )
{
  if ( runData_ && runData_->eatNextElement() )
  {
    return OK;
  }
  else
  {
    return EXIT;
  }
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void PacmanModule::shutdown ( const Properties& globdat )
{
  runData_ = nullptr;
}
