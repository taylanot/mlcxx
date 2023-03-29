
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


#include <jem/base/limits.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/select.h>
#include <jem/base/array/utilities.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jem/util/Event.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Constraints.h>
#include <jive/mp/Globdat.h>
#include <jive/algebra/MPMatrixObject.h>
#include <jive/solver/utilities.h>
#include <jive/solver/Names.h>
#include <jive/solver/SolverParams.h>
#include <jive/solver/RestrictorFactory.h>
#include <jive/solver/SimpleRestrictor.h>


JEM_DEFINE_CLASS( jive::solver::SimpleRestrictor );


JIVE_BEGIN_PACKAGE( solver )


//=======================================================================
//   class SimpleRestrictor
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  SimpleRestrictor::TYPE_NAME = "Simple";
const idx_t  SimpleRestrictor::MIN_NODES = 16;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SimpleRestrictor::SimpleRestrictor

  ( const String&         name,
    const Ref<DofSpace>&  dofs,
    const Properties&     globdat,
    Ref<AbstractMatrix>   matrix,
    Ref<Constraints>      cons ) :

    Super   ( name ),
    dofs_   ( dofs ),
    matrix_ ( matrix ),
    cons_   ( cons )

{
  using jem::iarray;
  using jem::dynamicCast;
  using jive::algebra::MPMatrixObj;

  JEM_PRECHECK ( dofs );

  const idx_t  typeCount = dofs_->typeCount ();

  dofTypes_.resize ( typeCount );

  minNodes_  = MIN_NODES;
  procCount_ = mp::Globdat::procCount ( globdat );
  dofTypes_  = iarray ( typeCount );

  // Unwrap the matrix if necessary.

  if ( matrix )
  {
    MPMatrixObj*  mpMat = dynamicCast<MPMatrixObj*> ( matrix );

    if ( mpMat )
    {
      matrix_ = mpMat->getInner ();
    }
  }

  connect_ ();
}


SimpleRestrictor::~SimpleRestrictor ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void SimpleRestrictor::resetEvents ()
{
  dofs_->resetEvents ();

  if ( matrix_ )
  {
    matrix_->resetEvents ();
  }

  if ( cons_ )
  {
    cons_->resetEvents ();
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void SimpleRestrictor::configure ( const Properties& props )
{
  using jem::maxOf;

  if ( ! props.contains( myName_ ) )
  {
    return;
  }

  Properties    myProps = props.findProps ( myName_ );
  bool          newConf = false;

  StringVector  typeNames;
  idx_t         nodeCount;


  if ( myProps.find( typeNames, PropNames::DOFS ) )
  {
    BoolVector  mask   ( dofs_->typeCount() );
    IdxVector   itypes ( typeNames.size() );

    mask = false;

    for ( idx_t i = 0; i < typeNames.size(); i++ )
    {
      idx_t  j = dofs_->findType ( typeNames[i] );

      if ( j < 0 )
      {
        myProps.propertyError (
          PropNames::DOFS,
          "unknown DOF type: " + typeNames[i]
        );
      }

      if ( mask[j] )
      {
        myProps.propertyError (
          PropNames::DOFS,
          String::format (
            "DOF type `%s\' specified more than once",
            typeNames[i]
          )
        );
      }

      itypes[i] = j;
      mask  [j] = true;
    }

    // The new DOF types are OK, so commit them.

    dofTypes_.swap ( itypes );

    newConf = true;
  }

  if ( myProps.find( nodeCount, PropNames::MIN_NODES,
                     1, maxOf( nodeCount ) ) )
  {
    if ( nodeCount != minNodes_ )
    {
      minNodes_ = nodeCount;
      newConf   = true;
    }
  }

  if ( newConf )
  {
    invalidate_ ();
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void SimpleRestrictor::getConfig ( const Properties& conf ) const
{
  Properties    myConf    = conf.makeProps ( myName_ );

  StringVector  typeNames ( dofTypes_.size() );

  for ( idx_t i = 0; i < dofTypes_.size(); i++ )
  {
    typeNames[i] = dofs_->getTypeName ( dofTypes_[i] );
  }

  myConf.set ( PropNames::DOFS,      typeNames );
  myConf.set ( PropNames::MIN_NODES, minNodes_ );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Restrictor> SimpleRestrictor::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  Ref<DofSpace>  dofs;

  if ( params.find( dofs, SolverParams::DOF_SPACE ) )
  {
    Ref<AbstractMatrix>  matrix;
    Ref<Constraints>     cons;

    params.find ( matrix, SolverParams::MATRIX );
    params.find ( cons,   SolverParams::CONSTRAINTS );

    return jem::newInstance<Self> ( name,   dofs, globdat,
                                    matrix, cons );
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void SimpleRestrictor::declare ()
{
  RestrictorFactory::declare ( TYPE_NAME,  & Self::makeNew );
  RestrictorFactory::declare ( CLASS_NAME, & Self::makeNew );
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void SimpleRestrictor::connect_ ()
{
  using jem::util::connect;

  connect ( dofs_->newSizeEvent, this, & Self::invalidate_ );

  if ( matrix_ )
  {
    connect ( matrix_->newStructEvent, this, & Self::invalidate_ );
  }

  if ( cons_ )
  {
    connect ( cons_ ->newStructEvent, this, & Self::invalidate_ );
  }
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void SimpleRestrictor::update_ ( int events )
{
  using jem::min;
  using jem::max;
  using jem::numeric::SparseUtils;

  const ItemSet&  items     = *dofs_->getItems ();

  const idx_t     dofCount  = dofs_->dofCount  ();
  const idx_t     itemCount = dofs_->itemCount ();
  const idx_t     typeCount = dofTypes_.size   ();

  IdxVector       idofs     ( itemCount );
  IdxVector       iitems    ( itemCount );
  IdxVector       itemMap   ( itemCount );
  BoolVector      dofMask   ( dofCount  );

  idx_t           rowCount;
  idx_t           partCount;
  idx_t           i, n, nnz;


  dofMask = true;

  if ( cons_ )
  {
    dofMask[cons_->getSlaveDofs()] = false;
  }

  partCount = (idx_t) (0.5 + (double) minNodes_ / (double) procCount_);
  partCount = min ( partCount, itemCount );
  partCount = max ( partCount, 1_idx );
  rowCount  = partCount * typeCount;

  groupItems ( myName_, itemMap, partCount, items,
               dofs_,   matrix_ );

  rowOffsets_.resize  ( rowCount + 1 );

  rowOffsets_ = 0_idx;

  for ( idx_t itype = 0; itype < typeCount; itype++ )
  {
    n = dofs_->getDofsForType ( idofs, iitems,
                                dofTypes_[itype] );

    for ( i = 0; i < n; i++ )
    {
      if ( dofMask[idofs[i]] )
      {
        idx_t  irow = itype + typeCount * itemMap[iitems[i]];

        rowOffsets_[irow]++;
      }
    }
  }

  SparseUtils::sumOffsets ( rowOffsets_ );

  colIndices_.resize ( rowOffsets_[rowCount] );

  for ( idx_t itype = 0; itype < typeCount; itype++ )
  {
    n = dofs_->getDofsForType ( idofs, iitems,
                                dofTypes_[itype] );

    for ( i = 0; i < n; i++ )
    {
      idx_t  idof = idofs[i];

      if ( dofMask[idof] )
      {
        idx_t  irow = itype + typeCount * itemMap[iitems[i]];
        idx_t  ipos = rowOffsets_[irow];

        colIndices_[ipos] = idofs[i];
        rowOffsets_[irow] = ipos + 1;
      }
    }
  }

  i = nnz = 0;

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    n = rowOffsets_[irow];
    rowOffsets_[i] = nnz;

    if ( n > nnz )
    {
      nnz = n;
      i++;
    }
  }

  rowOffsets_[i] = nnz;

  if ( i < rowCount )
  {
    rowCount = i;

    rowOffsets_.reshape ( rowCount + 1 );
  }

  if ( nnz < colIndices_.size() )
  {
    colIndices_.reshape ( nnz );
  }

  rtValues_.resize ( nnz );

  rtValues_ = 1.0;

  shape_[0] = rowCount;
  shape_[1] = dofCount;
}


//-----------------------------------------------------------------------
//   invalidate_
//-----------------------------------------------------------------------


void SimpleRestrictor::invalidate_ ()
{
  shape_ = 0;

  setEvents_          ( NEW_VALUES_ | NEW_STRUCT_ );
  newStructEvent.emit ( *this );
}


JIVE_END_PACKAGE( solver )
