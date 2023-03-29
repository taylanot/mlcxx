
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
#include <jem/base/Array.h>
#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jem/util/Event.h>
#include <jive/util/DofSpace.h>
#include <jive/util/PointSet.h>
#include <jive/util/Constraints.h>
#include <jive/mp/Globdat.h>
#include <jive/algebra/MPMatrixObject.h>
#include <jive/solver/utilities.h>
#include <jive/solver/Names.h>
#include <jive/solver/SolverParams.h>
#include <jive/solver/RestrictorFactory.h>
#include <jive/solver/RigidBodyRestrictor.h>


JEM_DEFINE_CLASS( jive::solver::RigidBodyRestrictor );


JIVE_BEGIN_PACKAGE( solver )


using jive::util::PointSet;


//=======================================================================
//   class RigidBodyRestrictor
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  RigidBodyRestrictor::TYPE_NAME = "RigidBody";
const idx_t  RigidBodyRestrictor::MIN_NODES = 16;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


RigidBodyRestrictor::RigidBodyRestrictor

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
  using jem::min;
  using jem::iarray;
  using jem::testany;
  using jem::dynamicCast;
  using jem::IllegalArgumentException;
  using jive::algebra::MPMatrixObj;

  JEM_PRECHECK ( dofs );

  PointSet*  points = dynamic_cast<PointSet*> ( dofs->getItems() );

  if ( ! points )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      dofs->getItems()->getContext() + " is not a PointSet"
    );
  }

  minNodes_  = MIN_NODES;
  procCount_ = mp::Globdat::procCount ( globdat );
  shape_     = 0;

  // Unwrap the matrix if necessary.

  if ( matrix )
  {
    MPMatrixObj*  mpMat = dynamicCast<MPMatrixObj*> ( matrix );

    if ( mpMat )
    {
      matrix_ = mpMat->getInner ();
    }
  }

  // Set the default DOF types.

  const char*  TYPE_NAMES[3] = { "dx", "dy", "dz" };

  idx_t  rank      = points->rank ();
  idx_t  typeCount = min ( 3_idx, dofs_->typeCount() );

  if ( rank > 0 )
  {
    typeCount = min ( typeCount, rank );
  }

  dofTypes_.resize ( typeCount );

  // Try the default DOF type names.

  for ( idx_t i = 0; i < typeCount; i++ )
  {
    dofTypes_[i] = dofs_->findType ( TYPE_NAMES[i] );
  }

  if ( testany( dofTypes_ < 0 ) )
  {
    dofTypes_ = iarray ( typeCount );
  }

  connect_ ();
}


RigidBodyRestrictor::~RigidBodyRestrictor ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void RigidBodyRestrictor::resetEvents ()
{
  ItemSet*  items = dofs_->getItems ();

  dofs_->resetEvents ();
  items->resetEvents ();

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


void RigidBodyRestrictor::configure ( const Properties& props )
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
    IdxVector  newTypes ( typeNames.size() );

    if ( typeNames.size() > 3 )
    {
      myProps.propertyError (
        PropNames::DOFS,
        String::format (
          "too many DOF types specified (%d); max is three",
          typeNames.size ()
        )
      );
    }

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

      newTypes[i] = j;
    }

    // The new DOF types are OK, so commit them.

    dofTypes_.swap ( newTypes );

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


void RigidBodyRestrictor::getConfig ( const Properties& conf ) const
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


Ref<Restrictor> RigidBodyRestrictor::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  using jem::newInstance;
  using jem::System;

  Ref<DofSpace>  dofs;

  if ( params.find( dofs, SolverParams::DOF_SPACE ) )
  {
    ItemSet*  items = dofs->getItems ();

    if ( dynamic_cast<PointSet*>( items ) )
    {
      Ref<AbstractMatrix>  matrix;
      Ref<Constraints>     cons;

      params.find ( matrix, SolverParams::MATRIX );
      params.find ( cons,   SolverParams::CONSTRAINTS );

      return newInstance<Self> ( name, dofs, globdat,
                                 matrix, cons );
    }

    print ( System::warn(), name, " : ",
            items->getContext(), " is not a PointSet\n" );
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void RigidBodyRestrictor::declare ()
{
  RestrictorFactory::declare ( TYPE_NAME,  & Self::makeNew );
  RestrictorFactory::declare ( CLASS_NAME, & Self::makeNew );
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void RigidBodyRestrictor::connect_ ()
{
  using jem::util::connect;

  ItemSet*  items = dofs_->getItems ();

  connect ( dofs_->newSizeEvent, this, & Self::invalidate_  );
  connect ( items->changeEvent,  this, & Self::invalidate_  );

  if ( matrix_ )
  {
    connect ( matrix_->newStructEvent, this, & Self::invalidate_ );
  }

  if ( cons_ )
  {
    connect ( cons_  ->newStructEvent, this, & Self::invalidate_ );
  }
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void RigidBodyRestrictor::update_ ( int events )
{
  using jem::min;
  using jem::max;
  using jem::iarray;
  using jem::testany;
  using jem::flatten;
  using jem::Error;
  using jem::System;
  using jem::numeric::SparseUtils;

  PointSet*    points    =

    dynamic_cast<PointSet*> ( dofs_->getItems() );

  const idx_t  rank      = points->rank    ();
  const idx_t  dofCount  = dofs_->dofCount ();
  const idx_t  itemCount = points->size    ();
  const idx_t  typeCount = dofTypes_.size  ();

  Matrix       coords    ( typeCount, itemCount );
  Matrix       rotmat    ( 2,         itemCount );
  IdxMatrix    dofMap    ( typeCount, itemCount );
  BoolVector   dofMask   ( dofCount );
  IdxVector    itemMap   ( itemCount );
  IdxVector    iitems    ( itemCount );
  IdxVector    idofs     ( itemCount );

  IdxVector    offsets;

  idx_t        partCount;
  idx_t        modeCount;
  idx_t        rowCount;
  idx_t        irow;
  idx_t        nnz;


  dofMask = true;

  // Get the DOF map.

  iitems =  iarray ( itemCount );
  dofMap = -1;

  dofs_->findDofIndices ( flatten( dofMap ),
                          iitems, dofTypes_ );

  if ( cons_ )
  {
    dofMask[cons_->getSlaveDofs()] = false;

    // Remove the slave DOFs from the DOF map.

    for ( idx_t j = 0; j < itemCount; j++ )
    {
      for ( idx_t i = 0; i < typeCount; i++ )
      {
        idx_t  idof = dofMap(i,j);

        if ( idof >= 0 && dofMask[idof] == false )
        {
          dofMap(i,j) = -1;
        }
      }
    }
  }

  // Create the partitions.

  partCount = (idx_t) (0.5 + (double) minNodes_ / (double) procCount_);
  partCount = min ( partCount, itemCount );
  partCount = max ( partCount, 1_idx );

  groupItems ( myName_, itemMap, partCount, *points,
               dofs_,   matrix_ );

  // Determine the number of rigid body modes per partition.

  if      ( typeCount <= 1_idx )
  {
    modeCount = typeCount;
  }
  else if ( typeCount == 2_idx )
  {
    modeCount = typeCount + 1;
  }
  else if ( typeCount == 3_idx )
  {
    modeCount = typeCount + 3;
  }
  else
  {
    throw Error (
      JEM_FUNC,
      "unexpected number of DOF types: " + String ( typeCount )
    );
  }

  // Group the items per partition.

  offsets.resize ( partCount + 1 );

  offsets = 0_idx;

  for ( idx_t iitem = 0; iitem < itemCount; iitem++ )
  {
    offsets[itemMap[iitem]]++;
  }

  SparseUtils::sumOffsets ( offsets );

  for ( idx_t iitem = 0; iitem < itemCount; iitem++ )
  {
    idx_t  ipart = itemMap[iitem];

    iitems[offsets[ipart]++] = iitem;
  }

  SparseUtils::shiftOffsets ( offsets );

  // Get the item coordinates.

  if ( typeCount == rank )
  {
    points->getCoords ( coords );
  }
  else
  {
    idx_t  n = min ( rank, typeCount );

    coords = 0.0;

    for ( idx_t i = 0; i < n; i++ )
    {
      points->getXCoords ( coords(i,ALL), i );
    }
  }

  // Estimate the number of rows and non-zeroes, and allocate memory.

  rowCount = partCount *  modeCount;
  nnz      = itemCount * (modeCount * 2 - typeCount);

  rowOffsets_.resize ( rowCount + 1 );
  colIndices_.resize ( nnz );
  rtValues_  .resize ( nnz );

  // Compute and store the rigid body modes.

  print ( System::debug( myName_ ), myName_,
          " : computing ", modeCount, " rigid body modes ...\n" );

  nnz    = 0;
  irow   = 0;
  rotmat = 0.0;

  for ( idx_t ipart = 0; ipart < partCount; ipart++ )
  {
    idx_t  pbegin = offsets[ipart];
    idx_t  pend   = offsets[ipart + 1];

    if ( pbegin == pend )
    {
      continue;
    }

    // Add the translational rigid body modes.

    for ( idx_t i = 0; i < typeCount; i++ )
    {
      rowOffsets_[irow] = nnz;

      for ( idx_t j = pbegin; j < pend; j++ )
      {
        idx_t  iitem = iitems[j];
        idx_t  idof  = dofMap(i,iitem);

        if ( idof >= 0 )
        {
          colIndices_[nnz] = idof;
          rtValues_  [nnz] = 1.0;
          nnz++;
        }
      }

      if ( nnz > rowOffsets_[irow] )
      {
        irow++;
      }
    }

    // Add the rotational rigid body modes.

    idx_t  iaxis = 0;
    idx_t  jaxis = 1;

    for ( idx_t i = typeCount; i < modeCount; i++ )
    {
      rowOffsets_[irow] = nnz;

      calcRotmat_ ( rotmat,
                    coords(iaxis,ALL),
                    coords(jaxis,ALL),
                    iitems[slice(pbegin,pend)] );

      for ( idx_t j = pbegin; j < pend; j++ )
      {
        idx_t  iitem = iitems[j];
        idx_t  idof  = dofMap(iaxis,iitem);
        idx_t  jdof  = dofMap(jaxis,iitem);

        if ( idof >= 0 )
        {
          colIndices_[nnz] = idof;
          rtValues_  [nnz] = rotmat(0,iitem);
          nnz++;
        }

        if ( jdof >= 0 )
        {
          colIndices_[nnz] = jdof;
          rtValues_  [nnz] = rotmat(1,iitem);
          nnz++;
        }
      }

      if ( nnz > rowOffsets_[irow] )
      {
        irow++;
      }

      iaxis =  jaxis;
      jaxis = (jaxis + 1) % typeCount;
    }
  }

  rowOffsets_[irow] = nnz;

  if ( irow < rowCount )
  {
    rowCount = irow;

    rowOffsets_.reshape ( rowCount + 1 );
  }

  if ( nnz < colIndices_.size() )
  {
    colIndices_.reshape ( nnz );
    rtValues_  .reshape ( nnz );
  }

  shape_[0] = rowCount;
  shape_[1] = dofCount;

  print ( System::debug( myName_ ), myName_,
          " : got ", rowCount, " rigid body vectors\n" );
}


//-----------------------------------------------------------------------
//   calcRotmat_
//-----------------------------------------------------------------------


void RigidBodyRestrictor::calcRotmat_

  ( const Matrix&     rotmat,
    const Vector&     xcoords,
    const Vector&     ycoords,
    const IdxVector&  iitems )

{
  using jem::sum;

  const idx_t  n = iitems.size ();

  // Determine the geometrical center of this partition.

  double  xmid = sum ( xcoords[iitems] );
  double  ymid = sum ( ycoords[iitems] );

  if ( n > 0 )
  {
    xmid /= (double) n;
    ymid /= (double) n;
  }

  // Compute the rotation matrix.

  for ( idx_t i = 0; i < n; i++ )
  {
    idx_t  iitem = iitems[i];

    double  dx = xcoords[iitem] - xmid;
    double  dy = ycoords[iitem] - ymid;

    rotmat(0,iitem) = -dy;
    rotmat(1,iitem) =  dx;
  }
}


//-----------------------------------------------------------------------
//   invalidate_
//-----------------------------------------------------------------------


void RigidBodyRestrictor::invalidate_ ()
{
  shape_ = 0;

  setEvents_          ( NEW_STRUCT_ | NEW_VALUES_ );
  newStructEvent.emit ( *this );
}


JIVE_END_PACKAGE( solver )
