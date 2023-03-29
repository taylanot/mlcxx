
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


#include <jem/base/assert.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/numeric/sparse/matmul.h>
#include <jem/numeric/sparse/utilities.h>
#include <jem/util/Event.h>
#include <jive/util/DofSpace.h>
#include <jive/algebra/SparseMatrixObject.h>
#include <jive/algebra/MBuilderParams.h>
#include <jive/algebra/MBuilderFactory.h>
#include <jive/fem/FEMatrixBuilder.h>


JEM_DEFINE_CLASS( jive::fem::FEMatrixBuilder );


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;


//=======================================================================
//   class FEMatrixBuilder
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  FEMatrixBuilder::TYPE_NAME = "FEM";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


FEMatrixBuilder::FEMatrixBuilder

  ( const String&         name,
    const ElementSet&     elems,
    const Ref<DofSpace>&  dofs,
    Ref<SparseMatrixObj>  matrix ) :

    Super  ( name, matrix ),
    elems_ ( elems ),
    dofs_  ( dofs )

{
  JEM_PRECHECK ( elems && dofs );

  if ( dofs->getItems() != elems.getNodes().getData() )
  {
    throw jem::IllegalArgumentException (
      getContext (),
      "elements and dof space do not share the same node set"
    );
  }

  connect_ ();
}


FEMatrixBuilder::~FEMatrixBuilder ()
{}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void FEMatrixBuilder::clear ()
{
  if ( ! updated_ )
  {
    updateStructure_ ();
  }
  else
  {
    Super::setToZero ();
  }
}


//-----------------------------------------------------------------------
//   setToZero
//-----------------------------------------------------------------------


void FEMatrixBuilder::setToZero ()
{
  if ( ! updated_ )
  {
    updateStructure_ ();
  }
  else
  {
    Super::setToZero ();
  }
}


//-----------------------------------------------------------------------
//   updateStructure
//-----------------------------------------------------------------------


void FEMatrixBuilder::updateStructure ()
{
  if ( ! updated_ )
  {
    updateStructure_ ();
  }
}


//-----------------------------------------------------------------------
//   setStructure
//-----------------------------------------------------------------------


void FEMatrixBuilder::setStructure ( const SparseStruct& st )
{
  Super::setStructure ( st );

  updated_ = false;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<algebra::MBuilder> FEMatrixBuilder::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::algebra::newSparseMatrix;
  using jive::algebra::MBuilderParams;

  Ref<DofSpace>  dofs;
  Ref<Self>      mbuilder;


  if ( params.find( dofs, MBuilderParams::DOF_SPACE ) )
  {
    ElementSet elems = ElementSet::find ( globdat );

    if ( elems &&
         dofs->getItems() == elems.getNodes().getData() )
    {
      mbuilder = newInstance<Self> (
        name,
        elems,
        dofs,
        newSparseMatrix ( name, conf, props, params )
      );
    }
  }

  return mbuilder;
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void FEMatrixBuilder::declare ()
{
  using jive::algebra::MBuilderFactory;

  MBuilderFactory::declare ( TYPE_NAME,  & makeNew );
  MBuilderFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void FEMatrixBuilder::connect_ ()
{
  using jem::util::connect;

  connect ( dofs_->newSizeEvent,  this, & Self::invalidate_ );
  connect ( dofs_->newOrderEvent, this, & Self::invalidate_ );
  connect ( elems_.changeEvent(), this, & Self::invalidate_ );

  updated_ = false;
}


//-----------------------------------------------------------------------
//   invalidate_
//-----------------------------------------------------------------------


void FEMatrixBuilder::invalidate_ ()
{
  updated_ = false;
}


//-----------------------------------------------------------------------
//   updateStructure_
//-----------------------------------------------------------------------


void FEMatrixBuilder::updateStructure_ ()
{
  using jem::System;
  using jem::numeric::matmul;
  using jem::numeric::sortRows;

  typedef ElementSet::Topology  ElemTopo;

  SparseIdxMatrix  dmat     = dofs_->toMatrix       ();
  ElemTopo         topo     = elems_.toMatrix       ();

  IdxVector     dofOffsets  = dmat.getRowOffsets    ();
  IdxVector     dofIndices  = dmat.getValues        ();
  IdxVector     topoOffsets = topo.getRowOffsets    ();
  IdxVector     topoIndices = topo.getColumnIndices ();

  const idx_t   dofCount    = dofs_->dofCount       ();
  const idx_t   elemCount   = elems_.size           ();

  SparseStruct  s;

  IdxVector     rowOffsets  ( elemCount + 1 );
  IdxVector     colIndices;

  idx_t         ielem;
  idx_t         inode;
  idx_t         i, j, k, m, n;


  print ( System::debug( myName_ ), myName_,
          " : updating matrix structure ...\n" );

  // Free memory:

  Super::setStructure ( SparseStruct() );

  rowOffsets[0] = 0;

  for ( ielem = 0; ielem < elemCount; ielem++ )
  {
    m = topoOffsets[ielem + 1];
    n = 0;

    for ( i = topoOffsets[ielem]; i < m; i++ )
    {
      inode = topoIndices[i];
      n    += (dofOffsets[inode + 1] - dofOffsets[inode]);
    }

    rowOffsets[ielem + 1] = rowOffsets[ielem] + n;
  }

  colIndices.resize ( rowOffsets[elemCount] );

  k = 0;

  for ( ielem = 0; ielem < elemCount; ielem++ )
  {
    m = topoOffsets[ielem + 1];

    for ( i = topoOffsets[ielem]; i < m; i++ )
    {
      inode = topoIndices[i];
      n     = dofOffsets[inode + 1];

      for ( j = dofOffsets[inode]; j < n; j++ )
      {
        colIndices[k++] = dofIndices[j];
      }
    }
  }

  // Free memory:

  dmat = SparseIdxMatrix ();
  topo = ElemTopo        ();

  dofOffsets .ref     ( IdxVector() );
  dofIndices .ref     ( IdxVector() );
  topoOffsets.ref     ( IdxVector() );
  topoIndices.ref     ( IdxVector() );

  s = SparseStruct    ( jem::shape ( elemCount, dofCount ),
                        rowOffsets, colIndices );

  s = matmul          ( s.transpose(), s );

  sortRows            ( s );

  Super::setStructure ( s );

  print ( System::debug( myName_ ), myName_,
          " : matrix structure updated.\n" );

  dofs_->resetEvents ();
  elems_.resetEvents ();

  updated_ = true;
}


JIVE_END_PACKAGE( fem )
