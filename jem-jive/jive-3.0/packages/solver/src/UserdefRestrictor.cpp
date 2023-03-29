
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
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/select.h>
#include <jem/base/array/utilities.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jem/util/Event.h>
#include <jem/util/HashMap.h>
#include <jive/util/utilities.h>
#include <jive/util/Table.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Constraints.h>
#include <jive/mp/Globdat.h>
#include <jive/algebra/MPMatrixObject.h>
#include <jive/solver/utilities.h>
#include <jive/solver/Names.h>
#include <jive/solver/SolverParams.h>
#include <jive/solver/RestrictorFactory.h>
#include <jive/solver/UserdefRestrictor.h>


JEM_DEFINE_CLASS( jive::solver::UserdefRestrictor );


JIVE_BEGIN_PACKAGE( solver )


using jem::IllegalInputException;


//=======================================================================
//   class UserdefRestrictor
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  UserdefRestrictor::TYPE_NAME = "Userdef";
const idx_t  UserdefRestrictor::MIN_NODES = 16;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


UserdefRestrictor::UserdefRestrictor

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat ) :

    Super ( name )

{
  using jem::dynamicCast;
  using jive::util::joinNames;
  using jive::algebra::MPMatrixObj;

  Properties  myConf  = conf .makeProps ( name );
  Properties  myProps = props.getProps  ( name );

  String      tableName;


  params.get  ( dofs_,   SolverParams::DOF_SPACE );
  params.find ( matrix_, SolverParams::MATRIX );
  params.find ( cons_,   SolverParams::CONSTRAINTS );

  myProps.get ( tableName, PropNames::TABLE );
  myConf .set ( PropNames::TABLE, tableName );

  table_ = Table::get ( tableName, dofs_->getItems(), globdat,
                        getContext() );

  minNodes_  = MIN_NODES;
  procCount_ = mp::Globdat::procCount ( globdat );

  // Unwrap the matrix if necessary.

  if ( matrix_ )
  {
    MPMatrixObj*  mpMat = dynamicCast<MPMatrixObj*> ( matrix_ );

    if ( mpMat )
    {
      matrix_ = mpMat->getInner ();
    }
  }

  connect_ ();
}


UserdefRestrictor::~UserdefRestrictor ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void UserdefRestrictor::resetEvents ()
{
  dofs_ ->resetEvents ();
  table_->resetEvents ();

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


void UserdefRestrictor::configure ( const Properties& props )
{
  using jem::maxOf;

  if ( ! props.contains( myName_ ) )
  {
    return;
  }

  Properties  myProps = props.findProps ( myName_ );

  bool        newConf = false;
  idx_t       nodeCount;


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


void UserdefRestrictor::getConfig ( const Properties& conf ) const
{
  Properties   myConf = conf.makeProps ( myName_ );

  myConf.set ( PropNames::MIN_NODES, minNodes_ );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Restrictor> UserdefRestrictor::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  using jem::newInstance;

  if ( params.contains( SolverParams::DOF_SPACE ) )
  {
    return newInstance<Self> ( name, conf, props, params, globdat );
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void UserdefRestrictor::declare ()
{
  RestrictorFactory::declare ( TYPE_NAME,  & Self::makeNew );
  RestrictorFactory::declare ( CLASS_NAME, & Self::makeNew );
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void UserdefRestrictor::connect_ ()
{
  using jem::util::connect;

  connect ( dofs_ ->newSizeEvent,   this, & Self::invalidate_ );
  connect ( table_->newValuesEvent, this, & Self::invalidate_ );
  connect ( table_->newStructEvent, this, & Self::invalidate_ );

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


void UserdefRestrictor::update_ ( int events )
{
  using jem::min;
  using jem::max;
  using jem::numeric::SparseUtils;

  ItemSet*     items     = dofs_->getItems ();
  const idx_t  itemCount = items->size     ();

  IdxVector    itemMap   ( itemCount );

  IdxMatrix    tabIndex;
  Vector       tabValues;

  idx_t        rowCount;
  idx_t        partCount;
  idx_t        basisSize;
  idx_t        tableSize;
  idx_t        nnz;


  partCount = (idx_t) (0.5 + (double) minNodes_ / (double) procCount_);
  partCount = min ( partCount, itemCount );
  partCount = max ( partCount, 1_idx );

  groupItems ( myName_, itemMap, partCount, *items,
               dofs_,   matrix_ );

  basisSize = readTable_ ( tabIndex, tabValues );
  tableSize = tabValues.size ();
  rowCount  = basisSize * partCount;

  rowOffsets_.resize  ( rowCount + 1 );

  rowOffsets_ = 0_idx;

  // Count the number of entries per row.

  for ( idx_t i = 0; i < tableSize; i++ )
  {
    idx_t  ivec = tabIndex(0,i);

    if ( ivec >= 0 )
    {
      idx_t  iitem = tabIndex(1,i);
      idx_t  irow  = ivec + basisSize * itemMap[iitem];

      rowOffsets_[irow]++;
    }
  }

  SparseUtils::sumOffsets ( rowOffsets_ );

  nnz = rowOffsets_[rowCount];

  colIndices_.resize ( nnz );
  rtValues_  .resize ( nnz );

  for ( idx_t i = 0; i < tableSize; i++ )
  {
    idx_t  ivec = tabIndex(0,i);

    if ( ivec >= 0 )
    {
      idx_t  iitem = tabIndex(1,i);
      idx_t  irow  = ivec + basisSize * itemMap[iitem];
      idx_t  ipos  = rowOffsets_[irow];

      colIndices_[ipos] = tabIndex(2,i);
      rtValues_  [ipos] = tabValues[i];
      rowOffsets_[irow] = ipos + 1;
    }
  }

  SparseUtils::shiftOffsets ( rowOffsets_ );

  shape_[0] = rowCount;
  shape_[1] = dofs_->dofCount ();
}


//-----------------------------------------------------------------------
//   invalidate_
//-----------------------------------------------------------------------


void UserdefRestrictor::invalidate_ ()
{
  shape_ = 0;

  setEvents_          ( NEW_VALUES_ | NEW_STRUCT_ );
  newStructEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   readTable_
//-----------------------------------------------------------------------


idx_t UserdefRestrictor::readTable_

  ( IdxMatrix&  tabIndex,
    Vector&     tabValues ) const

{
  using jem::max;

  SparseMatrix  tm         = table_->toMatrix    ();

  const idx_t   dofCount   = dofs_->dofCount     ();
  const idx_t   rowCount   = tm.size             (0);
  const idx_t   tableSize  = tm.nonZeroCount     ();

  IdxVector     rowOffsets = tm.getRowOffsets    ();
  IdxVector     colIndices = tm.getColumnIndices ();

  IdxMatrix     colMap     = getColumnMap_       ();

  BoolVector    dofMask    ( dofCount );


  dofMask = true;

  if ( cons_ )
  {
    dofMask[cons_->getSlaveDofs()] = false;
  }

  tabIndex .resize ( 3, tableSize );
  tabValues.ref    ( tm.getValues() );

  for ( idx_t irow = 0; irow < rowCount; irow++ )
  {
    idx_t  n = rowOffsets[irow + 1];

    for ( idx_t j = rowOffsets[irow]; j < n; j++ )
    {
      idx_t  jcol = colIndices[j];
      idx_t  idof = dofs_->findDofIndex ( irow, colMap(1,jcol) );

      if ( idof >= 0 && dofMask[idof] )
      {
        tabIndex(0,j) = colMap(0,jcol);
        tabIndex(1,j) = irow;
        tabIndex(2,j) = idof;
      }
      else
      {
        tabIndex(0,j) = -1;
        tabIndex(1,j) = -1;
        tabIndex(2,j) = -1;
      }
    }
  }

  idx_t  basisSize = 0;

  if ( colMap.size(1) > 0 )
  {
    basisSize = 1 + max ( colMap(0,ALL) );
  }

  return basisSize;
}


//-----------------------------------------------------------------------
//   getColumnMap_
//-----------------------------------------------------------------------


IdxMatrix UserdefRestrictor::getColumnMap_ () const
{
  using jem::util::HashMap;

  HashMap<String,idx_t>  vecMap;

  StringVector  colNames = table_->getColumnNames ();

  const idx_t   colCount = colNames.size ();

  IdxMatrix     colMap   ( 2, colCount );

  String        vecName;
  String        dofName;


  for ( idx_t j = 0; j < colCount; j++ )
  {
    splitColName_ ( vecName, dofName, colNames[j] );

    colMap(1,j) = dofs_->findType ( dofName );

    if ( colMap(1,j) < 0 )
    {
      throw IllegalInputException (
        getContext (),
        "unknown DOF type: " + dofName
      );
    }

    HashMap<String,idx_t>::Iterator  it =

      vecMap.find ( vecName );

    if ( it == vecMap.end() )
    {
      colMap(0,j) = vecMap.size ();

      vecMap.insert ( vecName, colMap(0,j));
    }
    else
    {
      colMap(0,j) = it->second;
    }
  }

  return colMap;
}


//-----------------------------------------------------------------------
//   splitColName_
//-----------------------------------------------------------------------


void UserdefRestrictor::splitColName_

  ( String&        vecName,
    String&        dofName,
    const String&  colName ) const

{
  idx_t  i = colName.find  ( '[' );

  vecName = "";
  dofName = "";

  if ( i >= 0 && colName.endsWith( ']' ) )
  {
    idx_t  j = colName.size() - 1;

    vecName = colName[slice(BEGIN,i)].stripWhite ();
    dofName = colName[slice(i + 1,j)].stripWhite ();
  }

  if ( vecName.size() == 0 || dofName.size() == 0 )
  {
    String  tabName = table_->getName ();
    String  details;

    if ( vecName.size() == 0 )
    {
      details = "missing vector name";
    }
    else
    {
      details = "missing DOF type name";
    }

    throw IllegalInputException (
      getContext (),
      String::format (
        "invalid column name `%s\' in table `%s\'; %s",
        colName,
        tabName,
        details
      )
    );
  }
}


JIVE_END_PACKAGE( solver )
