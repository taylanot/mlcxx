
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
#include <jem/base/array/operators.h>
#include <jem/base/array/intrinsics.h>
#include <jem/util/Flex.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/util/DenseTable.h>
#include <jive/util/SparseTable.h>
#include <jive/model/Model.h>
#include <jive/model/Globdat.h>
#include <jive/model/Actions.h>
#include <jive/gl/Names.h>
#include <jive/gl/DataSetFactory.h>
#include <jive/gl/TableData.h>


JEM_DEFINE_CLASS( jive::gl::TableData );


JIVE_BEGIN_PACKAGE( gl )


using jem::newInstance;
using jem::IllegalInputException;


//=======================================================================
//   class TableData
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  TableData::TYPE_NAME = "Table";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


TableData::TableData

  ( const String&  name,
    const String&  tpath,
    TableType      ttype ) :

    Super  (  name ),
    ttype_ ( ttype )

{
  nullValue_ = 0.0;

  setPath_ ( tpath );
}


TableData::TableData

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props ) :

    Super ( name )

{
  Properties  myConf  = conf.makeProps ( myName_ );
  Properties  myProps = props.getProps ( myName_ );

  String      ttype   = "Dense";
  String      tpath;

  idx_t       i;


  ttype_ = DENSE_TABLE;
  i      = myName_.rfind ( '.' );

  if ( i >= 0 )
  {
    tpath = myName_[slice(i + 1,END)];
  }
  else
  {
    tpath = myName_;
  }

  if ( ! tpath.size() )
  {
    myProps.get  ( tpath, PropNames::TABLE );
  }
  else
  {
    myProps.find ( tpath, PropNames::TABLE );
  }

  myProps.find ( ttype, PropNames::TABLE_TYPE );

  if      ( ttype == "Dense" )
  {
    ttype_ = DENSE_TABLE;
  }
  else if ( ttype == "Sparse" )
  {
    ttype_ = SPARSE_TABLE;
  }
  else
  {
    myProps.propertyError (
      PropNames::TABLE_TYPE,
      String::format (
        "invalid table type: `%s\'; valid types are: "
        "`Dense\' and `Sparse\'",
        ttype
      )
    );
  }

  nullValue_ = 0.0;

  myConf.set ( PropNames::TABLE,      tpath );
  myConf.set ( PropNames::TABLE_TYPE, ttype );

  setPath_   ( tpath );
}


TableData::~TableData ()
{}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void TableData::update

  ( const Properties&  globdat,
    const AppContext&  apx )

{
  using jem::System;
  using jive::util::expandString;

  if ( ! update_( globdat, apx ) )
  {
    print ( System::warn(), getContext(),
            " : unknown table: `",
            expandString( tpath_, globdat ), "\'\n" );
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void TableData::configure

  ( const Properties&  props,
    const Properties&  globdat,
    const AppContext&  apx )

{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    String      tpath;


    myProps.find ( nullValue_, PropNames::NULL_VALUE );

    if ( myProps.find( tpath, PropNames::TABLE ) &&
         tpath != tpath_ )
    {
      try
      {
        setPath_ ( tpath );
      }
      catch ( IllegalInputException& ex )
      {
        ex.setContext ( myProps.getContext( PropNames::TABLE ) );
        throw;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void TableData::getConfig ( const Properties&  props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set ( PropNames::TABLE,      tpath_     );
  myProps.set ( PropNames::NULL_VALUE, nullValue_ );
}


//-----------------------------------------------------------------------
//   rowCount
//-----------------------------------------------------------------------


idx_t TableData::rowCount ( const AppContext& apx ) const
{
  if ( table_[apx] )
  {
    return items_[apx]->size ();
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------
//   columnCount
//-----------------------------------------------------------------------


idx_t TableData::columnCount ( const AppContext& apx ) const
{
  if ( table_[apx] )
  {
    return table_[apx]->columnCount ();
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------
//   findColumn
//-----------------------------------------------------------------------


idx_t TableData::findColumn

  ( const String&      name,
    const AppContext&  apx  ) const

{
  if ( table_[apx] )
  {
    return table_[apx]->findColumn ( name );
  }
  else
  {
    return -1;
  }
}


//-----------------------------------------------------------------------
//   getItems
//-----------------------------------------------------------------------


ItemSet* TableData::getItems ( const AppContext& apx ) const
{
  return items_[apx].get ();
}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


void TableData::getData

  ( const Matrix&      buf,
    const IdxVector&   irows,
    const IdxVector&   jcols,
    const AppContext&  apx ) const

{
  JEM_PRECHECK ( buf.size(0) == irows.size() &&
                 buf.size(1) == jcols.size() );


  buf = nullValue_;

  if ( table_[apx] )
  {
    table_[apx]->findBlock ( buf, irows, jcols );
  }
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<DataSet> TableData::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat,
    const AppContext&  apx )

{
  return newInstance<Self> ( name, conf, props );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void TableData::declare ()
{
  DataSetFactory::declare ( TYPE_NAME,  & makeNew );
  DataSetFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   setPath_
//-----------------------------------------------------------------------


void TableData::setPath_ ( const String& tpath )
{
  String  tname;

  idx_t   i = tpath.find ( '/' );


  if ( i < 0 )
  {
    tname = tpath;
  }
  else
  {
    tname = tpath[slice(i + 1,END)].stripWhite ();
  }

  if ( i == 0 || ! tname.size() )
  {
    throw IllegalInputException (
      getContext (),
      String::format ( "invalid table name: `%s\'", tpath )
    );
  }

  tpath_  = tpath;
  tname_  = tname;
  items_  = nullptr;
  table_  = nullptr;
  xtable_ = nullptr;
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


bool TableData::update_

  ( const Properties&  globdat,
    const AppContext&  apx )

{
  using jem::where;
  using jem::Limits;
  using jive::util::expandString;
  using jive::util::DenseTable;
  using jive::util::SparseTable;
  using jive::model::Model;
  using jive::model::Actions;
  using jive::model::ActionParams;

  String  tname = expandString ( tname_, globdat );


  if ( ! items_[apx] )
  {
    getItems_ ( globdat, apx );
  }

  table_[apx] = Table::find ( tname, items_[apx], globdat );

  if ( table_[apx] )
  {
    return true;
  }

  Ref<Model>  model = Model::find ( globdat );

  if ( ! model )
  {
    return false;
  }

  Ref<DofSpace>  dofs    = DofSpace::find ( items_[apx], globdat );

  Properties     params  ( "actionParams" );

  Vector         weights ( items_[apx]->size() );

  bool           updated;


  if ( ! xtable_ )
  {
    if ( ttype_ == DENSE_TABLE )
    {
      xtable_ = newInstance<DenseTable>  ( tname, items_[apx] );
    }
    else
    {
      xtable_ = newInstance<SparseTable> ( tname, items_[apx] );
    }
  }
  else
  {
    xtable_->clearAll ();
  }

  weights = 0.0;

  params.set ( ActionParams::TABLE,         xtable_ );
  params.set ( ActionParams::TABLE_NAME,    tname   );
  params.set ( ActionParams::TABLE_WEIGHTS, weights );

  updated = model->takeAction ( Actions::GET_TABLE,
                                params, globdat );

  if ( updated )
  {
    weights = where ( abs( weights ) < Limits<double>::TINY_VALUE,
                      1.0, 1.0 / weights );
    table_[apx] = xtable_;

    xtable_->scaleRows ( weights );
  }

  return updated;
}


//-----------------------------------------------------------------------
//   getItems_
//-----------------------------------------------------------------------


void TableData::getItems_

  ( const Properties&  globdat,
    const AppContext&  apx )

{
  String  context = getContext  ();

  idx_t   i       = tpath_.find ( '/' );


  items_  = nullptr;
  table_  = nullptr;
  xtable_ = nullptr;

  if ( i < 0 )
  {
    Ref<DofSpace>  dofs = DofSpace::find ( globdat );

    if ( ! dofs )
    {
      throw IllegalInputException (
        context,
        "no item set has been specified and no primary "
        "DOF space has been set"
      );
    }

    items_[apx] = dofs->getItems ();
  }
  else
  {
    String  path = tpath_[slice(BEGIN,i)].stripWhite ();

    items_[apx] = ItemSet::get ( path, globdat, context );
  }
}


JIVE_END_PACKAGE( gl )
