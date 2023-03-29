
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


#include <jem/base/rtti.h>
#include <jem/base/Float.h>
#include <jem/base/ParseException.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/utilities.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/intrinsics.h>
#include <jem/util/HashMap.h>
#include <jem/util/Properties.h>
#include <jem/util/FastAllocator.h>
#include <jem/numeric/func/UserFunc.h>
#include <jive/util/FuncUtils.h>
#include <jive/gl/DataSource.h>


JIVE_BEGIN_PACKAGE( gl )


using jem::Float;
using jem::IllegalInputException;
using jem::util::HashMap;
using jem::util::FastAllocator;
using jem::numeric::UserFunc;


//=======================================================================
//   class DataSource::ArgList_
//=======================================================================


class DataSource::ArgList_ : public UserFunc::ArgList
{
 public:

  typedef HashMap
    < String,
      idx_t,
      FastAllocator >       ArgMap;


  inline                    ArgList_    ();

  virtual idx_t             argCount    () const override;

  virtual idx_t             getArgIndex

    ( const String&           arg )              override;

  StringVector              getArgList  () const;


 private:

  ArgMap                    args_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline DataSource::ArgList_::ArgList_ ()
{}


//-----------------------------------------------------------------------
//   argCount
//-----------------------------------------------------------------------


idx_t DataSource::ArgList_::argCount () const
{
  return args_.size ();
}


//-----------------------------------------------------------------------
//   getArgIndex
//-----------------------------------------------------------------------


idx_t DataSource::ArgList_::getArgIndex ( const String& arg )
{
  using jem::util::makePair;

  const idx_t  n = arg.size ();

  String       name;
  idx_t        index;


  if ( n < 2 )
  {
    return -1;
  }

  if ( arg[0] != '$' )
  {
    return -1;
  }

  if ( arg[1] == '(' || arg[1] == '{' || arg[1] == '[' )
  {
    name = arg[slice(2,n - 1)].stripWhite ();
  }
  else
  {
    name = arg[slice(1,END)];
  }

  ArgMap::Iterator  it = args_.find ( name );

  if ( it == args_.end() )
  {
    index = args_.size ();

    args_.insert ( makePair( name, index ) );
  }
  else
  {
    index = it->second;
  }

  return index;
}


//-----------------------------------------------------------------------
//   getArgList
//-----------------------------------------------------------------------


StringVector DataSource::ArgList_::getArgList () const
{
  ArgMap::ConstIterator  it  = args_.begin ();
  ArgMap::ConstIterator  end = args_.end   ();

  StringVector  names ( args_.size() );


  for ( ; it != end; ++it )
  {
    names[it->second] = it->first;
  }

  return names;
}


//=======================================================================
//   class DataSource
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


DataSource::DataSource ( double val ) : value_ ( val )
{}


DataSource::DataSource ( const DataSource& rhs ) :

  dataSet_  ( rhs.dataSet_ ),
  colFunc_  ( rhs.colFunc_ ),
  setName_  ( rhs.setName_ ),
  colNames_ ( rhs.colNames_.clone() ),
  value_    ( rhs.value_   )

{}


//-----------------------------------------------------------------------
//   assignment operator
//-----------------------------------------------------------------------


DataSource& DataSource::operator = ( const DataSource& rhs )
{
  dataSet_ = rhs.dataSet_;
  colFunc_ = rhs.colFunc_;
  setName_ = rhs.setName_;
  value_   = rhs.value_;

  if ( colNames_.size() != rhs.colNames_.size() )
  {
    colNames_.resize ( rhs.colNames_.size() );
  }

  colNames_ = rhs.colNames_;

  return *this;
}


//-----------------------------------------------------------------------
//   init (value)
//-----------------------------------------------------------------------


void DataSource::init ( double value )
{
  dataSet_ = nullptr;
  colFunc_ = nullptr;
  setName_ = "";
  value_   = value;

  colNames_.resize ( 0 );
}


//-----------------------------------------------------------------------
//   init (data set, column name)
//-----------------------------------------------------------------------


void DataSource::init

  ( const Ref<DataSet>&  dataSet,
    const String&        colName )

{
  idx_t  i;


  dataSet_ = dataSet;
  colFunc_ = nullptr;
  value_   = 0.0;

  if ( colNames_.size() != 1 )
  {
    colNames_.resize ( 1 );
  }

  colNames_[0] = colName;

  if ( ! dataSet )
  {
    setName_ = "";
  }
  else
  {
    setName_ = dataSet->getName ();
    i        = setName_.rfind ( '.' );

    if ( i >= 0 )
    {
      setName_ = setName_[slice(i + 1,END)];
    }
  }
}


//-----------------------------------------------------------------------
//   init (String, data sets)
//-----------------------------------------------------------------------


void DataSource::init

  ( const String&      spec,
    const Properties&  dataSets )

{
  using jem::ParseException;
  using jem::numeric::UserFunc;

  String         s = spec.stripWhite ();

  Ref<DataSet>   dataSet;
  Ref<Function>  colFunc;

  String         setName;
  StringVector   colNames;

  double         value = 0.0;

  idx_t          c;
  idx_t          i, n;


  n = s.size ();
  c = 0;

  for ( i = 0; i < n; i++ )
  {
    if ( s[i] == '[' || s[i] == '{' )
    {
      c = s[i];
      break;
    }
  }

  if ( c == '[' )
  {
    if ( s[n - 1] != ']' )
    {
      throw ParseException ( JEM_FUNC, "missing `]\'" );
    }

    colNames.resize ( 1 );

    setName     = s[slice(BEGIN,i)]    .stripWhite ();
    colNames[0] = s[slice(i + 1,n - 1)].stripWhite ();

    if ( ! colNames[0].size() )
    {
      throw ParseException ( JEM_FUNC,
                             "no data column specified" );
    }
  }
  else if ( c == '{' )
  {
    ArgList_  args;

    if ( s[n - 1] != '}' )
    {
      throw ParseException ( JEM_FUNC,  "missing `}\'" );
    }

    setName = s[slice(BEGIN,i)].stripWhite ();
    colFunc = UserFunc::parse ( args, s[slice(i + 1,n - 1)] );

    colNames.ref ( args.getArgList() );
  }
  else
  {
    setName = s;
  }

  if ( ! setName.size() )
  {
    throw ParseException ( JEM_FUNC, "no data set specified" );
  }

  if ( ! dataSets.find( dataSet, setName ) )
  {
    throw jem::IllegalInputException (
      JEM_FUNC,
      String::format ( "undefined data set: `%s\'", setName )
    );
  }

  // The new data are valid, so commit the changes

  dataSet_ .swap ( dataSet  );
  colFunc_ .swap ( colFunc  );
  setName_ .swap ( setName  );
  colNames_.swap ( colNames );
  jem::     swap ( value_, value );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void DataSource::configure

  ( const String&      name,
    const Properties&  props,
    const Properties&  dataSets )

{
  using jem::toValue;

  Ref<Object>  obj;

  if ( ! props.find( obj, name ) )
  {
    return;
  }

  double  x;

  if ( toValue( x, obj ) )
  {
    init ( x );

    return;
  }

  String  s;

  if ( toValue( s, obj ) )
  {
    try
    {
      init ( s, dataSets );
    }
    catch ( jem::IllegalInputException& ex )
    {
      ex.setContext ( props.getContext( name ) );
      throw;
    }
  }
  else
  {
    props.propertyError (
      name,
      "invalid data type; "
      "must be a floating point number or a string"
    );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void DataSource::getConfig

  ( const String&      name,
    const Properties&  props ) const

{
  props.set ( name, toString() );
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String DataSource::toString () const
{
  if      ( ! dataSet_ )
  {
    return String ( value_ );
  }
  else if ( ! colFunc_ )
  {
    if ( colNames_.size() == 0 )
    {
      return setName_;
    }
    else
    {
      return String::format ( "%s[%s]", setName_, colNames_[0] );
    }
  }
  else
  {
    String  expr = colFunc_->toString ();

    return String::format ( "%s{%s}", setName_, expr );
  }
}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


void DataSource::getData

  ( const Vector&      buf,
    const Properties&  globdat,
    const String&      context,
    const AppContext&  apx ) const

{
  if ( ! dataSet_ )
  {
    buf = value_;
  }
  else
  {
    IdxVector  iitems ( jem::iarray( buf.size() ) );

    getData ( buf, iitems, globdat, context, apx );
  }
}


void DataSource::getData

  ( const Vector&      buf,
    const IdxVector&   iitems,
    const Properties&  globdat,
    const String&      context,
    const AppContext&  apx ) const

{
  using jem::Exception;
  using jive::util::FuncUtils;

  if      ( ! dataSet_ )
  {
    buf = value_;
  }
  else if ( ! colFunc_ )
  {
    if ( colNames_.size() )
    {
      const idx_t  jcol = getColIndex_ ( colNames_[0], context, apx );

      dataSet_->getColData ( buf, iitems, jcol, apx );
    }
    else
    {
      const idx_t  n = dataSet_->columnCount ( apx );

      Matrix       mbuf ( buf.size(), 1 );
      Vector       vbuf ( mbuf[0] );

      IdxVector    jcol ( 1 );


      buf = 0.0;

      for ( idx_t j = 0; j < n; j++ )
      {
        vbuf    = 0.0;
        jcol[0] = j;

        dataSet_->getData ( mbuf, iitems, jcol, apx );

        buf += vbuf * vbuf;
      }

      buf = jem::sqrt ( buf );
    }
  }
  else
  {
    const idx_t  m = buf      .size ();
    const idx_t  n = colNames_.size ();

    Matrix       mbuf  ( n, m );

    IdxVector    jcols ( n );

    double*      args;


    for ( idx_t j = 0; j < n; j++ )
    {
      jcols[j] = getColIndex_ ( colNames_[j], context, apx );
    }

    dataSet_->getData ( mbuf.transpose(), iitems, jcols, apx );

    args = mbuf.addr ();

    try
    {
      FuncUtils::resolve ( *colFunc_, globdat );

      for ( idx_t i = 0; i < m; i++ )
      {
        buf[i] = colFunc_->getValue ( args );
        args  += n;
      }
    }
    catch ( Exception& ex )
    {
      ex.setContext ( context );
      throw;
    }
  }

  if ( Float::isNaN( jem::sum( buf ) ) )
  {
    throw IllegalInputException ( context, "invalid data: NaN" );
  }
}


//-----------------------------------------------------------------------
//   getColIndex_
//-----------------------------------------------------------------------


idx_t DataSource::getColIndex_

  ( const String&      name,
    const String&      context,
    const AppContext&  apx ) const

{
  idx_t  jcol = dataSet_->findColumn ( name, apx );

  if ( jcol < 0 )
  {
    throw IllegalInputException (
      context,
      String::format (
        "data set `%s\' has no column named `%s\'",
        setName_,
        name
      )
    );
  }

  return jcol;
}


JIVE_END_PACKAGE( gl )
