
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


#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/numeric/func/DataFunc.h>


JEM_DEFINE_SERIABLE_CLASS( jem::numeric::DataFunc );


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class DataFunc
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


DataFunc::DataFunc () :

  xData_ ( 2 ),
  yData_ ( 2 )

{
  xData_[0] = 0.0;
  xData_[1] = 1.0;
  yData_[0] = 0.0;
  yData_[1] = 0.0;
}


DataFunc::DataFunc

  ( const Array<double>&  xData,
    const Array<double>&  yData )

{
  setData ( xData, yData );
}


DataFunc::~DataFunc ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void DataFunc::readFrom ( ObjectInput& in )
{
  decode ( in, xData_, yData_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void DataFunc::writeTo ( ObjectOutput& out ) const
{
  encode ( out, xData_, yData_ );
}


//-----------------------------------------------------------------------
//   argCount
//-----------------------------------------------------------------------


idx_t DataFunc::argCount () const
{
  return 1;
}


//-----------------------------------------------------------------------
//   getValue
//-----------------------------------------------------------------------


double DataFunc::getValue ( const double* args ) const
{
  const double  x = args[0];

  double        a, b;
  idx_t         i, j, k;


  i = 0;
  j = xData_.size() - 2;

  while ( i < j )
  {
    k = i + (j - i + 1) / 2;

    if ( x >= xData_[k] )
    {
      i = k;
    }
    else
    {
      j = k - 1;
    }
  }

  k = j + 1;

  if      ( x <= xData_[j] )
  {
    return yData_[j];
  }
  else if ( x >= xData_[k] )
  {
    return yData_[k];
  }
  else
  {
    a = (xData_[k] - x) / (xData_[k] - xData_[j]);
    b = 1.0 - a;

    return (a * yData_[j] + b * yData_[k]);
  }
}


//-----------------------------------------------------------------------
//   getDeriv
//-----------------------------------------------------------------------


double DataFunc::getDeriv

  ( idx_t          iarg,
    const double*  args ) const

{
  JEM_PRECHECK2 ( iarg == 0, "invalid DataFunc argument index" );

  const double  x = args[0];
  const idx_t   n = xData_.size ();

  if ( x < xData_[0] || x > xData_[n - 1] )
  {
    return 0.0;
  }

  idx_t  i, j, k;


  i = 0;
  j = n - 2;

  while ( i < j )
  {
    k = i + (j - i + 1) / 2;

    if ( x >= xData_[k] )
    {
      i = k;
    }
    else
    {
      j = k - 1;
    }
  }

  k = j + 1;

  return ((yData_[k] - yData_[j]) / (xData_[k] - xData_[j]));
}


//-----------------------------------------------------------------------
//   markUsedArgs
//-----------------------------------------------------------------------


idx_t DataFunc::markUsedArgs ( bool* mask ) const
{
  mask[0] = true;

  return 1;
}


//-----------------------------------------------------------------------
//   setData
//-----------------------------------------------------------------------


void DataFunc::setData

  ( const Array<double>&  xData,
    const Array<double>&  yData )

{
  JEM_PRECHECK2 ( xData.size() == yData.size(),
                  "x-y data array sizes do not match" );
  JEM_PRECHECK2 ( xData.size() > 1,
                  "x-y data arrays must contain at least "
                  "one element" );

  if ( ! checkXData_( xData ) )
  {
    throw IllegalInputException (
      JEM_FUNC,
      "x-data array must contain strictly ascending elements"
    );
  }

  xData_.ref ( xData );
  yData_.ref ( yData );
}


//-----------------------------------------------------------------------
//   getXData & getYData
//-----------------------------------------------------------------------


Array<double> DataFunc::getXData () const noexcept
{
  return xData_;
}


Array<double> DataFunc::getYData () const noexcept
{
  return yData_;
}


//-----------------------------------------------------------------------
//   checkXData_
//-----------------------------------------------------------------------


bool DataFunc::checkXData_ ( const Array<double>& x )
{
  const idx_t  n = x.size() - 1;

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( x[i] >= x[i + 1] )
    {
      return false;
    }
  }

  return true;
}


JEM_END_PACKAGE( numeric )
