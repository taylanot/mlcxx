
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

#ifndef JIVE_GL_DATASOURCE_H
#define JIVE_GL_DATASOURCE_H

#include <jive/gl/DataSet.h>


JIVE_BEGIN_PACKAGE( gl )


class AppContext;


//-----------------------------------------------------------------------
//   class DataSource
//-----------------------------------------------------------------------


class DataSource
{
 public:

  typedef DataSource      Self;


  explicit                DataSource

    ( double                value = 0.0 );

                          DataSource

    ( const DataSource&     rhs );

  DataSource&             operator =

    ( const DataSource&     rhs );

  void                    init

    ( double                value );

  void                    init

    ( const Ref<DataSet>&   dataSet,
      const String&         colName );

  void                    init

    ( const String&         spec,
      const Properties&     dataSets );

  void                    configure

    ( const String&         name,
      const Properties&     props,
      const Properties&     dataSets );

  void                    getConfig

    ( const String&         name,
      const Properties&     props )      const;

  String                  toString    () const;

  inline ItemSet*         getItems

    ( const AppContext&     apx )        const;

  inline DataSet*         getDataSet  () const;

  void                    getData

    ( const Vector&         buf,
      const Properties&     globdat,
      const String&         context,
      const AppContext&     apx )        const;

  void                    getData

    ( const Vector&         buf,
      const IdxVector&      iitems,
      const Properties&     globdat,
      const String&         context,
      const AppContext&     apx )        const;


 private:

  idx_t                   getColIndex_

    ( const String&         name,
      const String&         context,
      const AppContext&     apx )        const;


 private:

  class                   ArgList_;


  Ref<DataSet>            dataSet_;
  Ref<Function>           colFunc_;

  String                  setName_;
  StringVector            colNames_;

  double                  value_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getItems
//-----------------------------------------------------------------------


inline ItemSet* DataSource::getItems

  ( const AppContext& apx ) const

{
  if ( ! dataSet_ )
  {
    return nullptr;
  }
  else
  {
    return dataSet_->getItems ( apx );
  }
}


//-----------------------------------------------------------------------
//   getDataSet
//-----------------------------------------------------------------------


inline DataSet* DataSource::getDataSet () const
{
  return dataSet_.get ();
}


JIVE_END_PACKAGE( gl )

#endif
