
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

#ifndef JIVE_GL_DATASET_H
#define JIVE_GL_DATASET_H

#include <jem/base/NamedObject.h>
#include <jive/Array.h>
#include <jive/gl/import.h>


JIVE_BEGIN_PACKAGE( gl )


class AppContext;


//-----------------------------------------------------------------------
//   class DataSet
//-----------------------------------------------------------------------


class DataSet : public NamedObject
{
 public:

  JEM_DECLARE_CLASS       ( DataSet, NamedObject );


  virtual void              update

    ( const Properties&       globdat,
      const AppContext&       apx )                = 0;

  virtual void              configure

    ( const Properties&       props,
      const Properties&       globdat,
      const AppContext&       apx );

  virtual void              getConfig

    ( const Properties&       props )        const;

  virtual idx_t             rowCount

    ( const AppContext&       apx )          const;

  virtual idx_t             columnCount

    ( const AppContext&       apx )          const = 0;

  virtual idx_t             findColumn

    ( const String&           name,
      const AppContext&       apx )          const = 0;

  virtual ItemSet*          getItems

    ( const AppContext&       apx )          const = 0;

  virtual void              getData

    ( const Matrix&           buf,
      const IdxVector&        irows,
      const IdxVector&        jcols,
      const AppContext&       apx )          const = 0;

  void                      getColData

    ( const Vector&           buf,
      idx_t                   jcol,
      const AppContext&       apx )          const;

  void                      getColData

    ( const Vector&           buf,
      const IdxVector&        irows,
      idx_t                   jcol,
      const AppContext&       apx )          const;

  void                      getRowData

    ( const Vector&           buf,
      idx_t                   irow,
      const AppContext&       apx )          const;

  void                      getRowData

    ( const Vector&           buf,
      idx_t                   irows,
      const IdxVector&        jcols,
      const AppContext&       apx )          const;

  virtual String            getContext    () const override;


 protected:

  explicit                  DataSet

    ( const String&           name = ""  );

  virtual                  ~DataSet     ();

};


JIVE_END_PACKAGE( gl )

#endif
