
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

#ifndef JIVE_APP_DATAPRINTER_H
#define JIVE_APP_DATAPRINTER_H

#include <jem/base/NamedObject.h>
#include <jive/Array.h>
#include <jive/app/import.h>


namespace jem
{
  namespace mp
  {
    class GatherPrinter;
  }
}


JIVE_BEGIN_PACKAGE( app )


//-----------------------------------------------------------------------
//   class DataPrinter
//-----------------------------------------------------------------------


class DataPrinter : public NamedObject
{
 public:

  JEM_DECLARE_CLASS       ( DataPrinter, NamedObject );

  typedef jem
    ::mp::GatherPrinter     Output;


  explicit                  DataPrinter

    ( const String&           name = "" );

  virtual void              init

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  virtual void              shutdown

    ( const Properties&       globdat );

  virtual void              printTable

    ( Output&                 out,
      const String&           label,
      const Table&            table,
      const Properties&       globdat )          = 0;

  virtual void              printVector

    ( Output&                 out,
      const String&           label,
      const Vector&           vec,
      const DofSpace&         dofs,
      const Properties&       globdat )          = 0;

  virtual void              printSparseVector

    ( Output&                 out,
      const String&           label,
      const Vector&           vec,
      const IdxVector&        idofs,
      const DofSpace&         dofs,
      const Properties&       globdat )          = 0;

  virtual String            getContext  () const override;


 protected:

  virtual                  ~DataPrinter ();

};


JIVE_END_PACKAGE( app )

#endif
