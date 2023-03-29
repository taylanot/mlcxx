
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

#ifndef JIVE_MP_DATAEXCHANGER_H
#define JIVE_MP_DATAEXCHANGER_H

#include <jem/base/Object.h>
#include <jem/base/array/Array.h>
#include <jive/util/Database.h>
#include <jive/mp/import.h>


JIVE_BEGIN_PACKAGE( mp )


class BorderSet;


//-----------------------------------------------------------------------
//   class DataExchanger
//-----------------------------------------------------------------------


class DataExchanger : public Object
{
 public:

  JEM_DECLARE_CLASS       ( DataExchanger, Object );

  typedef Database
    ::IntColArray           IntColArray;
  typedef Database
    ::FloatColArray         FloatColArray;


                            DataExchanger

    ( const Ref<Context>&     mpx,
      const BorderSet&        borders );

                            DataExchanger

    ( const Ref<Context>&     mpx,
      const BorderSet&        recvBorders,
      const BorderSet&        sendBorders );

  virtual String            toString      () const override;

  void                      exchange

    ( Database&               dbase );

  void                      exchange

    ( const IntColArray&      cols );

  void                      exchange

    ( const FloatColArray&    cols );

  void                      store

    ( const Properties&       globdat )      const;

  static Ref<Self>          find

    ( const Ref<ItemSet>&     items,
      const Properties&       globdat );

  static Ref<Self>          get

    ( const Ref<ItemSet>&     items,
      const Properties&       globdat );


 protected:

  virtual                  ~DataExchanger ();


 private:

  void                      init_

    ( const Ref<Context>&     mpx,
      const BorderSet&        recvBorders,
      const BorderSet&        sendBorders );

  void                      update_       ();
  void                      invalidate_   ();


 private:

  class                     XData_;
  class                     RecvData_;
  class                     SendData_;
  class                     Utils_;

  friend class              RecvData_;
  friend class              SendData_;
  friend class              Utils_;

  static const int          XTAG_;

  Ref<RecvData_>            recvData_;
  Ref<SendData_>            sendData_;

  bool                      updated_;

};


JIVE_END_PACKAGE( mp )

#endif
