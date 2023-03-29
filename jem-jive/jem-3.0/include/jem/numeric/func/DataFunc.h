
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

#ifndef JEM_NUMERIC_FUNC_DATAFUNC_H
#define JEM_NUMERIC_FUNC_DATAFUNC_H

#include <jem/base/array/Array.h>
#include <jem/io/Serializable.h>
#include <jem/numeric/import.h>
#include <jem/numeric/func/Function.h>


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   class DataFunc
//-----------------------------------------------------------------------


class DataFunc : public Function,
                 public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( DataFunc, Function );


                            DataFunc    ();

                            DataFunc

    ( const Array<double>&    xData,
      const Array<double>&    yData );

                            DataFunc

    ( Array<double>&&         xData,
      Array<double>&&         yData );

  virtual void              readFrom

    ( ObjectInput&            in )         override;

  virtual void              writeTo

    ( ObjectOutput&           out )        const override;

  virtual idx_t             argCount    () const override;

  virtual double            getValue

    ( const double*           args )       const override;

  virtual double            getDeriv

    ( idx_t                   iarg,
      const double*           args )       const override;

  virtual idx_t             markUsedArgs

    ( bool*                   mask )       const override;

  void                      setData

    ( const Array<double>&    xData,
      const Array<double>&    yData );

  Array<double>             getXData    () const noexcept;
  Array<double>             getYData    () const noexcept;


 protected:

  virtual                  ~DataFunc    ();


 private:

  static bool               checkXData_

    ( const Array<double>&    xData );


 private:

  Array<double>             xData_;
  Array<double>             yData_;

};


JEM_END_PACKAGE( numeric )

#endif
