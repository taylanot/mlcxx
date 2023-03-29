
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

#ifndef JEM_HDF5_H5FILE_H
#define JEM_HDF5_H5FILE_H

#include <jem/base/Object.h>


JEM_BEGIN_PACKAGE( hdf5 )


class Group;
class Handle;


//-----------------------------------------------------------------------
//   class H5File
//-----------------------------------------------------------------------


class H5File : public Object
{
 public:

  JEM_DECLARE_CLASS       ( H5File, Object );

  static const int          TRUNCATE;
  static const int          EXCLUSIVE;

  static const int          READ_WRITE;
  static const int          READ_ONLY;


                            H5File

    ( const Ref<Handle>&      fid,
      const String&           name,
      int                     flags = 0 );

  static Ref<H5File>        open

    ( const String&           name,
      int                     flags );

  static Ref<H5File>        create

    ( const String&           name,
      int                     flags );

  void                      flush       () const;
  void                      close       ();
  String                    getName     () const noexcept;
  int                       getFlags    () const noexcept;
  Ref<Group>                getRoot     () const;


 protected:

  virtual                  ~H5File      ();


 private:

  Ref<Handle>               fid_;
  String                    name_;
  int                       flags_;

};


JEM_END_PACKAGE( hdf5 )

#endif
