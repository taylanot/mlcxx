/**
 * @file model.h
 * @author Ozgur Taylan Turan
 *
 * Base class for your models
 *
 *
 */

#ifndef MODEL_H
#define MODEL_H

class BaseModel 
{

  public:

    BaseModel() {};

    BaseModel(const jem::util::Properties modelProps) : modelProps_(modelProps)
                                                                           {};
    
    virtual void Train(const utils::data::regression::Dataset& trainset) = 0;

    virtual double Test(const utils::data::regression::Dataset& testset) = 0;

    virtual std::tuple<arma::mat, arma::mat> LearningCurve
                        (const utils::data::regression::Dataset& dataset, 
                         const jive::IdxVector N_bnds,
                         const int N_res,
                         const int repat) = 0;

    void DoObjective(const jem::util::Properties objProps,
                     const utils::data::regression::Dataset& dataset)
    {
      jem::String obj;
      
      objProps.find(obj,"type");

      if ( obj == "learningcurve" )
      {
        N_res_  = 20;
        repeat_ = 100;

        N_bnds_.resize(2); 
        N_bnds_[0] = 5; 
        N_bnds_[1] = 100;
        
        objProps.find( repeat_, "rep"      );
        objProps.find( N_bnds_, "N_bounds" );
        objProps.find( N_res_,  "res"      );

        this -> LearningCurve(dataset, N_bnds_, N_res_, repeat_);
      }
      else if ( obj == "train/test" )
      {
        double err = this -> Test(dataset);
        std::cout << err << std::endl;
      }
    };

  protected:

    int                     N_res_;
    int                     repeat_;
    jive::IdxVector         N_bnds_;
    jem::util::Properties   modelProps_;

};

#endif

