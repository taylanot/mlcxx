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

class Model
{
public:
    virtual ~Model( ) { };

    virtual void Train( const arma::mat& inputs,
                        const arma::rowvec& labels ) = 0;

    virtual void Predict( const arma::mat& inputs,
                          arma::rowvec& labels ) = 0;

};

#endif

