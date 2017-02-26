#ifndef SNOOPY_ML_LAYER_FACTORY_H_
#define SNOOPY_ML_LAYER_FACTORY_H_

/**
 * Factory class to get the layer 
 */
#include <map>
#include <string>
#include <vector>
#include "../proto/snoopy.pb.h"
#include "../common/logging.h"

using std::shared_ptr;
using std::map;
using std::string;
using std::vector;

namespace snoopy {
namespace ml {

//prefix class
template<typename DataType>
class Layer;

template<typename DataType>
class LayerRegiste {
public:
    typedef shared_ptr<Layer<DataType>> (*LayerCreator) (const LayerParameter & lp);
    typedef map<string, LayerCreator> LayerReisterMap;

    static LayerReisterMap& layer_register() {
        static LayerReisterMap * lr = new LayerReisterMap;
        return *lr;
    }

    static void add_creator(const string & type, LayerCreator c) {
       LayerReisterMap & lr = layer_register();
       CHECK_EQ(lr.count(type), 0);
       lr[type] = c; 
    } 

    static shared_ptr<Layer<DataType>> create_layer(const LayerParameter & lp) {
        string t = lp.type();
        LayerReisterMap & lr = layer_register();
        CHECK_EQ(lr.count(t), 1);
        return lr[t](lp); 
    }

    static vector<string> layer_type_list() {
        vector<string> tmp_list;
        LayerReisterMap & lr = layer_register();
        for(auto iter = lr.begin();
                iter != lr.end();
                ++iter) {
            tmp_list.push_back(iter->first);
        }
        return tmp_list;
    }

    private:
        LayerRegiste() {}; //singlton
    
};


template<typename DataType>
class LayerRegister {
    public:
        LayerRegister(const string & type,
            shared_ptr<Layer<DataType>> (*c) (const LayerParameter & lp)) {
            LayerRegiste<DataType>::add_creator(type, c);
        };
};


#define LAYER_REGISTER_CREATOR(type, creator)                                               \
    static LayerRegister<float> g_creator_##type(#type, creator<float>);


#define LAYER_REGISTER_CLASS(type)                                                          \
    template<typename DataType>                                                             \
    shared_ptr<Layer<DataType> > create_##type##layer(const LayerParameter & lp) {   \
        return shared_ptr<Layer<DataType> > (new type##Layer<DataType>(lp));                           \
    }                                    \
    LAYER_REGISTER_CREATOR(type, create_##type##layer);   

}
}

#endif
