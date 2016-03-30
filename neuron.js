//************************************************
//
// neuron.js by Takashi Tanaka
//
//      dependency: jquery, mt.js
//
//************************************************
"use strict";

//--------------------------------------------------------
// Random Value generator
//--------------------------------------------------------
var mt = new MersenneTwister(1234);

// Uniform distribution 一様分布ランダム値
var getRandFuncUniform = function(min, max){
    var range = max - min;
    return function(){
        return mt.next() * range + min;
    }
};

// Triangular distribution 三角分布ランダム値
var getRandFuncTriangular = function(min, max){
    var range = max - min;
    return function(){
        return (mt.next() + mt.next()) * 0.5 * range + min;
    }
}

//--------------------------------------------------------
//Fire functions (from http://code.google.com/p/cuda-convnet/wiki/NeuronTypes)
//--------------------------------------------------------
var step = function(x){
    return Math.min(1, Math.max(0, x<<3));
};
var logistic = function(x){
    return 1 / (1 + Math.exp(-x));
};
var hyperbolicTangent  = function(x){
    return a * rational_tanh(b * x);
};
var rectifiedLinear  = function(x){
    return Math.max(0, x);
};
var boundedRectifiedLinear = function(x){
    return Math.min(1, Math.max(0, x));
};
var softRectifiedLinear  = function(x){
    return Math.ln(1 + Math.exp(x));
};
var absoluteValue  = function(x){
    return Math.abs(x); 
};
var square  = function(x){
    return x * x;
};
var squareRoot = function(x){
    return Math.sqrt(x)
};
var linear = function(x){
    return a * x + b;
};
//SoftSign
var softSign = function(x){  //output -> [-1, 1]
    return x / (1 + Math.abs(x))
};
var arcSoftSign = function(x){
    return x / (1 - Math.abs(x))
};
//
var softSign2 = function(x){  //output -> [0, 1]
    return (x / (1 + Math.abs(x))) * 0.5 + 0.5
};

//--------------------------------------------------------
//Penalty functions (lesser is better)
var penaltyLogistic = function(desired, x){
    return -(desired * Math.log(x) + (1 - desired) * Math.log(1 - x));
};

var penaltyDiff = function(desired, x){
    return Math.abs(desired - x);
};


//--------------------------------------------------------
//与えられた多次元配列の各次元の長さを求める
var getArraySize = function(argArray){
    var dimension = [];
    
    //recursive
    var rec = function(array){
        if(Array.isArray(array)){
            dimension.push(array.length);
            rec(array[0]);
        }else{
            return;
        }
    }
    
    rec(argArray);
    return dimension;
};

//--------------------------------------------------------
//与えられた多次元配列の値の合計を求める
var getArraySum = function(argArray){
    //recursive
    var rec = function(array){
        var localSum = 0;
        if(Array.isArray(array[0])){
            for(var i = 0; i < array.length; i++){
                localSum += rec(array[i]);
            }
        }else{
            for(var i = 0; i < array.length; i++){
                localSum += array[i];
            }
        }
        return localSum;
    }
    
    return rec(argArray);
};



//--------------------------------------------------------
// Main
//--------------------------------------------------------
var Neuron = function(){
    var my = {
        neuronHash: {},  //all neuron hash (key: neuron id)
        groupHash: {},  //all neuron-group hash (key: group id)
    };
    
    //--------------------------------------------------------
    //Get neuron
    my.neuron = function(id){
        return my.neuronHash[id];
    };
    
    //--------------------------------------------------------
    //Get group
    my.group = function(id){
        return my.groupHash[id];
    };
    
    //--------------------------------------------------------
    // Neuron
    //--------------------------------------------------------
    my.createNeuron = function(argOptions){
        var neuron = {
            id: null,  
            parent: null,
            pos: null,  //Position at Group
            output: 0,
            bias: 0,  //Threshould
            history: [], //Output history
            to: {}, //Connection hash to next neuron (key: neuron id, value: weight)
            from: {}, //Connection hash from pre neuron (key: neuron id, value: weight) 
            changes: {},  //Weight change history (only last. for moment action.) key: to neuron id)
            initWeightFunc: null,
            //preFunc: null,  //pre fire function(like Low pass filter)
            func: null,  //function for Fire
            //afterFunc: null,  //after fire function(like High pass filter)
            time: 0,
            temprature: 1,
            historyCount: 50,
        };
        neuron = $.extend(neuron, argOptions);
        
        //--------------------------------------------------------
        //Create connection to other neuron
        neuron.connectTo = function(toNeuron){
            var weight = neuron.initWeightFunc();
            toNeuron.from[ neuron.id ] = weight;
            neuron.to[ toNeuron.id ] = weight;
        };
        
        //--------------------------------------------------------
        //Remove connection to other neuron
        neuron.disconnectTo = function(toNeuron){
            var weight = neuron.initWeightFunc();
            delete toNeuron.from[ neuron.id ];
            delete neuron.to[ toNeuron.id ];
        };
        
        //--------------------------------------------------------
        //Get Sparseness (from viewpoint of time)
        neuron.getTimeSparseValue = function(){
            var len = neuron.history.length;
            var sum = 0;
            for(var i = 0; i < len; i++){
                sum += neuron.history[i];
            }
            return sum / len;
        };

        //--------------------------------------------------------
        //Fire (determin Output)
        neuron.fire = function(){
            //Calc fire
            var weightedSum = 0;
            var idarray = Object.keys(neuron.from);
            var len = idarray.length
            for(var i = 0; i < len; i++){
                var id = idarray[i];
                var val = my.neuronHash[ id ].output;
                var weight = neuron.from[ id ];
                weightedSum += val * weight;
            }
            weightedSum += neuron.bias;
            var output = neuron.func( weightedSum );
            neuron.setOutput( output );
        };
        
        //--------------------------------------------------------
        //Set output of neuron
        neuron.setOutput = function(value){
            neuron.time++;
            neuron.output = value;
            
            //Fire history
            neuron.history.push( neuron.output );
            if(neuron.historyCount < neuron.history.length){
                neuron.history.shift();
            }
        };
        
        //--------------------------------------------------------
        //Set teacher
        neuron.setTeacher = function(value){
            neuron.teacher = value;
        };
        
        //--------------------------------------------------------
        //init Neuron
        neuron.init = function(){
            neuron.bias = neuron.initWeightFunc();
            my.neuronHash[ neuron.id ] = neuron;
        };
        
        //--------------------------------------------------------
        neuron.init();
        return neuron;
    };
    
    
    
    //--------------------------------------------------------
    // Neuron Group
    //  Group can be a layer or any type of neuron set.
    //--------------------------------------------------------
    my.createGroup = function(argOptions){
        var group = {
            id: null,
            parent: my,
            counts: null,
            neurons: [],
            neuronHash: {},
            to: {}, //Connection hash to next group (key: group id, value: true)
            from: {}, //Connection hash from pre group (key: group id, value: true) 
            teacher: null,
            neuronCount: 0,
            //
            initWeightFuntion: function(){return Math.random() * 0.4 - 0.2;},
            fireFunction: null,
            fireProcess: null,
            learnProcess: null,
            //for viewer
            connectionW: null,
            connectionH: null,
        };
        group = $.extend(group, argOptions);
        
        //--------------------------------------------------------
        //Apply some function to all neurons in group
        group.applyFunc = function(func){
            var rec = function(neurons){
                var len = neurons.length;
                if(Array.isArray(neurons[0])){
                    for(var i = 0; i < len; i++){
                        rec(neurons[i]);
                    }
                }else{
                    for(var i = 0; i < len; i++){
                        func(neurons[i]);
                    }
                }
            }
            rec(group.neurons);
        };
        
        //--------------------------------------------------------
        //Connect To Group
        group.connectTo = function( toGroup, fromXYWH, toXYWH ){
            var fx = (fromXYWH && fromXYWH.x) || 0;
            var fy = (fromXYWH && fromXYWH.y) || 0;
            var fw = (fromXYWH && fromXYWH.w) || 1000000;
            var fh = (fromXYWH && fromXYWH.h) || 1000000;
            var tx = (toXYWH && toXYWH.x) || 0;
            var ty = (toXYWH && toXYWH.y) || 0;
            var tw = (toXYWH && toXYWH.w) || 1000000;
            var th = (toXYWH && toXYWH.h) || 1000000;
            
            group.to[ toGroup.id ] = true;
            toGroup.from[ group.id ] = true;
            
            group.applyFunc(function( neuron ){
                var nfx = neuron.pos[1];
                var nfy = neuron.pos[0];
                if(fx <= nfx && nfx < (fx + fw) && fy <= nfy && nfy < (fy + fh)){
                    toGroup.applyFunc(function( toNeuron ){
                        var ntx = toNeuron.pos[1];
                        var nty = toNeuron.pos[0];
                        if(tx <= ntx && ntx < (tx + tw) && ty <= nty && nty < (ty + th)){
                            neuron.connectTo( toNeuron );
                        }
                    });
                }
            });
        };
        
        //--------------------------------------------------------
        //Remove Connection (To Group)
        group.disconnectTo = function( toGroup ){
            delete group.to[ toGroup.id ];
            delete toGroup.from[ group.id ];
            
            group.applyFunc(function( neuron ){
                toGroup.applyFunc(function( toNeuron ){
                    neuron.disconnectTo( toNeuron )
                });
            });
        };
        
        //--------------------------------------------------------
        //return all outputs
        group.getOutput = function(){
            var rec = function(neurons){
                var arr = [];
                if(Array.isArray(neurons[0])){
                    for(var i = 0; i < neurons.length; i++){
                        arr.push( rec(neurons[i]) );
                    }
                }else{
                    for(var i = 0; i < neurons.length; i++){
                        arr.push( neurons[i].output );
                    }
                }
                return arr;
            };
            return rec(group.neurons);
        };
        
        //--------------------------------------------------------
        //set all outputs (usually, for Input Layer)
        group.setOutput = function(data){
            var rec = function(src, neuron){
                if(Array.isArray(src[0])){
                    for(var i = 0; i < src.length; i++){
                        rec(src[i], neuron[i]);
                    }
                }else{
                    for(var i = 0; i < src.length; i++){
                        neuron[i].setOutput( src[i] );
                    }
                }
            };
            rec(data, group.neurons);
        };
        
        //--------------------------------------------------------
        //set Teacher data (not required)
        group.setTeacher = function(data){
            var rec = function(src, neuron){
                if(Array.isArray(src[0])){
                    for(var i = 0; i < src.length; i++){
                        rec(src[i], neuron[i]);
                    }
                }else{
                    for(var i = 0; i < src.length; i++){
                        neuron[i].setTeacher( src[i] );
                    }
                }
            };
            rec(data, group.neurons);
        };
        
        //--------------------------------------------------------
        //Calculate sparseness (from Viewpoint of space sparseness)
        group.getSpaceSparseValue = function(){
            var count = 0;
            var sum = 0;
            group.applyFunc(function(neuron){
                count++;
                sum += neuron.output;
            });
            return sum / count;
        };
        
        //--------------------------------------------------------
        //firing process of neurons of group
        group.fire = function(){
            group.fireProcess(group);
        };
        
        //--------------------------------------------------------
        //learning process of neurons of group
        group.learn = function(){
            group.learnProcess(group);
        };
        
        //--------------------------------------------------------
        //Initial process
        group.init = function(){
            //Create Neurons
            var rec = function(pos){
                var dimension = pos.length + 1
                var count = group.counts[dimension - 1];
                var neuronArray = [];
                
                if(dimension == group.counts.length){  //if last dimention
                    for(var i = 0; i < count; i++){
                        var neuronId = group.id + ("000" + group.neuronCount).substr(-4);
                        pos.push(i);
                        var copiedPos = $.extend([], pos);
                        
                        var neuron = my.createNeuron({
                            id: neuronId,
                            parent: group,
                            pos: copiedPos,
                            initWeightFunc: group.initWeightFuntion,
                            func: group.fireFunction,
                        });
                        neuronArray.push( neuron );
                        pos.pop();
                        
                        group.neuronCount++;
                    }
                }else{
                    for(var i = 0; i < count; i++){
                        pos.push(i);
                        neuronArray.push( rec(pos) );
                        pos.pop();
                    }
                }
                
                return neuronArray;
            };
            group.neurons = rec([]);
        };
        
        //--------------------------------------------------------
        group.init();
        return group;
    };
    
    return my;
};
