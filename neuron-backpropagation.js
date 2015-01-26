//************************************************
//
// Methods of Fire and Learning.
//
//************************************************
"use strict";
//--------------------------------------------------------
// Back Propagetion
//--------------------------------------------------------
//BP fire Process
var fireBp = function(group){
    group.applyFunc(function(neuron){
        neuron.fire();
    });
};

//--------------------------------------------------------
//BP learn Process
var learnBp = function(group){
    var nn = group.parent;
    var learningRate = 0.3;
    var momentum = 0.2;
    var dropoutTime = 60000;
    
    group.applyFunc(function(neuron){
        //calculateDeltas
        var error = 0;
        if(typeof(neuron.teacher) !== 'undefined'){  //if teacher exist (probably layer of output)
            error = neuron.teacher - neuron.output;
        }else{
            var idarray = Object.keys(neuron.to);
            var len = idarray.length
            for(var i = 0; i < len; i++){
                var id = idarray[i];  //toNeuronId
                var nextDelta = nn.neuronHash[ id ].delta;
                var weight = neuron.to[ id ];
                error += nextDelta * weight;
            }
        }
        neuron.delta = error * neuron.output * (1 - neuron.output);
        
        //adjustWeights
        var idarray = Object.keys(neuron.from);
        var len = idarray.length
        for(var i = 0; i < len; i++){
            var id = idarray[i]; //fromNeuronId
            var weight = neuron.from[ id ];
            var incoming = nn.neuron( id ).output;
            var lastChange = 0;
            
            if(typeof(neuron.changes[ id ]) !== 'undefined'){
                lastChange = neuron.changes[ id ];
            }
            var change = (learningRate * neuron.delta * incoming) + (momentum * lastChange);
            neuron.changes[ id ] = change;
            
            //update weight
            neuron.from[ id ] += change;
            // copy weight to next neuron info
            nn.neuron( id ).to[ neuron.id ] = neuron.from[ id ];
            
            //auto dropout connection
            if(Math.abs(neuron.from[ id ]) < 0.10 && dropoutTime < neuron.time){
                delete neuron.from[ id ];
                delete nn.neuron( id ).to[ neuron.id ];
            }
        }
        neuron.bias += learningRate * neuron.delta;
    });
};
