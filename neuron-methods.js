//************************************************
//
// Methods of Fire and Learning.
//
//************************************************
"use strict";
//--------------------------------------------------------
//const
var TIME_SPARSE_DESIRED = 0.05;  //average of output history
var SPACE_SPARSE_DESIRED = 0.05;  //average of output at group



//--------------------------------------------------------
// New Algorithm
//--------------------------------------------------------
//Sparsenessを理想に近づける学習（重み＆バイアス修正）
var sparseLearn = function(group, spaceSparse, timeSparse){
    var nn = group.parent;
    var ss = group.getSpaceSparseValue();
    var ssError = ss - spaceSparse;
    //console.log("spaceSparse:" + ss);
    //console.log("ssError:" + ssError);
    
    group.applyFunc(function(neuron){
        var ts = neuron.getTimeSparseValue();
        var tsError = ts - timeSparse;
        //console.log("timeSparseness:" + ts);
        //console.log("tsError:" + tsError);
        
        //sparsenesによる重みとバイアス修正
        var ssWeight = 0;
        var tsWeight = 1;
        var alpha = -(ssWeight * ssError + tsWeight * tsError); 
        
        //
        var idarray = Object.keys(neuron.from);
        var len = idarray.length
        for(var i = 0; i < len; i++){
            var id = idarray[i]; //fromNeuronId
            // update weight
            neuron.from[ id ] += neuron.output * alpha * neuron.temprature;
            // copy weight to next neuron info
            nn.neuronHash[ id ].to[ neuron.id ] = neuron.from[ id ];
        }
        neuron.bias += alpha * neuron.temprature;
    });
};

//--------------------------------------------------------
//分散値を理想に近づける学習（重み＆バイアス修正）
var varianceLearn = function(group, desiredValiance){
    var nn = group.parent;
    group.applyFunc(function(neuron){
        //バラツキを大きくするよう重み、バイアスを修正
        var count = 0;
        var sum = 0;
        var squaredSum = 0;
        //
        var idarray = Object.keys(neuron.from);
        var len = idarray.length
        for(var i = 0; i < len; i++){
            var id = idarray[i]; //fromNeuronId
            var weight = neuron.from[id];
            count++;
            sum += weight;
            squaredSum += weight * weight;
        }
        count++;
        sum += neuron.bias;
        squaredSum += neuron.bias * neuron.bias;
        
        var avg = sum / count;
        var variance = (squaredSum / count) - (avg * avg);  //分散
        //console.log("variance:" + variance);
        
        //
        var idarray = Object.keys(neuron.from);
        var len = idarray.length
        for(var i = 0; i < len; i++){
            var id = idarray[i]; //fromNeuronId
            var weight = neuron.from[ id ];
            // update weight
            neuron.from[ id ] += (weight - avg) * (desiredValiance - variance) * 0.1;
            // copy weight to next neuron info
            nn.neuronHash[ id ].to[ neuron.id ] = neuron.from[ id ];
        }
        neuron.bias += (neuron.bias - avg) * (desiredValiance - variance) * 0.1;
    });
};

//fire Process
var fireTest = function(group){
    group.applyFunc(function(neuron){
        neuron.fire();
    });
};

//learn Process
var learnTest = function(group){
    sparseLearn(group, SPACE_SPARSE_DESIRED, TIME_SPARSE_DESIRED);
    varianceLearn(group, 1);
};





//--------------------------------------------------------
// SOM (Self-Organizing Model)
//--------------------------------------------------------
//SOM fire process
var fireSom = function(group){
    var nn = group.parent;
    var best = [];
    var fireCount = 1;
    for(var i = 0; i < fireCount; i++){
        best.push({
            id: null,
            distance: Infinity  //lesser is better
        });
    }
    
    group.applyFunc(function(neuron){
        var distance = 0;
        
        //
        var idarray = Object.keys(neuron.from);
        var len = idarray.length;
        for(var i = 0; i < len; i++){
            var id = idarray[i]; //fromNeuronId
            var weight = neuron.from[ id ];
            var value = nn.neuronHash[ id ].output;
            distance += (value - weight) * (value - weight);
        }
        
        var len = best.length;
        for(var i = 0; i < len; i++){
            if(distance < best[i].distance){
                for(var j = len-1; j > i; j--){
                    best[j] = best[j-1];
                }
                best[i] = {
                    id: neuron.id,
                    distance: distance,
                }
                break;
            }
        }
    });
    
    //Fire
    group.applyFunc(function(neuron){
        var newOutput = 0;
        var len = best.length;
        for(var i = 0; i < len; i++){
            if(neuron.id == best[i].id){
                var newOutput = 1;
                break;
            }
        }
        neuron.setOutput( newOutput );
    }); 
}

//SOM Learn Process
var learnSom = function(group){
    var nn = group.parent;
    var bestOutput = -Infinity;
    var winner = null;
    var learningRate = 0.05;
    var distancePara = 0.3;
    
    group.applyFunc(function(neuron){
        if(bestOutput < neuron.output){
            bestOutput = neuron.output;
            winner = neuron;
        }
    });
    
    group.applyFunc(function(neuron){
        var disY = Math.abs(neuron.pos[0] - winner.pos[0]);
        var disX = Math.abs(neuron.pos[1] - winner.pos[1]);
        var distanceRate = distancePara / (distancePara + disY * disY + disX * disX);  //less distance is better
        
        //
        var idarray = Object.keys(neuron.from);
        var len = idarray.length
        for(var i = 0; i < len; i++){
            var id = idarray[i]; //fromNeuronId
            var weight = neuron.from[ id ];
            var value = nn.neuronHash[ id ].output;
            
            // update weight
            neuron.from[ id ] += learningRate * (value - weight) * distanceRate;
            // copy weight to next neuron info
            nn.neuronHash[ id ].to[ neuron.id ] = neuron.from[ id ];
        }
    });
};





//--------------------------------------------------------
// Back Propagetion
//--------------------------------------------------------
//BP fire Process
var fireBp = function(group){
    group.applyFunc(function(neuron){
        neuron.fire();
    });
};

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






//--------------------------------------------------------
// Sparce Back Propagetion (for AutoEncorder) = BP + Sparse Penalty
//--------------------------------------------------------
//SparceBP fire Process
var fireRectifiedBp = function(group){
    group.applyFunc(function(neuron){
        neuron.fire();
    });
};

//SparceBP learn Process
var learnSparseBp = function(group){
    var nn = group.parent;
    var learningRate = 0.4;
    var momentum = 0.2;
        
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
        
        //add Sparseness penalty 
        var beta = 0.1;
        var ts = neuron.getTimeSparseValue();
        var desiredTs = 0.05;
        var penalty = beta * ( - (1-desiredTs) / (1-ts) + (desiredTs / ts));
        neuron.delta = (error + penalty) * neuron.output * (1 - neuron.output);
        
        //adjustWeights
        var idarray = Object.keys(neuron.from);
        var len = idarray.length
        for(var i = 0; i < len; i++){
            var id = idarray[i]; //fromNeuronId
            var weight = neuron.from[ id ];
            
            var incoming = nn.neuronHash[ id ].output;
            var lastChange = 0;
            
            if(typeof(neuron.changes[ id ]) !== 'undefined'){
                lastChange = neuron.changes[ id ];
            }
            var change = (learningRate * neuron.delta * incoming) + (momentum * lastChange);
            neuron.changes[ id ] = change;
            
            //update weight
            neuron.from[ id ] += change;
            // copy weight to next neuron info
            nn.neuronHash[ id ].to[ neuron.id ] = neuron.from[ id ];
        }
        neuron.bias += learningRate * neuron.delta;
    });
};
