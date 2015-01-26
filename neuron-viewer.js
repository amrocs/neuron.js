//************************************************
//  Neuron.js 
//    Weight and Firing Value Viewer
//
//************************************************
"use strict";
//--------------------------------------------------------
//Create Viewer For Group(Canvas, Labels, and so on) 
var createGroupViewer = function(argOptions){
    var viewer = {
        group: null,
        parentId: null,
        el: {},
        contexts: {},
        fireView: false,
        weightView: false,
        pixelRate: 1,
    };
    viewer = $.extend(viewer, argOptions);
    
    //--------------------------------------------------------
    // Weight Viewer
    //--------------------------------------------------------
    //Create Canvas For Weight View
    viewer.createWeightView = function(){
        //Weight View Container
        viewer.el["weightView"] = $("<div/>")
            .addClass("weight_viewer")
            .css("float", "left")
            .css("margin", 5)
            .appendTo(viewer.el["view_container"]);
        
        //label
        var fromLayerNames = [];
        $.each(viewer.group.from, function(name, val){
            fromLayerNames.push(name);
        });
        $("<div/>")
            .addClass("canvas_label")
            .html("weight (from " + fromLayerNames.join(", ") + ")")
            .appendTo(viewer.el["weightView"]);
        
        //Canvas
        viewer.el["canvas_weight"] = $("<canvas/>")
            .attr("width", 1)
            .attr("height", 1)
            .addClass("canvas_weight")
            .appendTo(viewer.el["weightView"]);
        viewer.contexts["canvas_weight"] = viewer.el["canvas_weight"].get(0).getContext('2d');
        
        //Canvas for pre-rendering
        viewer.el["canvas_weight_prerender"] = $("<canvas/>")
            .attr("width", 1)
            .attr("height", 1)
            .addClass("canvas_weight");
        viewer.contexts["canvas_weight_prerender"] = viewer.el["canvas_weight_prerender"].get(0).getContext('2d');
    };
    
    //--------------------------------------------------------
    //Update Canvas For Weight View
    viewer.updateWeightCanvasSize = function(){
        var group = viewer.group;
        var pixelRate = viewer.pixelRate;
        
        //Calculate width and height
        var h = group.connectionH * pixelRate;
        if(Array.isArray(group.neurons)){
            h = group.connectionH * group.neurons.length * pixelRate;
        }
        var w = group.connectionW * pixelRate;
        if(Array.isArray(group.neurons[0])){
            w = group.connectionW * group.neurons[0].length * pixelRate;
        }
        
        //Update size
        viewer.el["canvas_weight"]
            .attr("width", w)
            .attr("height", h);
        viewer.el["canvas_weight_prerender"]
            .attr("width", w)
            .attr("height", h);
    };
    
    //--------------------------------------------------------
    //Draw Weights of Neurons of Group
    viewer.drawWeight = function(){
        var group = viewer.group;
        var pixelRate = viewer.pixelRate;
        var nn = group.parent;
        var ctx = viewer.contexts["canvas_weight_prerender"];
                    
        group.applyFunc(function(neuron){
            var idarray = Object.keys(neuron.from);
            var len = idarray.length
            for(var i = 0; i < len; i++){
                var id = idarray[i];
                var weight = neuron.from[ id ];
                var expandedWeight = Math.floor(weight * 255);
                
                if(0 < expandedWeight){
                    var colorR = expandedWeight;
                    var colorG = expandedWeight;
                    var colorB = 0;
                }else{
                    var colorR = 0;
                    var colorG = -expandedWeight;
                    var colorB = -expandedWeight;
                }
                
                var pos = nn.neuronHash[ id ].pos;
                ctx.fillStyle = 'rgb(' + colorR + ', ' + colorG + ', ' + colorB + ')';
                ctx.fillRect(
                    (group.connectionW * neuron.pos[1] + pos[1]) * pixelRate,
                    (group.connectionH * neuron.pos[0] + pos[0]) * pixelRate,
                    pixelRate,
                    pixelRate);
            }
        });
    };
    
    //--------------------------------------------------------
    //Render Weight View
    viewer.renderWeight = function(){
        viewer.updateWeightCanvasSize();
        viewer.drawWeight();
        viewer.contexts["canvas_weight"].drawImage( viewer.el["canvas_weight_prerender"].get(0), 0, 0);
        requestAnimationFrame( viewer.renderWeight );
    };
    
    //--------------------------------------------------------
    // Fire Viewer
    //--------------------------------------------------------
    //Create Canvas For Fire View
    viewer.createFireView = function(){
        //Fire View Container
        viewer.el["fireView"] = $("<div/>")
            .addClass("fire_viewer")
            .css("float", "left")
            .css("margin", 5)
            .appendTo(viewer.el["view_container"]);
        
        //Label
        $("<div/>")
            .addClass("canvas_label")
            .html("fire")
            .appendTo(viewer.el["fireView"]);
        
        //Canvas
        viewer.el["canvas_fire"] = $("<canvas/>")
            .attr("width", 1)
            .attr("height", 1)
            .addClass("canvas_fire")
            .appendTo(viewer.el["fireView"]);
        viewer.contexts["canvas_fire"] = viewer.el["canvas_fire"].get(0).getContext('2d');
        
        //Canvas for pre-rendering
        viewer.el["canvas_fire_prerender"] = $("<canvas/>")
            .attr("width", 1)
            .attr("height", 1)
            .addClass("canvas_fire");
        viewer.contexts["canvas_fire_prerender"] = viewer.el["canvas_fire_prerender"].get(0).getContext('2d');
    };
    
    //--------------------------------------------------------
    //Update Canvas For Fire View
    viewer.updateFireCanvasSize = function(){
        var group = viewer.group;
        var pixelRate = viewer.pixelRate;
        
        //Calculate width and height
        var h = group.connectionH * pixelRate;
        if(Array.isArray(group.neurons)){
            h = group.connectionH * group.neurons.length * pixelRate;
        }
        var w = group.connectionW * pixelRate;
        if(Array.isArray(group.neurons[0])){
            w = group.connectionW * group.neurons[0].length * pixelRate;
        }
        
        //Update size
        viewer.el["canvas_fire"]
            .attr("width", w)
            .attr("height", h);
        viewer.el["canvas_fire_prerender"]
            .attr("width", w)
            .attr("height", h);
    };
    
    //--------------------------------------------------------
    //Draw Fire of Neurons of Group
    viewer.drawFire = function(){
        var group = viewer.group;
        var pixelRate = viewer.pixelRate;
        var ctx = viewer.contexts["canvas_fire_prerender"];
        
        group.applyFunc(function(neuron){
            var expandedWeight = Math.floor( neuron.output * 255 );
            if(0 < expandedWeight){
                var colorR = expandedWeight;
                var colorG = expandedWeight;
                var colorB = 0;
            }else{
                var colorR = 0;
                var colorG = -expandedWeight;
                var colorB = -expandedWeight;
            }
            
            ctx.fillStyle = 'rgb(' + colorR + ', ' + colorG + ', ' + colorB + ')';
            ctx.fillRect(
                group.connectionW * neuron.pos[1] * pixelRate,
                group.connectionH * neuron.pos[0] * pixelRate,
                group.connectionW * pixelRate,
                group.connectionH * pixelRate
            );
        });
    };
    
    //--------------------------------------------------------
    //Render Fire View
    viewer.renderFire = function(){
        viewer.updateFireCanvasSize();
        viewer.drawFire();
        viewer.contexts["canvas_fire"].drawImage( viewer.el["canvas_fire_prerender"].get(0), 0, 0 );
        requestAnimationFrame( viewer.renderFire );
    };
    
    //--------------------------------------------------------
    //Inital Process
    viewer.init = function(){
        //whole view area
        viewer.el["container"] = $("<div/>")
            .addClass("group_container")
            .appendTo($("#" + viewer.parentId));
        
        //group properties (group name,...)
        $("<div/>")
            .addClass("group_name")
            .html(viewer.group.id + " (" + viewer.group.counts.join(" x ") + ")")
            .css("font-weight", "bold")
            .appendTo(viewer.el["container"]);
        
        //canvas area
        viewer.el["view_container"] = $("<div/>")
            .addClass("view_container")
            .appendTo(viewer.el["container"]);
            
        if(viewer.weightView){
            viewer.createWeightView();
            viewer.renderWeight();
        }
        if(viewer.fireView){
            viewer.createFireView();
            viewer.renderFire();
        }
    };
    
    //--------------------------------------------------------
    viewer.init();
    return viewer;
};
