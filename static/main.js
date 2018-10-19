var camera, scene, renderer, container, controls;
var spheres = null;
var running = false;


$(function() {
    $("#toggle_button").on('click', toggle_simulation);
    $("#reset_button").on('click', reset_simulation);
    $("#dt_slider").on('input change', set_simulation_dt);

    init();
});

function dt() {
    v = $("#dt_slider").val();
    return v / 100000.0;
}

function reset_simulation() {
    $.getJSON('/reset', function () {
        running = false;
        update_bodies(render);
    });
}

function toggle_simulation() {
    $.getJSON('/toggle?dt='+dt().toString(), function() {
        running = !running;
    });
}

function set_simulation_dt() {
    if (!running) {
        console.log("starting...");
        toggle_simulation(function() {
            $.getJSON('/set?dt='+dt().toString());
        });
    } else {
        $.getJSON('/set?dt='+dt().toString());
    }
}

function update_bodies(callback) {
    fetch_bodies(function(p) {
        if (p) {
            var oi = 0;
            for (var i = 0; i < p.length; i+=3) {
                var object = spheres[oi];
                
                object.position.x = (p[i] - 0.5)*100;
                object.position.y = (p[i+1] - 0.5)*100;
                object.position.z = (p[i+2] - 0.5)*100;
                oi += 1;
            }
        } else {
            console.log("no bodies...");
        }
            
        callback();
    });
}

function fetch_bodies(callback) {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/bodies', true);
    xhr.responseType = 'arraybuffer'
    xhr.onload = function(event) {
        var buf = xhr.response;
        if (buf) {
            var p = new Float32Array(buf);
            callback(p);
        } else {
            callback();
        }
    }
    xhr.send(null);
}

function init() {
    container = document.createElement( 'div' );
    document.body.appendChild( container );
    
    camera = new THREE.PerspectiveCamera( 70, window.innerWidth / window.innerHeight, 1, 10000 );
    camera.position.set(200,0,0);
    
    scene = new THREE.Scene();

    controls = new THREE.TrackballControls( camera, container );
    controls.update();

    scene.background = new THREE.Color( 0xf0f0f0 );

    var light = new THREE.DirectionalLight( 0xffffff, 1 );
    light.position.set( 1, 1, 1 ).normalize();
    scene.add( light );

    var geometry = new THREE.SphereBufferGeometry( 5, 32, 32 );

    spheres = [];
    fetch_bodies(function(p) {
        for (var i = 0; i < p.length; i+=3) {
            var object = new THREE.Mesh( geometry, new THREE.MeshLambertMaterial( { color: Math.random() * 0xffffff } ) );

            object.position.x = (p[i] - 0.5)*100;
            object.position.y = (p[i+1] - 0.5)*100;
            object.position.z = (p[i+2] - 0.5)*100;

            spheres.push(object);
            scene.add(object);
        }

        renderer = new THREE.WebGLRenderer();
        renderer.setPixelRatio( window.devicePixelRatio );
        renderer.setSize( window.innerWidth, window.innerHeight );
        
        container.appendChild(renderer.domElement);

        animate()


    });    
}

function render() {
    controls.update();
    renderer.render( scene, camera );
}

function animate() {
    requestAnimationFrame( animate );
    
    if (running) {
        update_bodies(render);
    } else {
        render();
    }    
}
