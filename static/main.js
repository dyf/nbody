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
    return v / 500000.0;
}

function reset_simulation() {
    $.getJSON('/reset', function () {
        running = false;
        update_bodies(render);
    });
}

function toggle_simulation(cb) {
    running = !running;
    $.getJSON('/toggle?dt='+dt().toString(), function() {
        
        if (cb)
            cb();
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
    fetch_bodies(function(n, d, p, r) {
        if (p) {
            for (var i = 0; i < n; i++) {
                var object = spheres[i];

                object.scale.set(r[i]*100, r[i]*100, r[i]*100);
                object.position.set((p[d*i]   - 0.5)*100,
                                    (p[d*i+1] - 0.5)*100,
                                    (p[d*i+2] - 0.5)*100);
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
            
            callback(Math.round(p[0]), Math.round(p[1]),
                     p.slice(2, p[0]*p[1]+2),
                     p.slice(p[0]*p[1]+2));
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
    camera.position.set(0,0,100);
    
    scene = new THREE.Scene();

    controls = new THREE.TrackballControls( camera, container );
    controls.update();

    scene.background = new THREE.Color( 0xf0f0f0 );

    var light = new THREE.DirectionalLight( 0xffffff, 1 );
    light.position.set( 1, 1, 1 ).normalize();
    scene.add( light );

    var geometry = new THREE.SphereBufferGeometry( 1, 32, 32 );

    spheres = [];
    fetch_bodies(function(n, d, p, r) {
        for (var i = 0; i < n; i++) {
            var object = new THREE.Mesh( geometry, new THREE.MeshLambertMaterial( { color: Math.random() * 0xffffff } ) );
            object.scale.set(r[i]*100, r[i]*100, r[i]*100);
            object.position.set((p[d*i]   - 0.5)*100,
                                (p[d*i+1] - 0.5)*100,
                                (p[d*i+2] - 0.5)*100);
            
            spheres.push(object);
            scene.add(object);
        }

        renderer = new THREE.WebGLRenderer();
        renderer.setPixelRatio( window.devicePixelRatio );
        renderer.setSize( window.innerWidth, window.innerHeight );
        
        container.appendChild(renderer.domElement);

        animate();
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
