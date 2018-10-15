$(function() {
    init();
    animate();
});

var camera, scene, renderer, container, controls;
var spheres = null;

function update_bodies(callback) {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/step?dt=0.1', true);
    xhr.responseType = 'arraybuffer'
    xhr.onload = function(event) {
        var buf = xhr.response;
        if (buf) {
            var p = new Float32Array(buf);

            var oi = 0;
            for (var i = 0; i < p.length; i+=3) {
                var object = spheres[oi];
                
                object.position.x = (p[i] - 0.5)*100;
                object.position.y = (p[i+1] - 0.5)*100;
                object.position.z = (p[i+2] - 0.5)*100;
                oi += 1;
            }
            callback();
        } else {
            console.log(event);
        }
    }
    xhr.send(null);
}

function init() {
    container = document.createElement( 'div' );
    document.body.appendChild( container );
    
    camera = new THREE.PerspectiveCamera( 70, window.innerWidth / window.innerHeight, 1, 10000 );
    camera.position.set(1000,0,0);
    
    scene = new THREE.Scene();

    controls = new THREE.TrackballControls( camera );
    controls.update();

    scene.background = new THREE.Color( 0xf0f0f0 );

    var light = new THREE.DirectionalLight( 0xffffff, 1 );
    light.position.set( 1, 1, 1 ).normalize();
    scene.add( light );

    var geometry = new THREE.SphereBufferGeometry( 5, 32, 32 );

    spheres = [];
    $.getJSON("/bodies", function(data) {
        for (var i = 0; i < data.length; i++) {
            var object = new THREE.Mesh( geometry, new THREE.MeshLambertMaterial( { color: Math.random() * 0xffffff } ) );
            var p = data[i];

            object.position.x = (p[0] - 0.5)*100;
            object.position.y = (p[1] - 0.5)*100;
            object.position.z = (p[2] - 0.5)*100;

            spheres.push(object);
            scene.add(object);
        }
    })

    renderer = new THREE.WebGLRenderer();
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( window.innerWidth, window.innerHeight );

    container.appendChild(renderer.domElement);
}

function animate() {
    update_bodies(function() {    
        requestAnimationFrame( animate );
        controls.update();
        renderer.render( scene, camera );
    });
}
