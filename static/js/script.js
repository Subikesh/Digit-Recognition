var clickX = new Array();
var clickY = new Array();
var clickDrag = new Array();
var canvasHeight = 420;
var canvasWidth = 420;
var paint;

var canvas = document.getElementById('canvas-draw');
var context = canvas.getContext("2d");

// Get the image drawn in the canvas and send it as test input for prediction
$('#get-image').click(function imageFromCanvas(e) {
    document.getElementById('message').innerHTML = '<img src="static/load.gif">';
    img = context.getImageData(0, 0, 420, 420);
    var imgData = new Array();
    for (let index = 0; index < img.data.length; index++) {
        if (index%4 == 0) {
            imgData.push(img.data[index]);
        }
    }
    $.ajax({
        type: 'POST',
        url: '',
        data: JSON.stringify({
            'imgData': imgData,
            'message': "This is a sample message"
        }),
        success: function(data) {
            $('#message').html(data);
        }
    });
});

// To clear the canvas
$('#clear').click(function(e) {
    context.clearRect(0, 0, canvasWidth, canvasHeight);
    resetPointer();
    context.beginPath();
});

// Starts the line when mouse starts dragging inside canvas
$('#canvas-draw').on("mousedown touchstart", function(e) {
    paint = true;
    // addclick(mouseX, mouseY, paint);
    // redraw();
});

// Continue line for mouse drag
$('#canvas-draw').on("mousemove", function(e) {
    if(paint) {
        mouseX = e.pageX - this.offsetLeft;
        mouseY = e.pageY - this.offsetTop;
        addclick(mouseX, mouseY, paint);
        redraw();
    }
});

// Continue line for touch move event
$('#canvas-draw').on("touchmove", function(e) {
    if(paint) {
        mouseX = e.originalEvent.touches[0].pageX - this.offsetLeft;
        mouseY = e.originalEvent.touches[0].pageY - this.offsetTop;
        addclick(mouseX, mouseY, paint);
        redraw();
    }
});

// set paint to false if mouse stops dragging
$('#canvas-draw').on("mouseup touchend", function(e) {
    paint = false;
    resetPointer();
});

// If marker goes off the canvas
$('#canvas-draw').on("mouseleave touchcancel touchleave", function(e) {
    paint = false;
    resetPointer();
});

// Adds page coordinates to the list for drawing
function addclick(x, y, drag) {
    clickX.push(x);
    clickY.push(y);
    clickDrag.push(drag);
}

// Main function to refresh the canvas draw the line
function redraw() {
    context.clearRect(0, 0, canvasWidth, canvasHeight);
    context.fillStyle = "black"
    context.fillRect(0, 0, canvasWidth, canvasHeight);
    
    context.strokeStyle = "#ffffff";
    context.lineJoin = "round";
    context.lineWidth = 17;
    context.shadowColor = "#ffffff";
    context.shadowOffsetX = 0;
    context.shadowOffsetY = 0;
    context.shadowBlur = 10;
    context.lineCap = "round";

    for (let i = 0; i < clickX.length; i++) {
        if (i && clickDrag[i]) {
            context.moveTo(clickX[i-1], clickY[i-1]);
        } else {
            context.moveTo(clickX[i]-1, clickY[i]);
        }
        context.lineTo(clickX[i], clickY[i])
        context.closePath();
        context.stroke();
    }
}

function resetPointer() {
    clickX = [];
    clickY = [];
    clickDrag = [];
}