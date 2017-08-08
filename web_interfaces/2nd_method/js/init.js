var form;

var index_image1 = 0;
var index_image2 = 1;

var old_image1_index = 0;
var old_image2_index = 1;

var back = 0;

var pano1 = '';
var pano2 = '';

window.onload = init;


function init(){
    /**
    Initialisation
    */
    addImages();
    form = document.getElementById("form");
    form.addEventListener('submit', validateForm);

    goback = document.getElementById("goback");
    goback.addEventListener('click', goBack);

}

function goBack(event){
    back = 1;
    index_image1 = old_image1_index;
    index_image2 = old_image2_index;
    addImages();
    back = 0;
}

function validateForm(event){
    /**
    When validating the form
    */
    event.preventDefault();
    var xhr = new XMLHttpRequest();
    xhr.addEventListener('readystatechange', function(){
        if(xhr.readyState == 4 && xhr.status == 200){
            old_image1_index = index_image1;
            old_image2_index = index_image2;

            var response = xhr.responseText;
            var responseJSON = JSON.parse(response);

            index_image1 = responseJSON.id1;
            index_image2 = responseJSON.id2;
        
        }
    });

    var winner = document.querySelector('input[name="yn"]:checked').value;
    xhr.open('GET', 'write_db.php?id1=' + index_image1 + '&id2=' + index_image2 + '&pano1=' + pano1 + '&pano2=' + pano2 + '&winner=' + winner, false);
    xhr.send();

    addImages();
}

function addImages(){

    var xhr = new XMLHttpRequest();
    xhr.addEventListener('readystatechange', function(){
        if(xhr.readyState == 4 && xhr.status == 200){
            var response = xhr.responseText;
            var responseJSON = JSON.parse(response);

            pano1 = responseJSON.pano1;
            pano2 = responseJSON.pano2;

            var imagespaths = responseJSON.images;

            var imagesDiv = document.getElementById('images');

            while(imagesDiv.firstChild){
                imagesDiv.removeChild(imagesDiv.firstChild);
            }

            for(i=0; i < imagespaths.length; i++){
                var newImg = document.createElement("img");
                newImg.setAttribute('src',imagespaths[i]);
                imagesDiv.appendChild(newImg);
            }



        }
    });

    // On envoie la requête au serveur
    xhr.open('GET', 'get_images.php?index1=' + index_image1 + '&index2=' + index_image2 + '&back=' + back);
    xhr.send();
}