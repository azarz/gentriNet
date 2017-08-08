var form;

var index_folder = 0;
var index_image = 0;

var old_folder_index = 0;
var old_image_index = 0;

var back = 0;

var lat = 0;
var lon = 0;
var date1 = 0;
var date2 = 1;

var regex = /\/[0-9][0-9][0-9][0-9]/;

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
    if (index_image > 0){
        index_image--;
        addImages();
    } else{
        index_folder = old_folder_index;
        index_image = old_image_index;
        addImages();
    }
    back = 0;
}

function validateForm(event){
    /**
    When validating the form
    */
	event.preventDefault();
    var xhr = new XMLHttpRequest();

    var yn = document.querySelector('input[name="yn"]:checked').value;
    xhr.open('GET', 'write_db.php?id=' + String(index_folder) + "-" + String(index_image) + "-" + String(index_image + 1) + '&lat=' + lat + '&lon=' + lon + '&date1=' + date1 + '&date2=' + date2 + '&YN=' + yn);
    xhr.send();
    index_image++;
    addImages();
}

function addImages(){

    var xhr = new XMLHttpRequest();
    xhr.addEventListener('readystatechange', function(){
        if(xhr.readyState == 4 && xhr.status == 200){
            var response = xhr.responseText;
            var responseJSON = JSON.parse(response);

            lat = responseJSON.lat;
            lon = responseJSON.lon;

            if(lat != 1000){

                imagespaths = responseJSON.images;

                date1 = regex.exec(imagespaths[0])[0].replace("/","");
                date2 = regex.exec(imagespaths[1])[0].replace("/","");

                var date1html = document.getElementById('date1');
                var date2html = document.getElementById('date2');
                date1html.innerHTML = date1
                date2html.innerHTML = date2

                var imagesDiv = document.getElementById('images');

                while(imagesDiv.firstChild){
                    imagesDiv.removeChild(imagesDiv.firstChild);
                }

                for(i=0; i < imagespaths.length; i++){
                    var newImg = document.createElement("img");
                    newImg.setAttribute('src',imagespaths[i]);
                    imagesDiv.appendChild(newImg);
                }
            } else{
                old_image_index = index_image-1;
                index_image = 0;
                old_folder_index = index_folder;
                index_folder = responseJSON.indexfolder;
                addImages();
            }


        }
    });

    // On envoie la requête au serveur
    xhr.open('GET', 'get_images.php?index=' + index_folder + '&index_image=' + index_image + '&back=' + back);
    xhr.send();
}