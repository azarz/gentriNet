var form;

var index = 0;

var old_index = 0;

var back = 0;

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
    if (index > 0){
        index = old_index;
        addImages();
    } else{
        index = 0;
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
    xhr.addEventListener('readystatechange', function(){
        if(xhr.readyState == 4 && xhr.status == 200){
            old_index = index;

            var response = xhr.responseText;
            var responseJSON = JSON.parse(response);

            index++;
        
        }
    });

    var yn = document.querySelector('input[name="yn"]:checked').value;
    xhr.open('GET', 'write_db.php?id=' + String(index) + '&YN=' + yn);
    xhr.send();
    index++;
    addImages();
}

function addImages(){

    var xhr = new XMLHttpRequest();
    xhr.addEventListener('readystatechange', function(){
        if(xhr.readyState == 4 && xhr.status == 200){
            var response = xhr.responseText;
            var responseJSON = JSON.parse(response);

            imagespaths = responseJSON.images;

            if(imagespaths != 1000){

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
                index++;
                addImages();
            }

        }
    });

    // On envoie la requête au serveur
    xhr.open('GET', 'get_images.php?index=' + index + '&back=' + back);
    xhr.send();
}