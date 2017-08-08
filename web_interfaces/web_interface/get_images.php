<?php
	$IMAGES_DIR = "../voteimages/";
	$images = glob($IMAGES_DIR .'*.jpg');

	$imagenumber1 = $_GET["index1"];
	$imagenumber2 = $_GET["index2"];

	$back = $_GET["back"];

	if($back == 1){
		// load the data and delete the line from the array 
		$lines = file('votes.csv');
		if(sizeof($lines)>1){ 
			$last = sizeof($lines) - 1 ; 
			unset($lines[$last]); 

			// write the new data to the file 
			$fp = fopen('votes.csv', 'w'); 
			fwrite($fp, implode('', $lines)); 
			fclose($fp);
		}
	}

	$img1 = $images[$imagenumber1];
	$img2 = $images[$imagenumber2];

	$panojpg1 = substr($img1, -26);
	$panojpg2 = substr($img2, -26);

	$pano1 = explode('.', $panojpg1);
	$pano2 = explode('.', $panojpg2);

	$image = array($img1, $img2);
	$result = array(
		'pano1' => $pano1[0],
		'pano2' => $pano2[0],
		'images' => $image);

	echo json_encode($result);
?>