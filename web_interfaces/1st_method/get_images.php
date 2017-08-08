<?php
	$IMAGES_DIR = "../trainottawa/";
	$dirs = array_filter(glob($IMAGES_DIR .'*'), 'is_dir');

	$index = $_GET["index"];
	$imagenumber = $_GET["index_image"];
	$imagenumber2 = $imagenumber + 1;

	$back = $_GET["back"];

	if($back == 1){
		// load the data and delete the line from the array 
		$lines = file('train.csv');
		if(sizeof($lines)>1){ 
			$last = sizeof($lines) - 1 ; 
			unset($lines[$last]); 

			// write the new data to the file 
			$fp = fopen('train.csv', 'w'); 
			fwrite($fp, implode('', $lines)); 
			fclose($fp);
		}
	}


	// Searching if the id already exists in csv
	$search      = $index . '-' . $imagenumber . '-' . $imagenumber2;
	$lines       = file('train.csv');
	$line_number = false;

	while (list($key, $line) = each($lines) and !$line_number) {
   		$line_number = (strpos($line, $search) !== FALSE);
	}

	// If it exists, we go to another folder
	if($line_number){
	   		$cnt = count($dirs);
			$folderindex = rand(0, $cnt);

			$result = array(
				'lat' => 1000,
				'lon' => 1000,
				'images' => 1000,
				'indexfolder' => $folderindex);
	}

	else{

		$dir = $dirs[$index];

		$latlonstr = substr($dir, -20);

		$latlon = explode(',', $latlonstr);

		$images = glob($dir . '/*.jpg');

		if(count($images) > $imagenumber + 1){
			$image = array($images[$imagenumber], $images[$imagenumber + 1]);
			$result = array(
				'lat' => $latlon[0],
				'lon' => $latlon[1],
				'images' => $image);

		} else{
			$cnt = count($dirs);
			$folderindex = rand(0, $cnt);

			$result = array(
				'lat' => 1000,
				'lon' => 1000,
				'images' => 1000,
				'indexfolder' => $folderindex);
		}
	}

	echo json_encode($result);
?>
