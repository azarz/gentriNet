<?php
	$IMAGES_DIR = "../reviewcouples/";

	$index = $_GET["index"];

	$back = $_GET["back"];

	if($back == 1){
		// load the data and delete the line from the array 
		$lines = file('train_review.csv');
		if(sizeof($lines)>1){ 
			$last = sizeof($lines) - 1 ; 
			unset($lines[$last]); 

			// write the new data to the file 
			$fp = fopen('train_review.csv', 'w'); 
			fwrite($fp, implode('', $lines)); 
			fclose($fp);
		}
	}

	// Searching if the id already exists in csv
	$search      = $index . ',';
	$lines       = file('train_review.csv');
	$line_number = false;

	while (list($key, $line) = each($lines) and !$line_number) {
   		$line_number = (strpos($line, $search) !== FALSE);
	}

	// If it exists, we go to another folder
	if($line_number){
			$result = array(
				'images' => 1000);
	}

	else{

		$path = glob($IMAGES_DIR . $index . '.jpg');

		$result = array(
					'images' => array($path));
	}

	echo json_encode($result);
?>
